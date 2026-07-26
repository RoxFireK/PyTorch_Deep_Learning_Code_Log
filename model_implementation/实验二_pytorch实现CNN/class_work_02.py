# import library
import torch
from torch import nn
import torchvision
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


# 超参数配置
class Config:
    # 数据参数
    BATCH_SIZE = 64

    # 训练参数
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # 模型参数
    DROPOUT_RATE = 0.3
    USE_BATCH_NORM = True

    # 学习率调度
    USE_SCHEDULER = True
    SCHEDULER_STEP = 10
    SCHEDULER_GAMMA = 0.5

    # 数据增强
    USE_AUGMENTATION = True

    # 设备
    device = "cuda:0" if torch.cuda.is_available() else "cpu"


config = Config()


# 数据预处理（包含数据增强）
def get_transform(is_train=True):
    transforms = [torchvision.transforms.ToTensor(),
                  torchvision.transforms.Normalize(mean=[0.5], std=[0.5])]

    if is_train and config.USE_AUGMENTATION:
        # 添加数据增强
        augmentations = [
            torchvision.transforms.RandomRotation(10),
            torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ]
        transforms = augmentations + transforms

    return torchvision.transforms.Compose(transforms)


# 改进的模型结构
class ImprovedNet(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.3, use_batch_norm=True):
        super(ImprovedNet, self).__init__()
        self.use_batch_norm = use_batch_norm

        # 第一个卷积块
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate)
        )

        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate)
        )

        # 第三个卷积块
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate)
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x


# 标签平滑交叉熵损失
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        smooth_target = torch.zeros_like(pred).fill_(self.smoothing / (n_classes - 1))
        smooth_target.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        log_prob = nn.functional.log_softmax(pred, dim=1)
        loss = (-smooth_target * log_prob).sum(dim=1).mean()
        return loss


# 训练函数
def train_epoch(net, train_loader, optimizer, lossF, device, epoch, writer):
    net.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    processBar = tqdm(train_loader, desc=f'Epoch {epoch}/{config.EPOCHS}')

    for step, (trainImgs, labels) in enumerate(processBar):
        trainImgs = trainImgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(trainImgs)
        loss = lossF(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        predictions = torch.argmax(outputs, dim=1)
        accuracy = torch.sum(predictions == labels) / labels.shape[0]

        train_loss += loss.item()
        train_correct += torch.sum(predictions == labels).item()
        train_total += labels.shape[0]

        processBar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{accuracy.item():.4f}'
        })

        if step % 100 == 0:
            global_step = (epoch - 1) * len(train_loader) + step
            writer.add_scalar('Batch/Train_Loss', loss.item(), global_step)
            writer.add_scalar('Batch/Train_Accuracy', accuracy.item(), global_step)

    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_correct / train_total

    return avg_train_loss, avg_train_acc


# 测试函数
def test_epoch(net, test_loader, lossF, device, epoch, writer):
    net.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for testImgs, labels in test_loader:
            testImgs = testImgs.to(device)
            labels = labels.to(device)
            outputs = net(testImgs)
            loss = lossF(outputs, labels)

            predictions = torch.argmax(outputs, dim=1)
            test_loss += loss.item()
            test_correct += torch.sum(predictions == labels).item()
            test_total += labels.shape[0]

    avg_test_loss = test_loss / len(test_loader)
    avg_test_acc = test_correct / test_total

    return avg_test_loss, avg_test_acc


# 主函数
def main():
    # 数据加载（不使用多进程）
    print("加载数据...")
    trainData = torchvision.datasets.MNIST(
        './data', train=True, transform=get_transform(is_train=True), download=True
    )

    testData = torchvision.datasets.MNIST(
        './data', train=False, transform=get_transform(is_train=False)
    )

    # 重要：设置 num_workers=0 避免多进程问题
    trainDataLoader = torch.utils.data.DataLoader(
        dataset=trainData, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0
    )

    testDataLoader = torch.utils.data.DataLoader(
        dataset=testData, batch_size=config.BATCH_SIZE, num_workers=0
    )

    # 创建模型
    net = ImprovedNet(
        num_classes=10,
        dropout_rate=config.DROPOUT_RATE,
        use_batch_norm=config.USE_BATCH_NORM
    ).to(config.device)

    print("模型结构：")
    print(net)
    print(f"\n模型参数量：{sum(p.numel() for p in net.parameters()):,}")
    print(f"使用设备: {config.device}")

    # 损失函数
    lossF = LabelSmoothingCrossEntropy(smoothing=0.1)

    # 优化器
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # 学习率调度器
    scheduler = None
    if config.USE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.SCHEDULER_STEP,
            gamma=config.SCHEDULER_GAMMA
        )

    # TensorBoard
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'runs/mnist_experiment_{current_time}'
    writer = SummaryWriter(log_dir)

    # 记录模型结构
    dummy_input = torch.randn(1, 1, 28, 28).to(config.device)
    writer.add_graph(net, dummy_input)

    # 训练历史
    history = {
        'Train Loss': [], 'Train Accuracy': [],
        'Test Loss': [], 'Test Accuracy': []
    }

    best_accuracy = 0
    best_model_path = f'best_model_{current_time}.pth'

    print("\n开始训练...")
    print(f"TensorBoard日志目录: {log_dir}")
    print(f"运行 'tensorboard --logdir={log_dir}' 查看可视化结果\n")
    #tensorboard --logdir=C:\Users\Admin\Desktop\deep_learning\src\CNN\runs\mnist_experiment_20260416_185516

    for epoch in range(1, config.EPOCHS + 1):
        # 训练
        avg_train_loss, avg_train_acc = train_epoch(
            net, trainDataLoader, optimizer, lossF, config.device, epoch, writer
        )

        # 测试
        avg_test_loss, avg_test_acc = test_epoch(
            net, testDataLoader, lossF, config.device, epoch, writer
        )

        # 保存历史记录
        history['Train Loss'].append(avg_train_loss)
        history['Train Accuracy'].append(avg_train_acc)
        history['Test Loss'].append(avg_test_loss)
        history['Test Accuracy'].append(avg_test_acc)

        # 学习率调度
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar('Training/Learning_Rate', current_lr, epoch)

        # 记录到TensorBoard
        writer.add_scalars('Loss', {
            'Train': avg_train_loss,
            'Test': avg_test_loss
        }, epoch)

        writer.add_scalars('Accuracy', {
            'Train': avg_train_acc,
            'Test': avg_test_acc
        }, epoch)

        # 记录模型参数分布
        if epoch % 5 == 0:
            for name, param in net.named_parameters():
                if param.requires_grad:
                    writer.add_histogram(f'Parameters/{name}', param.data, epoch)
                    if param.grad is not None:
                        writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

        # 保存最佳模型
        if avg_test_acc > best_accuracy:
            best_accuracy = avg_test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy': avg_test_acc,
                'test_loss': avg_test_loss,
            }, best_model_path)
            print(f"\n✓ 保存最佳模型，准确率: {avg_test_acc:.4f}")

        # 打印epoch结果
        print(f"\nEpoch {epoch}/{config.EPOCHS}")
        print(f"训练 - Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}")
        print(f"测试 - Loss: {avg_test_loss:.4f}, Acc: {avg_test_acc:.4f}")
        print(f"最佳准确率: {best_accuracy:.4f}\n")

    writer.close()

    print("训练完成！")
    print(f"最佳模型保存在: {best_model_path}")
    print(f"最佳测试准确率: {best_accuracy:.4f}")

    # 可视化训练曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['Train Loss'], label='Train Loss', marker='o')
    ax1.plot(history['Test Loss'], label='Test Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['Train Accuracy'], label='Train Accuracy', marker='o')
    ax2.plot(history['Test Accuracy'], label='Test Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.show()

    print("\n最终结果统计：")
    print(f"最终训练准确率: {history['Train Accuracy'][-1]:.4f}")
    print(f"最终测试准确率: {history['Test Accuracy'][-1]:.4f}")
    print(f"最佳测试准确率: {best_accuracy:.4f}")


# 重要：添加这个保护
if __name__ == '__main__':
    main()