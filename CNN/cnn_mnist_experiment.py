import torch
from torch import nn
import torchvision
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

# 设置中文字体，避免 matplotlib 显示乱码
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ════════════════════════════════════════════════════════════
# 超参数对比实验配置
# 共设计 3 组实验，每组仅改变 1~2 个超参数，保持其余不变，
# 以便观察单一变量对训练效果的影响。
# ════════════════════════════════════════════════════════════
EXPERIMENTS = [
    {
        'name': '实验组1-基准',
        'BATCH_SIZE':    100,
        'EPOCHS':        10,
        'learning_rate': 1e-4,   # 基准学习率（原始代码默认值）
        'keep_prob_rate': 0.7,   # Dropout 保留率 0.7，即丢弃 30% 神经元
    },
    {
        'name': '实验组2-提高学习率',
        'BATCH_SIZE':    100,
        'EPOCHS':        10,
        'learning_rate': 1e-3,   # 提高 10 倍学习率，预期收敛更快但可能震荡
        'keep_prob_rate': 0.7,
    },
    {
        'name': '实验组3-增大Dropout减小BatchSize',
        'BATCH_SIZE':    64,     # 更小的批次，梯度噪声更大，具有一定正则化效果
        'EPOCHS':        10,
        'learning_rate': 1e-4,
        'keep_prob_rate': 0.5,   # 增大 Dropout（丢弃 50%），增强泛化、抑制过拟合
    },
]

# 优先使用 GPU，否则退回 CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# ────────────────────────────────────────────────────────────
# 数据预处理（三组实验共用，只需准备一次）
# ToTensor：将 PIL/NumPy 图像 [0,255] 转为 Float Tensor [0,1]
# Normalize(0.5, 0.5)：将 [0,1] 线性映射到 [-1,1]，加速收敛
# ────────────────────────────────────────────────────────────
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
])

path = './data/'  # 数据集下载保存目录

# 下载并加载 MNIST 训练集（60000 张）和测试集（10000 张）
trainData = torchvision.datasets.MNIST(path, train=True,  transform=transform, download=True)
testData  = torchvision.datasets.MNIST(path, train=False, transform=transform)


# ════════════════════════════════════════════════════════════
# 网络结构定义（保持原始结构不变）
# 输入(28×28×1) → Conv1(32) → Pool → Conv2(64) → Pool
# → Flatten → FC(1024) → Dropout → Output(10) → Softmax
# ════════════════════════════════════════════════════════════
class Net(torch.nn.Module):
    def __init__(self, keep_prob_rate=0.7):
        """
        keep_prob_rate: Dropout 层保留神经元的比例，
        作为参数传入，方便不同实验组切换
        """
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
            # 卷积层1：1 通道 → 32 通道，7×7 卷积核，padding=3 保持 28×28 尺寸
            torch.nn.Conv2d(in_channels=1, out_channels=32,
                            kernel_size=7, padding=3, stride=1),
            torch.nn.ReLU(),
            # 最大池化：2×2，步长 2，特征图 28×28 → 14×14
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # 卷积层2：32 通道 → 64 通道，5×5 卷积核，padding=2 保持 14×14 尺寸
            torch.nn.Conv2d(in_channels=32, out_channels=64,
                            kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            # 最大池化：特征图 14×14 → 7×7
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # 展平：将 7×7×64 = 3136 维特征送入全连接层
            torch.nn.Flatten(),

            # 全连接层：3136 → 1024
            torch.nn.Linear(in_features=7 * 7 * 64, out_features=1024),
            torch.nn.ReLU(),
            # Dropout：训练时随机丢弃神经元，防止过拟合
            torch.nn.Dropout(1 - keep_prob_rate),
            # 输出层：1024 → 10（对应 0~9 共 10 个类别）
            torch.nn.Linear(in_features=1024, out_features=10),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, input):
        return self.model(input)


# ════════════════════════════════════════════════════════════
# 主循环：依次跑三组实验
# ════════════════════════════════════════════════════════════
all_results = {}   # 保存所有实验组的 history，用于最终绘图

for exp in EXPERIMENTS:
    exp_name       = exp['name']
    BATCH_SIZE     = exp['BATCH_SIZE']
    EPOCHS         = exp['EPOCHS']
    learning_rate  = exp['learning_rate']
    keep_prob_rate = exp['keep_prob_rate']

    print(f"\n{'='*65}")
    print(f"  开始 {exp_name}")
    print(f"  LR={learning_rate}  Epochs={EPOCHS}  "
          f"BatchSize={BATCH_SIZE}  KeepProb={keep_prob_rate}")
    print(f"{'='*65}")

    # 根据本组 batch_size 重建 DataLoader
    trainDataLoader = torch.utils.data.DataLoader(
        dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
    testDataLoader  = torch.utils.data.DataLoader(
        dataset=testData,  batch_size=BATCH_SIZE)

    # 每组独立初始化网络，避免上一组的权重影响结果
    net = Net(keep_prob_rate=keep_prob_rate).to(device)
    print(net)

    # 交叉熵损失（PyTorch 内部已包含 log，与 Softmax 输出配合无误）
    lossF = torch.nn.CrossEntropyLoss()
    # Adam 优化器：自适应学习率，适合 CNN 训练
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate)

    history = {'Test Loss': [], 'Test Accuracy': []}

    for epoch in range(1, EPOCHS + 1):
        processBar = tqdm(trainDataLoader, unit='step')
        net.train(True)   # 开启训练模式（Dropout 生效）

        for step, (trainImgs, labels) in enumerate(processBar):
            trainImgs = trainImgs.to(device)
            labels    = labels.to(device)

            net.zero_grad()                        # 清空梯度缓存
            outputs  = net(trainImgs)              # 前向传播
            loss     = lossF(outputs, labels)      # 计算本批次损失

            predictions = torch.argmax(outputs, dim=1)
            accuracy    = torch.sum(predictions == labels) / labels.shape[0]

            loss.backward()    # 反向传播，计算梯度
            optimizer.step()   # 更新模型权重

            processBar.set_description(
                "[%d/%d] Loss: %.4f, Acc: %.4f" %
                (epoch, EPOCHS, loss.item(), accuracy.item()))

            # 每个 epoch 的最后一个 batch 结束后，在测试集上评估
            if step == len(processBar) - 1:
                correct, totalLoss = 0, 0
                net.train(False)  # 关闭训练模式（Dropout 不再丢弃）

                for testImgs, labels in testDataLoader:
                    testImgs = testImgs.to(device)
                    labels   = labels.to(device)
                    outputs  = net(testImgs)
                    loss     = lossF(outputs, labels)
                    predictions = torch.argmax(outputs, dim=1)
                    totalLoss += loss
                    correct   += torch.sum(predictions == labels)

                # 用测试集实际总样本数计算准确率，避免因 batch_size 变化而出错
                testAccuracy = correct / len(testData)
                testLoss     = totalLoss / len(testDataLoader)

                history['Test Loss'].append(testLoss.item())
                history['Test Accuracy'].append(testAccuracy.item())

                processBar.set_description(
                    "[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" % (
                        epoch, EPOCHS, loss.item(), accuracy.item(),
                        testLoss.item(), testAccuracy.item()))

        processBar.close()

    all_results[exp_name] = history

    # 打印本组最终结果
    final_loss = history['Test Loss'][-1]
    final_acc  = history['Test Accuracy'][-1]
    print(f"\n【{exp_name}】最终结果 → "
          f"Test Loss: {final_loss:.4f},  Test Accuracy: {final_acc:.4f}")


# ════════════════════════════════════════════════════════════
# 汇总表：打印各组最终 Test Loss / Test Accuracy
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"  各实验组最终结果汇总")
print(f"  {'实验组':<35} {'Test Loss':>10} {'Test Acc':>12}")
print("-" * 65)
for exp in EXPERIMENTS:
    name = exp['name']
    h    = all_results[name]
    print(f"  {name:<35} {h['Test Loss'][-1]:>10.4f} {h['Test Accuracy'][-1]:>12.4f}")
print("=" * 65)


# ════════════════════════════════════════════════════════════
# 绘制对比曲线图
# 左图：Test Loss 曲线；右图：Test Accuracy 曲线
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('CNN 超参数对比实验（MNIST）', fontsize=14, fontweight='bold')

colors  = ['steelblue', 'darkorange', 'forestgreen']
markers = ['o', 's', '^']

for idx, exp in enumerate(EXPERIMENTS):
    name   = exp['name']
    h      = all_results[name]
    epochs = range(1, len(h['Test Loss']) + 1)

    axes[0].plot(epochs, h['Test Loss'],
                 color=colors[idx], marker=markers[idx], markersize=5,
                 label=name, linewidth=1.8)

    axes[1].plot(epochs, h['Test Accuracy'],
                 color=colors[idx], marker=markers[idx], markersize=5,
                 label=name, linewidth=1.8)

# Loss 曲线装饰
axes[0].set_title('Test Loss 对比曲线')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Test Loss')
axes[0].legend(fontsize=8, loc='upper right')
axes[0].grid(True, linestyle='--', alpha=0.5)

# Accuracy 曲线装饰
axes[1].set_title('Test Accuracy 对比曲线')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Test Accuracy')
axes[1].legend(fontsize=8, loc='lower right')
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('cnn_experiment_results.png', dpi=150, bbox_inches='tight')
print("\n训练曲线已保存至 cnn_experiment_results.png")
plt.show()
