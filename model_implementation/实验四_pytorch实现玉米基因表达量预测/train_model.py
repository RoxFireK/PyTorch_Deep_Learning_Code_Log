import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from einops.layers.torch import Rearrange
import os
import json
import warnings

warnings.filterwarnings('ignore')

# 导入数据预处理函数
from data_preprocess import load_and_preprocess_data


# ==================== 数据集类 ====================
class MyDataset(Dataset):
    def __init__(self, input_data, label_data):
        self.inputs = torch.tensor(input_data, dtype=torch.float32)
        self.labels = torch.tensor(label_data, dtype=torch.float32)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


# ==================== 模型组件 ====================
class upd_GELU(nn.Module):
    def __init__(self):
        super(upd_GELU, self).__init__()
        self.constant_param = nn.Parameter(torch.Tensor([1.702]))
        self.sig = nn.Sigmoid()

    def forward(self, input: Tensor) -> Tensor:
        return torch.mul(self.sig(torch.mul(self.constant_param, input)), input)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size=1, padding='same',
                 stride=1, dilation_rate=1, dropout=0.0, bn_momentum=0.1):
        super().__init__()

        # 处理padding
        if padding == 'same':
            padding = kernel_size // 2

        layers = [
            upd_GELU(),
            nn.Conv1d(in_channels, filters, kernel_size, stride=stride,
                      padding=padding, dilation=dilation_rate, bias=False),
            nn.BatchNorm1d(filters, momentum=bn_momentum)
        ]

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ConvTower(nn.Module):
    def __init__(self, in_channels, filters_init, filters_end=None,
                 repeat=2, kernel_size=5, **kwargs):
        super().__init__()

        if filters_end is None:
            filters_end = filters_init

        filters_mult = np.exp(np.log(filters_end / filters_init) / (repeat - 1))

        reps_filters = filters_init
        in_ch = in_channels
        tower = []

        for _ in range(repeat):
            tower.append(ConvBlock(in_ch, round(reps_filters),
                                   kernel_size=kernel_size, **kwargs))
            in_ch = round(reps_filters)
            reps_filters *= filters_mult

        self.tower = nn.Sequential(*tower)
        self.out_channels = in_ch

    def forward(self, x):
        return self.tower(x)


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, module):
        super().__init__()
        self.module = module
        # 如果输入输出通道数不同，使用1x1卷积调整
        self.skip = nn.Conv1d(in_channels, out_channels,
                              kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.module(x) + self.skip(x)


class DilatedResidual(nn.Module):
    def __init__(self, in_channels, filters, kernel_size=3, rate_mult=2,
                 dropout=0.0, repeat=1, **kwargs):
        super().__init__()
        dilation_rate = 1
        in_ch = in_channels
        blocks = []

        for i in range(repeat):
            # 第一个卷积：in_ch -> filters
            conv1 = ConvBlock(in_ch, filters, kernel_size=kernel_size,
                              dilation_rate=int(round(dilation_rate)), **kwargs)
            # 第二个卷积：filters -> in_ch (恢复原始通道数)
            conv2 = ConvBlock(filters, in_ch, dropout=dropout, **kwargs)

            # 残差连接：输入通道in_ch，输出通道in_ch
            residual_block = nn.Sequential(conv1, conv2)
            blocks.append(Residual(in_ch, in_ch, residual_block))

            dilation_rate *= rate_mult

        self.block = nn.Sequential(*blocks)
        self.out_channels = in_ch

    def forward(self, x):
        return self.block(x)


class BasenjiModel(nn.Module):
    def __init__(self, seq_len=3000):
        super().__init__()

        # 参数配置
        self.seq_len = seq_len

        # ConvBlock 1: 4 -> 8, 下采样 stride=2
        self.conv1 = ConvBlock(in_channels=4, filters=8, kernel_size=15,
                               padding=7, dropout=0.4, bn_momentum=0.1)

        # MaxPooling 下采样
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # ConvTower: 8 -> 16 -> 32
        self.conv_tower = ConvTower(in_channels=8, filters_init=16,
                                    filters_end=32, repeat=2, kernel_size=5,
                                    dropout=0.3, bn_momentum=0.1)

        # MaxPooling 下采样
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # DilatedResidual: 32 -> 16 (内部保持通道数不变)
        self.dil_res = DilatedResidual(in_channels=32, filters=16,
                                       kernel_size=3, rate_mult=2,
                                       dropout=0.3, repeat=2,
                                       bn_momentum=0.1)

        # MaxPooling 下采样
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # ConvBlock 2 (1x1 conv) 调整通道数
        self.conv2 = ConvBlock(in_channels=32, filters=32, kernel_size=1,
                               dropout=0.2, bn_momentum=0.1)

        # ConvBlock 3 (1x1 conv) 降到1通道
        self.conv3 = ConvBlock(in_channels=32, filters=1, kernel_size=1,
                               dropout=0.2, bn_momentum=0.1)

        # 计算最终特征维度
        # 经过3次stride=2的pooling: 3000 -> 1500 -> 750 -> 375
        final_length = seq_len // (2 ** 3)  # 3000/8 = 375
        final_features = final_length * 1  # 通道数为1

        # Global average pooling and final output
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 全局平均池化到1个时间步
            nn.Flatten(),  # 展平
            nn.Linear(1, 1)  # 输出单个值
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv_tower(x)
        x = self.pool2(x)

        x = self.dil_res(x)
        x = self.pool3(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final(x)
        return x.flatten()


# ==================== 早停类 ====================
class EarlyStopping:
    """早停机制，当验证集指标不再提升时停止训练"""

    def __init__(self, patience=10, verbose=True, delta=0, path='checkpoints/best_model.pt'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

        # 创建保存目录
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def __call__(self, val_loss, model, epoch=None, optimizer=None):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, optimizer):
        """保存最佳模型"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'val_loss': val_loss
        }
        torch.save(checkpoint, self.path)
        self.val_loss_min = val_loss


# ==================== 学习率调度器包装 ====================
def get_lr_scheduler(optimizer, scheduler_type='reduce_on_plateau', **kwargs):
    """获取学习率调度器"""
    if scheduler_type == 'reduce_on_plateau':
        patience = kwargs.pop('patience', 5)
        factor = kwargs.pop('factor', 0.5)
        min_lr = kwargs.pop('min_lr', 1e-6)

        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            verbose=False,
            min_lr=min_lr,
            **kwargs
        )
    elif scheduler_type == 'cosine':
        T_max = kwargs.get('T_max', 50)
        eta_min = kwargs.get('eta_min', 1e-6)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
    elif scheduler_type == 'step':
        step_size = kwargs.get('step_size', 30)
        gamma = kwargs.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    else:
        return None


# ==================== 训练函数 ====================
def train_epoch(model, dataloader, optimizer, loss_fn, device, scaler=None,
                grad_clip=1.0, use_amp=False):
    """训练一个epoch，支持混合精度和梯度裁剪"""
    model.train()
    total_loss = 0

    for inputs, labels in tqdm(dataloader, desc='Training'):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # 混合精度训练
        if use_amp and scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * len(labels)

    return total_loss / len(dataloader.dataset)


def validate(model, dataloader, loss_fn, device):
    """验证函数"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validating'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * len(labels)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)

    # 计算PCC，处理可能出现的NaN
    try:
        pcc, _ = pearsonr(all_preds, all_labels)
        if np.isnan(pcc):
            pcc = 0.0
    except:
        pcc = 0.0

    return avg_loss, pcc


def test(model, dataloader, device):
    """测试函数"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Testing'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    try:
        pcc, p_value = pearsonr(all_preds, all_labels)
        if np.isnan(pcc):
            pcc = 0.0
            p_value = 1.0
    except:
        pcc = 0.0
        p_value = 1.0

    # 计算R²
    ss_res = np.sum((np.array(all_labels) - np.array(all_preds)) ** 2)
    ss_tot = np.sum((np.array(all_labels) - np.mean(all_labels)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    return pcc, p_value, r2, all_preds, all_labels


# ==================== 主程序 ====================
def main():
    # 1. 配置参数
    config = {
        'excel_path': '/mnt/cgshare/dataset.xlsx',
        'batch_size': 32,
        'epochs': 100,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        'use_amp': torch.cuda.is_available(),
        'patience': 15,
        'scheduler_type': 'reduce_on_plateau',
        'save_dir': 'checkpoints',
        'random_seed': 2023
    }

    # 设置随机种子
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['random_seed'])

    # 2. 加载数据
    print("加载数据...")
    train_data, y_train, valid_data, y_valid, test_data, y_test = load_and_preprocess_data(config['excel_path'])

    print(f"训练集大小: {train_data.shape}")
    print(f"验证集大小: {valid_data.shape}")
    print(f"测试集大小: {test_data.shape}")

    # 3. 创建DataLoader
    train_dataset = MyDataset(train_data, y_train)
    valid_dataset = MyDataset(valid_data, y_valid)
    test_dataset = MyDataset(test_data, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'],
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                             num_workers=4, pin_memory=True)

    # 4. 创建模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BasenjiModel(seq_len=train_data.shape[2])
    model = model.to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 5. 训练配置
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config['lr'],
                                  weight_decay=config['weight_decay'])

    # 学习率调度器
    scheduler = get_lr_scheduler(optimizer, config['scheduler_type'])

    # 损失函数
    loss_fn = nn.HuberLoss(delta=1.0)

    # 混合精度训练
    if config['use_amp'] and torch.cuda.is_available():
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None

    # 早停
    checkpoint_path = os.path.join(config['save_dir'], 'best_model.pt')
    early_stopping = EarlyStopping(patience=config['patience'],
                                   verbose=True,
                                   path=checkpoint_path)

    # 6. 训练循环
    history = {
        'train_loss': [],
        'valid_loss': [],
        'valid_pcc': [],
        'learning_rates': []
    }
    best_pcc = -np.inf

    print(f"\n开始训练，最大epochs: {config['epochs']}")
    print(f"早停耐心值: {config['patience']}")
    print(f"使用混合精度: {config['use_amp']}")
    print(f"使用设备: {device}\n")

    for epoch in range(1, config['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")

        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn,
                                 device, scaler, config['grad_clip'], config['use_amp'])

        # 验证
        valid_loss, valid_pcc = validate(model, valid_loader, loss_fn, device)

        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 保存历史记录
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['valid_pcc'].append(valid_pcc)
        history['learning_rates'].append(current_lr)

        # 打印结果
        print(
            f"Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}, Valid PCC: {valid_pcc:.4f}, LR: {current_lr:.2e}")

        # 保存最佳PCC模型
        if valid_pcc > best_pcc:
            best_pcc = valid_pcc
            best_model_path = os.path.join(config['save_dir'], f'best_pcc_model_epoch{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_pcc': valid_pcc,
                'valid_loss': valid_loss
            }, best_model_path)
            print(f"保存最佳PCC模型 (PCC: {valid_pcc:.4f})")

        # 学习率调度
        if config['scheduler_type'] == 'reduce_on_plateau':
            scheduler.step(valid_loss)
        elif scheduler is not None:
            scheduler.step()

        # 早停检查
        early_stopping(valid_loss, model, epoch, optimizer)
        if early_stopping.early_stop:
            print(f"早停触发！在第 {epoch} 轮停止训练")
            break

    # 7. 加载最佳模型进行测试
    print(f"\n加载最佳模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"最佳模型来自 epoch {checkpoint['epoch']}, 验证损失: {checkpoint['val_loss']:.6f}")

    # 8. 测试
    print("\n测试模型...")
    test_pcc, test_p_value, test_r2, preds, labels = test(model, test_loader, device)
    print(f"Test PCC: {test_pcc:.4f} (p-value: {test_p_value:.2e})")
    print(f"Test R²: {test_r2:.4f}")

    # 9. 保存结果
    results = {
        'config': config,
        'test_pcc': float(test_pcc),
        'test_r2': float(test_r2),
        'best_valid_pcc': float(best_pcc),
        'best_valid_loss': float(checkpoint['val_loss']),
        'history': {k: [float(x) for x in v] for k, v in history.items()}
    }

    results_path = os.path.join(config['save_dir'], 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"结果已保存到 {results_path}")

    # 10. 绘图
    plt.figure(figsize=(15, 10))

    # 损失曲线
    plt.subplot(2, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['valid_loss'], label='Valid Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True, alpha=0.3)

    # PCC曲线
    plt.subplot(2, 3, 2)
    plt.plot(history['valid_pcc'], label='Valid PCC', color='green', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('PCC')
    plt.legend()
    plt.title('Validation PCC')
    plt.grid(True, alpha=0.3)

    # 学习率曲线
    plt.subplot(2, 3, 3)
    plt.plot(history['learning_rates'], label='Learning Rate', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.legend()
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)

    # 测试集散点图
    plt.subplot(2, 3, 4)
    plt.scatter(preds, labels, s=10, alpha=0.5, c='blue', edgecolors='none')
    plt.plot([min(labels), max(labels)], [min(labels), max(labels)], 'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Predictions')
    plt.ylabel('True Values')
    plt.title(f'Test Set Predictions (PCC = {test_pcc:.4f}, R² = {test_r2:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 残差图
    plt.subplot(2, 3, 5)
    residuals = np.array(labels) - np.array(preds)
    plt.scatter(preds, residuals, s=10, alpha=0.5, c='purple', edgecolors='none')
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predictions')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)

    # 预测分布直方图
    plt.subplot(2, 3, 6)
    plt.hist(preds, bins=50, alpha=0.5, label='Predictions', color='blue')
    plt.hist(labels, bins=50, alpha=0.5, label='True Values', color='orange')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Distribution of Predictions vs True Values')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config['save_dir'], 'training_results.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n所有结果已保存到 {config['save_dir']}/")
    print(f"  - 最佳模型: {checkpoint_path}")
    print(f"  - 结果JSON: {results_path}")
    print(f"  - 训练图表: {config['save_dir']}/training_results.png")


if __name__ == '__main__':
    main()