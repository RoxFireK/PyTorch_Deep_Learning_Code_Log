import numpy as np
import re
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime
import os


### Hyper parameter - 可配置的超参数
class Config:
    # 基础参数
    Learning_rate = 1e-4
    Max_epoch = 20
    Batch_Size = 128
    use_gpu = torch.cuda.is_available()

    # 模型参数
    embed_dim = 256  # 嵌入维度 (原512，降低避免过拟合)
    hidden_size = 256  # 隐藏层大小 (原512)
    num_layers = 2  # RNN层数
    dropout = 0.5  # Dropout率

    # 其他参数
    n_step = 20  # 序列长度
    max_vocab = 10000  # 最大词汇量
    grad_clip = 5  # 梯度裁剪值
    top_n = 5  # 预测时从前N个中选择

    # 优化器参数 (新增)
    optimizer_type = 'adam'  # 可选: 'adam', 'sgd', 'adamw'
    weight_decay = 1e-5  # 权重衰减 (L2正则化)
    momentum = 0.9  # SGD动量

    # 学习率调度 (新增)
    lr_scheduler = 'step'  # 可选: 'step', 'cosine', 'reduce'
    lr_step_size = 5  # step调度器的步长
    lr_gamma = 0.5  # 学习率衰减因子


config = Config()


### 在预测的概率最高的前N个字符中随机选择 - 将函数定义移到前面
def pick_top_n(preds, top_n=5):
    """从预测结果中选择top_n个概率最高的字符，按概率随机采样"""
    top_pred_prob, top_pred_label = torch.topk(preds, top_n, 1)
    top_pred_prob = top_pred_prob / torch.sum(top_pred_prob)
    top_pred_prob = top_pred_prob.squeeze(0).cpu().numpy()
    top_pred_label = top_pred_label.squeeze(0).cpu().numpy()
    c = np.random.choice(top_pred_label, size=1, p=top_pred_prob)
    return c


### 引入诗歌文件
text_path = r'C:\Users\Admin\Desktop\deep_learning\src\RNN\data\poetry.txt'
try:
    with open(text_path, 'r', encoding='utf-8') as f:
        poetry_corpus = f.read()
except FileNotFoundError:
    print(f"错误：找不到文件 {text_path}")
    print("请确保文件路径正确，或修改 text_path 变量")
    exit(1)

### 修改诗歌中的符号
poetry_corpus = poetry_corpus.replace('\\n', '').replace('\\r', '\r').replace(' ', ' ')

print(f"诗歌语料库加载成功，总字符数: {len(poetry_corpus)}")


### 诗歌字符转换
class TextConverter(object):
    def __init__(self, text_path, max_vocab=5000):
        """建立一个字符索引转换器
        Args:
            text_path: 文本位置
            max_vocab: 最大的单词数量
        """
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        text = text.replace(' \n', ' ').replace(' \r', ' ') \
            .replace(' ', ' ').replace('. ', ' ')

        # 去掉重复的字符
        vocab = set(text)

        # 如果单词总数超过最大数值，去掉频率最低的
        vocab_count = {}

        # 计算单词出现频率并排序
        for word in vocab:
            vocab_count[word] = 0

        for word in text:
            vocab_count[word] += 1

        vocab_count_list = []
        for word in vocab_count:
            vocab_count_list.append((word, vocab_count[word]))
        vocab_count_list.sort(key=lambda x: x[1], reverse=True)

        # 如果超过最大值，截取频率最低的字符
        if len(vocab_count_list) > max_vocab:
            vocab_count_list = vocab_count_list[:max_vocab]
        vocab = [x[0] for x in vocab_count_list]
        self.vocab = vocab

        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_word_table = dict(enumerate(self.vocab))

    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)

    def int_to_word(self, index):
        if index == len(self.vocab):
            return ','
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')

    def text_to_arr(self, text):
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return np.array(arr)

    def arr_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)


convert = TextConverter(text_path, max_vocab=config.max_vocab)
print(f"词汇表大小: {convert.vocab_size}")

## 拆分诗歌文件为多个长度为 n_step 序列
n_step = config.n_step

# 总的序列个数
num_seq = int(len(poetry_corpus) / n_step)

# 去掉最后不足一个序列长度的部分
text = poetry_corpus[:num_seq * n_step]

### 分割 ###
arr = convert.text_to_arr(text)
arr = arr.reshape((num_seq, -1))
arr = torch.from_numpy(arr)


class TextDataset(object):
    def __init__(self, arr):
        self.arr = arr

    def __getitem__(self, item):
        x = self.arr[item, :]
        # 构造 label
        y = torch.zeros(x.shape)
        # 将输入的第一个字符作为最后一个输入的 label
        y[: -1], y[-1] = x[1:], x[0]
        return x, y

    def __len__(self):
        return self.arr.shape[0]


train_set = TextDataset(arr)
print(f"训练集样本数: {len(train_set)}")


# 模型构建 (改进版，加入Dropout)
class myRNN(nn.Module):
    def __init__(self, num_classes, embed_dim, hidden_size,
                 num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.word_to_vec = nn.Embedding(num_classes, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.RNN(embed_dim, hidden_size, num_layers,
                          dropout=dropout if num_layers > 1 else 0)
        self.project = nn.Linear(hidden_size, num_classes)

    def forward(self, x, hs=None):
        batch = x.shape[0]
        if hs is None:
            hs = Variable(
                torch.zeros(self.num_layers, batch, self.hidden_size))
            if config.use_gpu:
                hs = hs.cuda()
        word_embed = self.word_to_vec(x)  # (batch, len, embed)
        word_embed = self.dropout(word_embed)
        word_embed = word_embed.permute(1, 0, 2)  # (len, batch, embed)
        out, h0 = self.rnn(word_embed, hs)  # (len, batch, hidden)
        le, mb, hd = out.shape
        out = out.view(le * mb, hd)
        out = self.project(out)
        out = out.view(le, mb, -1)
        out = out.permute(1, 0, 2).contiguous()  # (batch, len, hidden)
        return out.view(-1, out.shape[2]), h0


# 创建 TensorBoard writer
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join('tb_logs', f'poetry_rnn_{timestamp}')
writer = SummaryWriter(log_dir)

# 记录超参数
writer.add_text('hyperparameters', str(config.__dict__), 0)

# 数据加载
batch_size = config.Batch_Size
train_data = DataLoader(train_set, batch_size, True, num_workers=0)

model = myRNN(convert.vocab_size, config.embed_dim, config.hidden_size,
              config.num_layers, config.dropout)
if config.use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()

# 优化器选择
if config.optimizer_type == 'adam':
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.Learning_rate,
                                 weight_decay=config.weight_decay)
elif config.optimizer_type == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.Learning_rate,
                                  weight_decay=config.weight_decay)
elif config.optimizer_type == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=config.Learning_rate,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.Learning_rate)

# 学习率调度器
if config.lr_scheduler == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)
elif config.lr_scheduler == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.Max_epoch)
elif config.lr_scheduler == 'reduce':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config.lr_gamma, patience=3)
else:
    scheduler = None

# 训练循环
epochs = config.Max_epoch
global_step = 0

print("=" * 50)
print("开始训练诗歌生成模型")
print(f"词汇表大小: {convert.vocab_size}")
print(f"嵌入维度: {config.embed_dim}")
print(f"隐藏层大小: {config.hidden_size}")
print(f"序列长度: {config.n_step}")
print(f"优化器: {config.optimizer_type}")
print(f"学习率: {config.Learning_rate}")
print(f"TensorBoard日志目录: {log_dir}")
print("=" * 50)

for e in range(epochs):
    train_loss = 0
    model.train()

    for batch_idx, data in enumerate(train_data):
        x, y = data
        y = y.long()
        if config.use_gpu:
            x = x.cuda()
            y = y.cuda()
        x, y = Variable(x), Variable(y)

        # Forward
        score, _ = model(x)
        loss = criterion(score, y.view(-1))

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        train_loss += loss.item()

        # TensorBoard 记录每个batch的损失
        writer.add_scalar('Loss/batch', loss.item(), global_step)

        # 记录梯度范数
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        writer.add_scalar('Gradient/norm', total_norm, global_step)

        global_step += 1

        # 每100个batch显示进度
        if batch_idx % 100 == 0:
            print(f'Epoch {e + 1}/{epochs}, Batch {batch_idx}/{len(train_data)}, Loss: {loss.item():.4f}')

    # 计算困惑度和平均损失
    avg_loss = train_loss / len(train_data)
    perplexity = np.exp(avg_loss)

    # 学习率调度
    if scheduler is not None:
        if config.lr_scheduler == 'reduce':
            scheduler.step(avg_loss)
        else:
            scheduler.step()

    # 记录当前学习率
    current_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('Learning_rate', current_lr, e)

    # TensorBoard 记录epoch级别的指标
    writer.add_scalar('Loss/epoch', avg_loss, e)
    writer.add_scalar('Perplexity', perplexity, e)

    # 打印训练信息
    print(f'Epoch: {e + 1}/{epochs}, Loss: {avg_loss:.3f}, '
          f'Perplexity: {perplexity:.2f}, LR: {current_lr:.6f}')

    # 每5个epoch生成示例文本
    if (e + 1) % 5 == 0 or e == 0:
        model.eval()
        with torch.no_grad():
            begin = "月"
            text_len = 28

            samples = [convert.word_to_int(c) for c in begin]
            input_txt = torch.LongTensor(samples)[None]
            if config.use_gpu:
                input_txt = input_txt.cuda()
            input_txt = Variable(input_txt)
            _, init_state = model(input_txt)
            result = samples.copy()
            model_input = input_txt[:, -1][:, None]

            while True:
                out, init_state = model(model_input, init_state)
                pred = pick_top_n(out.data, config.top_n)

                model_input = Variable(torch.LongTensor(pred))[None]
                if config.use_gpu:
                    model_input = model_input.cuda()

                if pred[0] != 0:
                    result.append(pred[0])
                    if len(result) > text_len:
                        break
                else:
                    break

            text = convert.arr_to_text(result)
            # 处理生成的文本
            if len(text) >= 7:
                formatted_text = ', '.join([text[i:i + 7] for i in range(0, len(text), 7) if i + 7 <= len(text)])
            else:
                formatted_text = text
            formatted_text = formatted_text + '。'

            print(f'生成示例:\n{formatted_text}\n')

            # 记录生成的文本到TensorBoard
            writer.add_text(f'Generated/Epoch_{e + 1}', formatted_text, e)

        model.train()

# 关闭writer
writer.close()

# 最终生成
print("\n" + "=" * 50)
print("最终生成测试")
print("=" * 50)

begin = "月"
text_len = 28

model = model.eval()
samples = [convert.word_to_int(c) for c in begin]
input_txt = torch.LongTensor(samples)[None]
if config.use_gpu:
    input_txt = input_txt.cuda()
input_txt = Variable(input_txt)
_, init_state = model(input_txt)
result = samples
model_input = input_txt[:, -1][:, None]

while True:
    out, init_state = model(model_input, init_state)
    pred = pick_top_n(out.data, config.top_n)

    model_input = Variable(torch.LongTensor(pred))[None]
    if config.use_gpu:
        model_input = model_input.cuda()

    if pred[0] != 0:
        result.append(pred[0])
        if len(result) > text_len:
            break
    else:
        break

# 输出修饰
text = convert.arr_to_text(result)
# 处理生成的文本
if len(text) >= 7:
    formatted_text = ', '.join([text[i:i + 7] for i in range(0, len(text), 7) if i + 7 <= len(text)])
else:
    formatted_text = text
formatted_text = formatted_text + '。'

print(f'输出：\n{formatted_text}')

# 保存模型
model_path = os.path.join(log_dir, 'model.pth')
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config.__dict__,
    'vocab': convert.vocab,
    'epoch': config.Max_epoch,
}, model_path)
print(f"\n模型已保存至: {model_path}")
print(f"TensorBoard日志已保存至: {log_dir}")
print(f"\n启动TensorBoard命令: tensorboard --logdir={log_dir}")