import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from torch.utils.tensorboard import SummaryWriter
from tensorflow.keras import layers,datasets,optimizers
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def mnist_dataset():
    (x,y),(x_test,y_test) = datasets.mnist.load_data()
    x = x/255.0
    x_test = x_test/255.0
    return (x,y),(x_test,y_test)

#矩阵乘法类
class Matmul():
    def __init__(self):
        self.mem = {}
    def forward(self,x,W):
        h = np.matmul(x,W)
        self.mem = {"x":x,"W":W}
        return h
#矩阵乘法运算的反向传播
    def backward(self,grad_y):
        x = self.mem["x"]
        W = self.mem["W"]
        grad_x = np.matmul(grad_y,W.T)
        grad_W = (np.matmul(x.T,grad_y))
        return grad_x,grad_W

#定义ReLU类
class Relu():
    def __init__(self):
        self.mem = {}
    def forward(self,x):
        self.mem["x"] = x
        #ReLU:大于0返回原值，小于等于0返回0
        return np.where(x>0,x,np.zeros_like(x))
    def backward(self,grad_y):
        x = self.mem["x"]
        return (x>0).astype(np.float32)*grad_y

#Softmax:先进行e指数求解，再进行归一化
# 定义 softmax 类
class Softmax():
    def __init__(self):
        # 构造函数
        self.mem = {}
        self.epsilon = 1e-12  # 防止分母出现 0

    # softmax 函数的前向传播
    def forward(self, x):
        # x_shape(N, c)
        x_exp = np.exp(x)
        denominator = np.sum(x_exp, axis=1, keepdims=True)
        # softmax 的分母
        out = x_exp / (denominator + self.epsilon)
        # softmax 的输出，对应式 (4)
        self.mem["out"] = out
        self.mem["x_exp"] = x_exp
        # 将 out 和 x_exp 保存在字典 mem 中，用于反向传播中的梯度计算
        return out

    # softmax 函数的反向传播
    def backward(self, grad_y):
        # grad_y: same shape as x
        s = self.mem["out"] # shape (N, c)
        sisj = np.matmul(np.expand_dims(s, axis=2),
        np.expand_dims(s, axis=1))
        g_y_exp = np.expand_dims(grad_y, axis=1) # shape: (N, 1, c)
        tmp = np.matmul(g_y_exp, sisj) # shape: (N, 1, c)
        tmp = np.squeeze(tmp, axis=1) # shape: (N, c)
        # squeeze()为expand_dims()的逆向操作
        softmax_grad = -tmp + grad_y*s # 对应式(9)
        return softmax_grad

        # 返回softmax函数的梯度

# 定义交叉熵类
class Cross_entropy():
    # 构造函数
    def __init__(self):
        self.epsilon = 1e-12
        self.mem = {}

    # 交叉熵函数的前向传播
    def forward(self, x, labels):
        log_prob = np.log(x + self.epsilon) # 对输出的概率取对数
        out = np.mean(np.sum(-log_prob*labels, axis=1))
        # 对应式(1)，axis=1表示沿水平方向求和
        self.mem["x"] = x
        return out

        # 返回交叉熵，表示loss

    # 交叉熵函数的反向传播
    def backward(self, labels):
        x = self.mem["x"]
        return -1/(x + self.epsilon)*labels # 对应式(3)

# 建立模型
class myModel():
    def __init__(self):
        self.W1 = np.random.normal(size = [28*28+1, 100])
        '''
        np.random.normal() 用于生成正态分布的随机数  
        28*28表示 mnist 数据集中每张图片中像素的个数，  
        +1表示的是偏置项  

        100表示隐藏层的节点数  
        '''

        self.W2 = np.random.normal(size = [100, 10])
        #10表示输出的类别有 10 个

        self.mul_h1 = Matmul()
        self.relu = Relu()
        self.mul_h2 = Matmul()
        self.softmax = Softmax()
        self.cross_en = Cross_entropy()

        #生成类的实例

    ##模型前向传播
    def forward(self, x, labels):
        x = x.reshape(-1, 28*28) #将x铺平
        bias = np.ones(shape=[x.shape[0], 1])
        x = np.concatenate([x, bias], axis=1)

        #在矩阵x和列向量b按水平方向拼到一起
        self.h1 = self.mul_h1.forward(x, self.W1)

        #输入层与W1之间做矩阵乘法
        self.h1_relu = self.relu.forward(self.h1)

        #relu激活
        self.h2 = self.mul_h2.forward(self.h1_relu, self.W2)

        #mul_h2与W2做矩阵乘法
        self.h2_soft = self.softmax.forward(self.h2)

        #softmax函数输出概率
        self.loss = self.cross_en.forward(self.h2_soft, labels)

        #计算交叉熵

    # 模型反向传播
    def backward(self, labels):
        # 交叉熵函数对输入求梯度
        self.loss_grad = self.cross_en.backward(labels)

        # softmax层梯度
        self.h2_soft_grad = self.softmax.backward(self.loss_grad)

        # mul_h2层梯度（矩阵乘法层）
        self.h2_grad, self.W2_grad = \
            self.mul_h2.backward(self.h2_soft_grad)

        # relu层梯度
        self.h1_relu_grad = self.relu.backward(self.h2_grad)

        # mul_h1层梯度（矩阵乘法层）
        self.h1_grad, self.W1_grad = \
            self.mul_h1.backward(self.h1_relu_grad)
model = myModel()

# 计算准确率
def compute_accuracy(prob, labels):
    predictions = np.argmax(prob, axis=1)
    # 返回输出的概率每一行最大值所在的列数
    truth = np.argmax(labels, axis=1)
    # 返回标签每一行最大值所在的列数
    return np.mean(predictions==truth)

# 迭代一个 epoch
def train_one_step(model, x, y):
    model.forward(x, y)
    model.backward(y)
    model.W1 -= 1e-5 * model.W1_grad
    model.W2 -= 1e-5 * model.W2_grad

    # 更新参数 W1, W2
    loss = model.loss # 得到 loss
    accuracy = compute_accuracy(model.h2_soft, y)
    return loss, accuracy

# 计算测试集上的 loss 和准确率
def test(model, x, y):
    model.forward(x, y)
    loss = model.loss
    accuracy = compute_accuracy(model.h2_soft, y)
    return loss, accuracy

# 实际训练
train_data, test_data = mnist_dataset()
train_label = np.zeros(shape=[train_data[0].shape[0],10])

# 生成0矩阵
test_label = np.zeros(shape=[test_data[0].shape[0],10])
train_label[np.arange(train_data[0].shape[0]),np.array(train_data[1])]=1

# 将标签转换成one-hot矩阵
test_label[np.arange(test_data[0].shape[0]),np.array(test_data[1])] = 1

for epoch in range(500):
    loss , accuracy = train_one_step(model, train_data[0], train_label)
    print('epoch', epoch, ': loss', loss, '; accuracy', accuracy)


# 测试
loss , accuracy = test(model, test_data[0], test_label)
print('test loss', loss, '; accuracy', accuracy)


