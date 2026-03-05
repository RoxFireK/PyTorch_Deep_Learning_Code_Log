import torch
from torch import nn
from torch.nn import Conv2d, Sequential  # 注意使用conv2d时导入这个库
from torch.nn import MaxPool2d, Flatten
from torch.utils.tensorboard import SummaryWriter


class Rox(nn.Module):
    def __init__(self):
        super(Rox, self).__init__()
        #用几个卷积核就会形成几个特征矩阵
        #dilation默认为1
        #如何计算padding：官方文档公式代入
        #in_channel:输入通道数，out_channel:输出通道数
        #前后通道不变情况下，padding = (sizeof kernel_size-1)/2
        #注意padding填充 和 stride步长的计算只和尺寸有关

        #------------第一种写法-------------
        """self.conv1 = Conv2d(3,32,5,padding = 2)
        self.maxpool1 = MaxPool2d(kernel_size = 2)
        self.conv2 = Conv2d(32,32,5,padding = 2)
        self.maxpool2 = MaxPool2d(kernel_size = 2)
        self.conv3 = Conv2d(32,64,5,padding = 2)
        self.maxpool3 = MaxPool2d(kernel_size = 2)
        self.flatten = Flatten()
        self.linear1 = nn.Linear(1024,64)
        self.linear2 = nn.Linear(64,10)"""

        # ------------第二种写法-------------
        self.model1 = Sequential(

            #复制的时候记得改数
            Conv2d(3,32,5,padding = 2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)

        )

    def forward(self,x):
        # ------------第一种写法-------------
        """x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        #如何求全连接层参数：删掉线性层看最后剩的参数
        x = self.linear1(x)
        x = self.linear2(x)"""
        # ------------第二种写法-------------
        x = self.model1(x)
        return x

rox = Rox()
print(rox)

#ones用于建立张量
#batch_size批量大小 channels通道数 高，宽
input = torch.ones(64,3,32,32)
output = rox(input)
print(output.shape)

writer = SummaryWriter("../logs")
writer.add_graph(rox,input)
writer.close()