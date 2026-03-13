import torch
import torchvision
from torch import nn
#解决方法2 从文件导入神经网络
from model_save import *

#方式一->保存方式1.加载模型
model = torch.load("vgg16_method1.pth",weights_only=False)
#print(model)

#方式二.加载模型 保存字典型数据
vgg16 = torchvision.models.vgg16(weights = None)
#转结构
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
#model = torch.load("vgg16_method2.pth")
#print(model)

#陷阱1
#解决方法1 把神经网络复制过来
"""class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,1)
    def forward(self, input):
        output = self.conv1(input)
        return output"""

#不需要写创建 net = Net()
model = torch.load("net_method1.pth",weights_only=False)
print(model)