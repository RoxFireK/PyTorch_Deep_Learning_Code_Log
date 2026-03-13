import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(weights = None)
# 保存方式1 模型结构+模型参数
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2 模型参数(体量小，推荐)
torch.save(vgg16.state_dict(), "vgg16_method2.pth")

#陷阱
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,1)
    def forward(self, input):
        output = self.conv1(input)
        return output

net = Net()

torch.save(net,"net_method1.pth")