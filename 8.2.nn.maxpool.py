#Floor:向下取整  Ceiling:向上取整
#最大池化操作:取区域内最大值  Ceil_model判定池化核覆盖是否全部覆盖，设定ceil_model为True时，未全覆盖也会进行保留，ceil_model为False不进行保留
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='../dataset', train=False,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)
"""input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]],dtype=torch.float32
                     )"""
#dtype将内容转化为浮点数
"""input = torch.reshape(input,(-1,1,5,5))
print(input.shape)"""

#神经网络
class Rox(nn.Module):
    def __init__(self):
        super(Rox, self).__init__()
        #定义最大池化层
        #池化窗口大小3*3并且保留未全覆盖边界
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,
                                    ceil_mode=True)
    #forward 定义前向传播
    def forward(self, input):
        output = self.maxpool1(input)
        return output

rox = Rox()

writer = SummaryWriter("../logs_maxpool")
step = 0
for data in dataloader:
    imgs,targets = data
    writer.add_images("input",imgs,step)
    output = rox(imgs)
    writer.add_images("output",output,step)
    step += 1
#注意images加s
writer.close()