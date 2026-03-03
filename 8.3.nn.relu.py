import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1,-0.5],
                      [-1,3]])
input = torch.reshape(input,(-1,1,2,2))
print(input.shape)

dataset = torchvision.datasets.CIFAR10(root='../dataset', train=False,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)

class Rox(nn.Module):
    def __init__(self):
        super(Rox,self).__init__()
        #ReLU用法:模拟神经元，输入值为负时输出为0，输入值为正输出原输入值
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()
    def forward(self,input):
        output = self.sigmoid1(input)
        return output

rox = Rox()
writer = SummaryWriter("../logs_relu")
step = 0
for data in dataloader:
    imgs,targets = data
    writer.add_images("input",imgs,global_step=step)
    output = rox(imgs)
    writer.add_images("output",output,step)
    step+=1

writer.close()
