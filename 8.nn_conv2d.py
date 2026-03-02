import torch
import torchvision
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../dataset",train=False,transform = torchvision.transforms.ToTensor(),download = False)
dataloader = DataLoader(dataset,batch_size = 64)

class Mod(nn.Module):
    def __init__(self):
        super(Mod,self).__init__()
        self.conv1 = Conv2d(3,6,3,stride = 1,padding = 0)

    def forward(self, x):
        x = self.conv1(x)
        return x

mod = Mod()

writer = SummaryWriter("../logs")
step = 0
for data in dataloader:
    imgs,targets = data
    output = mod(imgs)
    print(imgs.shape)
    print(output.shape)
    #输入大小torch.Size([16, 3, 32, 32])
    writer.add_images("input",imgs,step)
    #输出大小torch.Size([64, 6, 30, 30])->[xxx,3,30,30]
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images("output",output,step)
    step = step + 1
