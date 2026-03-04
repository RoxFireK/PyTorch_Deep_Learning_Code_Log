import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root='../dataset', train=False,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64,drop_last=True)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.linear1 = nn.Linear(196608,10)
    def forward(self, input):
        output = self.linear1(input)
        return output

net = Net()

for data in dataloader:
    # 基本格式[batch_size批量大小，一次处理多少图片,channels通道数,height高,width宽]
    imgs,targets = data
    print(imgs.shape)#torch.Size([64, 3, 32, 32])
    #在reshape中，-1是特殊参数，表示自动计算这个维度的大小，其他维度指定为1
    #原始imgs中，总元素数为64*3*32*32=196608
    #(1,1,1,-1)表示对imgs进行展开，用于全连接层的接入
    #output = torch.reshape(imgs,(1,1,1,-1))
    #flatten将输入展成一行可以代替上面操作
    output = torch.flatten(imgs)
    print(output.shape)
    output = net(output)
    print(output.shape)