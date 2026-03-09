import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, Sequential
from torch.nn import MaxPool2d, Flatten
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../dataset",train=False,transform = torchvision.transforms.ToTensor(),download = False)
dataloader = DataLoader(dataset,batch_size = 64)
class Rox(nn.Module):
    def __init__(self):
        super(Rox, self).__init__()

        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)

        )

    def forward(self, x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
rox = Rox()
for data in dataloader:
    imgs,targets = data
    outputs = rox(imgs)
    result_loss = loss(outputs, targets)
    #result_loss.backward()
    print("ok")

