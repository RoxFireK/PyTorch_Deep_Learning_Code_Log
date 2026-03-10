#Lr = learning rate 学习速率
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, Sequential
from torch.nn import MaxPool2d, Flatten
from torch.utils.data import DataLoader


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
#定义优化器，lr为学习率
optim = torch.optim.SGD(rox.parameters(), lr=0.01)
#epoch训练轮数
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs,targets = data
        outputs = rox(imgs)
        result_loss = loss(outputs, targets)
        #清除之前的梯度
        optim.zero_grad()
        #反向传播，计算当前批次的梯度
        result_loss.backward()
        #根据计算得到的梯度更新模型的参数
        #SGD:参数 = 参数-学习率*梯度
        optim.step()
        running_loss = running_loss + result_loss
    print(running_loss)
"""结果：可见错误率在下降
tensor(361.1764, grad_fn=<AddBackward0>)
tensor(358.2453, grad_fn=<AddBackward0>)
tensor(348.5228, grad_fn=<AddBackward0>)
tensor(330.6453, grad_fn=<AddBackward0>)
tensor(316.0331, grad_fn=<AddBackward0>)
tensor(306.2532, grad_fn=<AddBackward0>)
tensor(295.3839, grad_fn=<AddBackward0>)
tensor(285.4776, grad_fn=<AddBackward0>)
tensor(277.8412, grad_fn=<AddBackward0>)
tensor(271.1747, grad_fn=<AddBackward0>)
tensor(265.2223, grad_fn=<AddBackward0>)
tensor(259.7782, grad_fn=<AddBackward0>)
tensor(254.7496, grad_fn=<AddBackward0>)
tensor(250.1437, grad_fn=<AddBackward0>)
tensor(245.9507, grad_fn=<AddBackward0>)
tensor(242.0193, grad_fn=<AddBackward0>)
tensor(238.3427, grad_fn=<AddBackward0>)
tensor(234.8589, grad_fn=<AddBackward0>)
tensor(231.5855, grad_fn=<AddBackward0>)
tensor(228.4863, grad_fn=<AddBackward0>)"""