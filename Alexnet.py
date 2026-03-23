import torch
from torch import nn
from torch.nn import Conv2d, Sequential, Flatten
from torch.nn import MaxPool2d


class Alexnet(nn.Module):
    def __init__(self):
        super(Alexnet, self).__init__()

        self.model1 = Sequential(
            Conv2d(3, 96, 11, stride=4,padding = 1),
            nn.ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(96, 256, 5, padding=2),
            nn.ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(256, 384, 3,padding=1),
            nn.ReLU(inplace=True),
            Conv2d(384, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
            Flatten(),
            nn.Linear(6400, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1000),

            )

        """self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256*6*6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, ),
            )"""
    def forward(self, x):
        x = self.model1(x)
        return x

if __name__ == "__main__":
    alex = Alexnet()
    input = torch.ones(1,3,224,224)
    output = alex(input)
    print(output.shape)