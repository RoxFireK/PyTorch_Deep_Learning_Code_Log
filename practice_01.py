import torch
from torch import nn
from torch.nn import Conv2d, Sequential, Flatten
from torch.nn import MaxPool2d


class Rox(nn.Module):
    def __init__(self):
        super(Rox, self).__init__()

        self.model1 = Sequential(
            Conv2d(1, 6, 5),
            MaxPool2d(kernel_size=2),
            Conv2d(6, 16, 5),
            MaxPool2d(kernel_size=2),
            Conv2d(16, 120, 5),
            Flatten(),
            nn.Linear(120, 84),
            nn.Linear(84, 10)

        )

    def forward(self, x):
        x = self.model1(x)
        return x

if __name__ == "__main__":
    rox = Rox()
    input = torch.ones(64,1,32,32)
    output = rox(input)
    print(output.shape)