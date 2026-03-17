import torch
from torch import nn
from torch.nn import Conv2d, Sequential
from torch.nn import MaxPool2d, Flatten


class Rox(nn.Module):
    def __init__(self):
        super(Rox, self).__init__()

        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            nn.ReLU(),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 32, 5, padding=2),
            nn.ReLU(),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            MaxPool2d(kernel_size=2),
            Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)

        )

    def forward(self, x):
        x = self.model1(x)
        return x

if __name__ == "__main__":
    rox = Rox()
    input = torch.ones(64,3,32,32)
    output = rox(input)
    print(output.shape)