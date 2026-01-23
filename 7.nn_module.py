#神经网络的导入和使用
import torch
from torch import nn

#方法来源于pytorch官方文档中1.8.1版本torch.nn的contains的具体用法
class Rox(nn.Module):
    def __init__(self):
        super(Rox,self).__init__()

    def forward(self,input):
        output = input + 1
        return output

rox = Rox()
#此处定义了一个张量
#什么是张量：一种数据结构，是多维数组，0维一个数，一维一行数，二维表格，三维立体，四维视频

x = torch.tensor(1.0)
output = rox(x)
print(output)