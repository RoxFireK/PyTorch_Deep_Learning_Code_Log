import torch
from torch.nn import L1Loss, MSELoss

#Softmax:
#是一个归一化或者说是概率化的工具
#第一步，对输入的所有数取指数，这一步能把所有数变成正数，并且拉开数值间的差距
#第二步，除以总和(归一化)
#Softmax和sigmoid的区别
#softmax是多选一，sigmoid是多选多
#softmax总和为一，sigmoid总和不为一

inputs = torch.tensor([1,2,3],dtype = torch.float)
targets = torch.tensor([1,2,5],dtype = torch.float)

inputs = torch.reshape(inputs,(1,1,1,3))
targets = torch.reshape(targets,(1,1,1,3))

#L1范数损失，平均绝对误差，默认对预测值和真实标签取平均值
#计算过程:(|x1-x2|+|y1-y2|+|z1-z2|)/3
#reduction修改：sum求和 none直接输出每个元素的差值绝对值
loss = L1Loss(reduction = 'none')
result = loss(inputs,targets)

#MSE均方误差也就是L2Loss，差值平方除以n
loss_mse = MSELoss()
result_mse = loss_mse(inputs,targets)

print(result)
print(result_mse)