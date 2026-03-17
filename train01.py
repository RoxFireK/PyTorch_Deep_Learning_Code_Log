#先导入库再导入网络
import torchvision
from torch.utils.data import DataLoader
#注意model文件要和main文件在同一文件夹下
from model import *

#准备数据集
train_data = torchvision.datasets.CIFAR10("../dataset",train=True,transform = torchvision.transforms.ToTensor(),download = False)
test_data = torchvision.datasets.CIFAR10("../dataset",train=False,transform = torchvision.transforms.ToTensor(),download = False)
#length长度
train_data_size = len(train_data)
test_data_size = len(test_data)
#如果train_data_size = 10，训练数据集长度为10
print("训练数据集的长度为:{}".format(train_data_size))
print("测试数据集的长度为:{}".format(test_data_size))

#利用Dataloader 加载数据
train_dataloader = DataLoader(train_data,batch_size = 64)
test_dataloader = DataLoader(test_data,batch_size = 64)

#创建网络模型
rox = Rox()

#损失函数
loss_fn = nn.CrossEntropyLoss()

#优化器
#learning_rate = 0.01
#1e-2 = 1x(10)^(-2) = 1/100 = 0.01
learning_rate = 0.01
optimizer = torch.optim.SGD(rox.parameters(),lr=learning_rate)

#设置训练网络的一些参数
#记录训练次数
total_train_step = 0
#记录测试次数
total_test_step = 0
#训练轮数
epoch = 10

for i in range(epoch):
    print("epoch:{}".format(i+1))

    #训练步骤开始
    for data in train_dataloader:
        imgs,targets = data
        outputs = rox(imgs)
        loss = loss_fn(outputs,targets)

        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        print("训练次数:{},loss:{}".format(total_train_step,loss.item()))
