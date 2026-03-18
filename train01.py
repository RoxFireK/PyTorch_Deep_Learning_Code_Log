#先导入库再导入网络
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#注意model文件要和main文件在同一文件夹下
from model01 import *

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

#添加tensorboard
writer = SummaryWriter("../logs_train")

for i in range(epoch):
    print("epoch:{}".format(i+1))

    #训练步骤开始
    rox.train()
    for data in train_dataloader:
        imgs,targets = data
        outputs = rox(imgs)
        loss = loss_fn(outputs,targets)

        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数:{},loss:{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    #测试步骤开始 no_gard:没有梯度
    rox.eval()
    total_test_loss = 0
    total_accuracy = 0
    #不需要梯度来进行优化
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            outputs = rox(imgs)
            loss =loss_fn(outputs,targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy.item()
    print("整体测试集上的loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
    total_train_step = total_train_step + 1

    torch.save(rox,"rox_{}.pth".format(i))
    print("模型已保存")

writer.close()
#访问 tensorboard --logdir=logs_train

