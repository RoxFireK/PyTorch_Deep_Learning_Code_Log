import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#准备测试数据集
test_data = torchvision.datasets.CIFAR10("./dataset",train=False,transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),]))
#batch_size:每次取多少个数据集进行打包
#drop_last:最后不足batch_size的是否要舍去
#shuffle:下一次数据抓取时是否和上一次一样
test_loader = DataLoader(dataset = test_data,batch_size = 64,shuffle = True,num_workers = 0,drop_last = True)

#target:目标索引   测试数据集第一张图片及target
img,target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")
#取0,1进行两轮
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs,targets = data
        #print(imgs.shape)
        #print(targets)
        #测试的时候一定要修改tag!!!!!,epoch上面已经定义了
        writer.add_images("Epoch:{}".format(epoch),imgs,step)
        step = step + 1
writer.close()