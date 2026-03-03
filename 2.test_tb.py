#将训练图像过程上传到logs，可用于服务器开发
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

#()里放存储训练后数据的文件夹地址
writer = SummaryWriter("../logs")
image_path = "练手数据集/train/bees_image/21399619_3e61e5bb6f.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)#转化为numpy数组
print(img_array.shape)

#添加数据到tensorboard，括号内容为本次训练名称，数据，计数器
#dataformat用于定义如何解释图像数组的维度
#HWC：opencv,PIL转numpy默认格式
#CHW:tensor格式
#HW:灰度图像，无通道维度
writer.add_image("train",img_array,1,dataformats="HWC")
#可以进行图像绘制
for i in range(100):
    writer.add_scalar("y=2x",2*i,i)
#记得关闭writer
writer.close()
#打开tensorboard地址：tensorboard --logdir=logs弹出窗口点击进入
#一台服务器多人合作，如何指定端口防止冲突,设定port参数：tensorboard --logdir=logs --port=6007


