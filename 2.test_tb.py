#将训练图像过程上传到logs，可用于服务器开发
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "练手数据集/train/bees_image/21399619_3e61e5bb6f.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(img_array.shape)

writer.add_image("train",img_array,1,dataformats="HWC")
#y = x
for i in range(100):
    writer.add_scalar("y=2x",2*i,i)

writer.close()
#打开tensorboard地址：tensorboard --logdir=logs弹出窗口点击进入
#一台服务器多人合作，如何指定端口防止冲突,设定port参数：tensorboard --logdir=logs --port=6007

