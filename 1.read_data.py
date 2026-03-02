#用于显示图片，补充数据集
#图像数据类型：1.PIL Image保持原始图像格式信息，支持多种图像模式，便于进行图像处理操作
#2.NumPy数组类型，便于进行数值计算,opencv使用,float32/64，BGR
#3.pytorch Tensor类型 使用ToTensor进行转化，是神经网络的输入格式
from pygments.formatters import img
from torch.utils.data import Dataset
from PIL import Image
import os
#定义MyData类，继承pytorch的Dataset基类
class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir#保存根目录路径
        self.label_dir = label_dir#保存标签目录名
        self.path = os.path.join(self.root_dir,self.label_dir)#拼接完整路径
        self.img_path = os.listdir(self.path)#获取该目录下所有图片文件名,返回列表

    def __getitem__(self, idx):
        #支持通过索引获取数据
        img_name = self.img_path[idx]#数组索引
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        #使用PIL打开图片
        #PIL：python imaging library即py图像库转化使用To_PILImage()
        img = Image.open(img_item_path)
        #使用目录名作为标签
        label = self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_path)#返回数据集大小（图片列表长度）


root_dir = "hymenoptera_data/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir,ants_label_dir)
bees_dataset = MyData(root_dir,bees_label_dir)

train_dataset =ants_dataset+bees_dataset
#用法：img,label = dataset[];img.show()

