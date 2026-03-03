from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

#python的用法 tensor数据类型
#通过transform.ToTensor去看两个问题
#1.transforms该如何使用
#2.Tensor数据类型相较于普通数据类型有什么区别，用处

img_path = "../dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)#PIL

writer = SummaryWriter("../logs")

#创建一个tensor转换对象
#python知识：可调用对象
#ToTensor类实现了Python的__call__特殊方法，即该类实例可以像函数一样被调用,优点是可以存储状态，能执行功能
tensor_trans = transforms.ToTensor()
#将numpy类型转换为tensor类型
tensor_img = tensor_trans(img)

print(tensor_img)

writer.add_image("Tensor_img",tensor_img)

writer.close()
