from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

#python的用法 tensor数据类型
#通过transform.ToTensor去看两个问题
#1.transforms该如何使用
#2.Tensor数据类型相较于普通数据类型有什么区别，用处

img_path = "dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

print(tensor_img)

writer.add_image("Tensor_img",tensor_img)

writer.close()