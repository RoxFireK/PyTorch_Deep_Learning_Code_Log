#不知道返回值时，print(type()),或者断点调试
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img =Image.open("soyo.png").convert("RGB")
print(img)

#ToTensor使用
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("totensor", img_tensor)


#Normalize归一化
#print(img_tensor[0][0][0])
#第一个列表：RGB均值；第二个列表：RGB标准差
trans_norm = transforms.Normalize([0.1,0.3,0.2],[0.5,0.5,0.5])
img_norm = trans_norm(img_tensor)
#print(img_norm[0][0][0])
writer.add_image("norm", img_norm,3)

#Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
img_Resize = trans_resize(img)
#img_Resize PIL -> totensor ->img_Resize tensor
img_Resize = trans_totensor(img_Resize)
writer.add_image("resize", img_Resize)
print(img_Resize)

#Compose - resize -2适用于大量数据转换
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("resize_2", img_resize_2,1)

#RandomCrop随机裁剪
trans_random = transforms.RandomCrop((200,200))
trans_compose_2 = transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("crop_2", img_crop,i)
writer.close()