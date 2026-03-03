#不知道返回值时，print(type()),或者断点调试
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("../logs")
#convert作用：统一图像通道数和色彩模式
#RGB可处理透明背景(RGBA)
#支持L灰度 1通道 RGB 3通道 CMYK转换 4通道
#L 单通道，表示亮度
#CMYK 四通道，青色，品红色，黄色，黑色 印刷色，常见于打印机
# 各种模式的通道数
modes = {
    "1": 1,      # 1位像素，黑白
    "L": 1,      # 8位像素，灰度
    "P": 1,      # 8位像素，使用调色板映射到其他模式
    "RGB": 3,    # 3x8位像素，真彩色
    "RGBA": 4,   # 4x8位像素，带透明度
    "CMYK": 4,   # 4x8位像素，分色
    "YCbCr": 3,  # 3x8位像素，视频格式
    "LAB": 3,    # 3x8位像素，L*a*b色彩空间
    "HSV": 3,    # 3x8位像素，色相、饱和度、明度
    "I": 1,      # 32位整型像素
    "F": 1       # 32位浮点型像素
}
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
