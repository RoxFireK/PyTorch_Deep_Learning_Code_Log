#炼丹初步入门
import torchvision
from torch import nn
from torchvision.models import VGG16_Weights

#train_data = torchvision.datasets.ImageNet(root='../data_image_net', split='train',download =True,transform=torchvision.transforms.ToTensor())
#下到C:\Users\Admin/.cache\torch\hub\checkpoints\vgg16-397923af.pth记得删
vgg16_false = torchvision.models.vgg16(weights = None)
vgg16_true = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)

#print出来看每层都是啥
print(vgg16_true)

dataset = torchvision.datasets.CIFAR10("../dataset",train=False,transform = torchvision.transforms.ToTensor(),download = False)
#在已有模型里添加层,参数:层名称，层参数
vgg16_true.classifier.add_module('add_Linear',nn.Linear(1000,10))
#在已有模型里修改层
vgg16_false.classifier[6] = (nn.Linear(4096,10))
print(vgg16_false)
print(vgg16_true)
print(vgg16_false)
