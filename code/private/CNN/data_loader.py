import warnings
warnings.filterwarnings("ignore", message=".*align should be passed.*")
import torch
import torchvision
import torchvision.transforms as transforms


BATCH_SIZE = 256

#预处理流水线
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),#随机裁剪图像为32x32，并在裁剪前对图像进行4像素的零填充，增加数据多样性，数据增强手段之一。
    transforms.RandomHorizontalFlip(),#随机水平翻转图像，增加数据多样性，数据增强手段之一。
    transforms.ToTensor(),#将PIL图像或NumPy数组转换为PyTorch张量，形状由(H,W,C)转为(C,H,W)，并将像素值缩放到[0.0, 1.0]范围内。
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))#对三通道图像进行标准化，使用均值(0.5, 0.5, 0.5)和标准差(0.5, 0.5, 0.5)，将像素值从[0.0, 1.0]范围缩放到[-1.0, 1.0]范围内。
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform
)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')