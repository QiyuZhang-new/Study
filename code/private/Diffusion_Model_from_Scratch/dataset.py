from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch


IMG_SIZE=32
BATCH_SIZE=128


def load_transformed_dataset():
    """
    加载 CIFAR‑10 数据集，将训练集和测试集合并，
    并应用统一的预处理：缩放到 32x32，随机水平翻转，转为张量，归一化到 [-1, 1]。
    """
    data_transforms=[
        transforms.Resize((IMG_SIZE,IMG_SIZE)),#调整图像大小为32x32
        transforms.RandomHorizontalFlip(),#随机水平翻转，数据增强
        transforms.ToTensor(),#转为张量，范围[0,1]
        transforms.Lambda(lambda x:x*2.0-1.0)#归一化到[-1,1]，DDPM 的目标是预测标准高斯噪声ϵ∼N(0,I)，其值域就是集中在 [-1, 1] 附近。输入图像与噪声在同一个数值尺度下，损失函数的梯度才会稳定，不会因为数值范围不匹配而发散。这是 DDPM 训练的标配操作。
    ]
    data_transform=transforms.Compose(data_transforms)#组合多个变换步骤
    #加载训练集和测试集，并应用相同的变换
    train=torchvision.datasets.CIFAR10(root="./data",train=True,download=True,transform=data_transform)
    test=torchvision.datasets.CIFAR10(root="./data",train=False,download=True,transform=data_transform)
    #合并训练集和测试集
    return torch.utils.data.ConcatDataset([train,test])

def show_tensor_image(image):
    """
    将 [-1,1] 的张量图像复原为 PIL 图片并显示。
    支持单张图 (C,H,W) 或批量图 (B,C,H,W)，批量时只展示第一张。
    """
    reverse_transforms=transforms.Compose([
        transforms.Lambda(lambda x:(x+1.0)/2.0),#将 [-1,1] 还原到 [0,1]
        transforms.Lambda(lambda t:t.permute(1,2,0)),#将 (C,H,W) 转为 (H,W,C)
        transforms.Lambda(lambda x:x*255.0),#将 [0,1] 还原到 [0,255]
        transforms.Lambda(lambda x:x.cpu().numpy().astype(np.uint8)),#转为 NumPy 数组并转换为 uint8 类型
        transforms.ToPILImage()#转为 PIL 图片
    ])
    #批量图像，取第一张
    if len(image.shape)==4:
        image=image[0,:,:,:]
    #显示并去除坐标轴
    plt.imshow(reverse_transforms(image))
    plt.axis("off")


# 实例化数据集和 DataLoader
data=load_transformed_dataset()
dataloader=DataLoader(data,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)

#（可选）快速测试：展示一个 batch 的样本
if __name__=="__main__":
    images,_=next(iter(dataloader))#获取一个 batch 的图像和标签
    print("一个batch的图像张量形状:",images.shape) #(BATCH_SIZE, 3, 32, 32)
    show_tensor_image(images)#展示一个 batch 的第一张图像
    plt.show()

    
    
