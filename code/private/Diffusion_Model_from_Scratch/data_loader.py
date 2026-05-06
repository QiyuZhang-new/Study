import torch
import torchvision
import matlotlib.pyplot as plt


def show_images(dataset,num_samples=20,cols=4):
    #设置画布大小
    plt.figure(figsize=(5,5))
    #enumerate提供一个从0开始的计数器
    for i,image in enumerate(dataset):
        if i>=num_samples:
            break
        #将画布划分为网格，之所以是i+1是因为matlab的索引从1开始，历史习惯了
        plt.subplot(num_samples//cols+1,cols,i+1)
        #绘制图片全貌，PyTorch 存储图像的格式是 (C, H, W)，而 Matplotlib 需要 (H, W, C)，因此我们需要转置维度。
        plt.imshow(image.permute(1,2,0))


#数据集采用CIFAR10，train=True表示加载训练集
data=torchvision.datasets.CIFAR10(root="./data",train=True,download=True)
#展示图像示例
show_images(data)