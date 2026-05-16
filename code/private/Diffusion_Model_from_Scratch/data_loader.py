import torch
import torchvision
import matplotlib.pyplot as plt
import scipy


def show_images(dataset,num_samples=20,cols=4):
    #设置画布大小（增大画布以容纳高分辨率子图）
    rows = num_samples // cols
    plt.figure(figsize=(cols * 3, rows * 3))
    #enumerate提供一个从0开始的计数器
    for i,(image,label) in enumerate(dataset):
        if i>=num_samples:
            break
        #将画布划分为网格，之所以是i+1是因为matlab的索引从1开始，历史习惯了
        plt.subplot(rows, cols, i+1)
        #绘制图片全貌，PyTorch 存储图像的格式是 (C, H, W)，而 Matplotlib 需要 (H, W, C)，因此我们需要转置维度。
        plt.imshow(image.permute(1,2,0))
        #显示标签
        plt.title(f'Label: {label}', fontsize=8)
        #关闭坐标轴显示
        plt.axis('off')
    #调整子图间距
    plt.tight_layout()
    #显示图像失败，Linux系统上可能需要安装图形界面，或者使用plt.savefig()将图像保存到文件中查看
    #plt.show()
    plt.savefig(save_path, dpi=150)
    print(f"图像已保存到 {save_path}")

#设置保存路径
save_path="./data/example/stanfordcars_samples.png"
#数据集采用StanfordCars，split="train"表示加载训练集
data=torchvision.datasets.StanfordCars(root="./data",split="train",download=False,transform=torchvision.transforms.ToTensor())
#展示图像示例
show_images(data)