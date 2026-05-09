"""
demo_forward.py
演示 DDPM 前向扩散过程：取出一张图片，在不同时间步下展示带噪图像。
"""

import torch
import matplotlib.pyplot as plt
import os

# 从自定义模块导入
from dataset import load_transformed_dataset, show_tensor_image, BATCH_SIZE
from torch.utils.data import DataLoader
from forward import T, forward_diffusion_sample

# 1. 准备数据
data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# 2. 取出单张图像
images, _ = next(iter(dataloader))
single_image = images[0:1]   # (1, 3, 32, 32)

# 3. 可视化前向扩散
plt.figure(figsize=(15, 15))
plt.axis('off')

num_images = 10
stepsize = max(1, int(T / num_images))

for idx in range(0, T, stepsize):
    t = torch.tensor([idx], dtype=torch.int64)
    plt.subplot(1, num_images + 1, int(idx / stepsize) + 1)
    img_noisy, _ = forward_diffusion_sample(single_image, t)
    show_tensor_image(img_noisy)

plt.tight_layout()

# 保存到文件而非显示
save_path = "./data/example/demo_forward_1.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=150)
plt.close()
print(f"前向扩散演示图已保存到 {save_path}")