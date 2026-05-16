# inference.py
import torch
from unet import SimpleUnet
from forward import T
from dataset import IMG_SIZE
from sample import sample_plot_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载训练好的模型
model = SimpleUnet().to(device)
model.load_state_dict(torch.load("models/ddpm_model.pth", map_location=device))
model.eval()
print(f"模型已加载，参数量: {sum(p.numel() for p in model.parameters()):,}")

# 生成并可视化完整去噪过程
sample_plot_image(model, device, T, IMG_SIZE)