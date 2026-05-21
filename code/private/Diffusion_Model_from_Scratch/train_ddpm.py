import torch
import torch.nn.functional as F  
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from forward import T,forward_diffusion_sample
from unet import SimpleUnet
from dataset import dataloader, IMG_SIZE, BATCH_SIZE
from sample import sample_plot_image      # 从 sample.py 导入

device = "cuda:0" if torch.cuda.is_available() else "cpu"  # CUDA_VISIBLE_DEVICES=2 时 cuda:0 即物理 GPU 2

model = SimpleUnet().to(device)
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 300

# 余弦退火学习率调度：从 lr=1e-3 平滑衰减到 1e-5
# 帮助模型在训练后期精细收敛，避免 loss 震荡
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.mse_loss(noise, noise_pred)  # L2 损失，与 DDPM 理论推导一致

for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        # 根据实际 batch 大小动态生成时间步，避免最后一个 batch 维度不匹配
        actual_bs = batch[0].shape[0]
        t = torch.randint(0, T, (actual_bs,), device=device).long()
        loss = get_loss(model, batch[0], t)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 and step == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item():.4f} | lr: {current_lr:.2e}")
            sample_plot_image(model, device, T, IMG_SIZE)  # 传入 model
    
    scheduler.step()  # 每个 epoch 结束后更新学习率

# 训练结束后保存模型
torch.save(model.state_dict(), "models/ddpm_model.pth")
print("模型已保存到 models/ddpm_model.pth")