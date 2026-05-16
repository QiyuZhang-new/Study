# sample.py
import torch
import matplotlib.pyplot as plt
from forward import (
    get_index_from_list, betas, sqrt_one_minus_alphas_cumprod,
    sqrt_recip_alphas, posterior_variance, T
)
from dataset import show_tensor_image, IMG_SIZE

@torch.no_grad()
def sample_timestep(model, x, t):
    """单步去噪（需要传入 model）"""
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t.item() == 0:   # t 是单个标量，用 item() 取 Python 值
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image(model, device, T, IMG_SIZE):
    """生成并展示从噪声到图像的完整去噪过程（左：纯噪声 → 右：最终生成图）"""
    img = torch.randn((1, 3, IMG_SIZE, IMG_SIZE), device=device)
    plt.figure(figsize=(20, 5))
    plt.axis('off')
    num_images = 10
    stepsize = int(T / num_images)

    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(model, img, t)
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i / stepsize) + 1)
            show_tensor_image(img.detach().cpu())
            plt.title(f't={i}', fontsize=8)
    
    plt.tight_layout()
    plt.savefig("outputs/ddpm_sample.png", dpi=150)
    plt.close()
    print("采样结果已保存到 outputs/ddpm_sample.png")