import torch
import torch.nn.functional as F


#1.噪声调度（生成beta序列）
def linear_beta_schedule(timesteps, start=0.0001, end=0.02)->torch.Tensor:
    """
    生成一个长度为 timesteps 的 beta 序列，从 start 线性递增到 end。每一步的 beta_t 表示该步添加的高斯噪声的方差。
    """
    return torch.linspace(start, end, timesteps)


#定义总步数和beta序列
T=1000
betas=linear_beta_schedule(T)


#2.预计算前向过程所需的全部系数
alphas=1.0-betas
alphas_cumprod=torch.cumprod(alphas,dim=0)#overline(alpha_t)序列
alphas_cumprod_prev=F.pad(alphas_cumprod[:-1],(1,0),value=1.0)#overline(alpha_(t-1))序列，第一项补一个1.0
sqrt_alphas_cumprod=torch.sqrt(alphas_cumprod)#sqrt(overline(alpha_t))序列
sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0-alphas_cumprod)#sqrt(1-overline(alpha_t))序列
sqrt_recip_alphas=torch.rsqrt(alphas)#sqrt(1/(alpha_t))序列
posterior_variance=betas*(1.0-alphas_cumprod_prev)/ (1.0-alphas_cumprod)#后验方差序列sigma_t^2

#3.工具函数
def get_index_from_list(vals: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """
    从预计算的系数数组 vals (形状 [T]) 中，按照时间步 t 取出对应的值，
    并将形状调整为 (batch_size, 1, 1, 1) 以便与图像张量进行广播。

    参数:
        vals:   形状为 (T,) 的一维张量，存储每一步的某个系数
        t:      形状为 (batch_size,) 的整数张量，每个元素是一个时间步索引
        x_shape:图像 x 的形状，例如 (batch_size,  channels, height, width)
    返回:
        形状为 (batch_size, 1, 1, 1) 的张量，每个样本取到自己的系数
    """
    batch_size=t.shape[0]
    out=vals.gather(-1,t.cpu())#从vals中按照t索引取值，结果形状为(batch_size,)
    return out.reshape(batch_size, *((1,)*(len(x_shape)-1))).to(t.device)#调整形状为(batch_size, 1, 1, 1)
    

def forward_diffusion_sample(x_0: torch.Tensor, t: torch.Tensor, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """
    给定干净图像 x_0 和时间步 t，利用闭式解一步生成带噪图像 x_t，
    同时返回添加的真实噪声（供训练时作为标签）。

    参数:
        x_0:   干净图像，形状 (batch_size, C, H, W)
        t:     时间步索引，形状 (batch_size,)
        device:计算设备 ('cpu' 或 'cuda')
    返回:
        x_t:   带噪图像，形状同 x_0
        noise: 添加的高斯噪声，形状同 x_0
    """
    noise=torch.randn_like(x_0)#生成与x_0形状相同的标准正态分布噪声
    sqrt_alphas_cumprod_t=get_index_from_list(sqrt_alphas_cumprod,t,x_0.shape)#获取sqrt(overline(alpha_t))，形状为(batch_size, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t=get_index_from_list(sqrt_one_minus_alphas_cumprod,t,x_0.shape)#获取sqrt(1-overline(alpha_t))，形状为(batch_size, 1, 1, 1)
    x_t=sqrt_alphas_cumprod_t.to(device)*x_0.to(device)+sqrt_one_minus_alphas_cumprod_t.to(device)*noise.to(device)#根据闭式解生成x_t
    return x_t, noise.to(device)