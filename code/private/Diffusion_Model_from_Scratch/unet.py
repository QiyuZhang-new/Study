import torch
import math
from torch import nn

# x_t (B, 3, H, W)      t (B,)
#     │                     │
#     │ conv0               │ time_mlp → t (B, time_emb_dim)
#     ▼                     │
#  (B, 64, H, W)           │
#     │                     │
#     │ 下采样×5 ────────── t (传入每个 Block)
#     │   Block0: 64→128  (H/2, W/2)
#     │   Block1: 128→256 (H/4, W/4) → Attention(256ch)
#     │   Block2: 256→512 (H/8, W/8) → Attention(512ch)
#     │   Block3: 512→1024(H/16,W/16)
#     │ 存储跳跃连接
#     ▼
#  (B, 1024, H/32, W/32) ← 最底层
#     │
#     │ 上采样×5 ────────── t + 拼接跳跃连接
#     │   Block0: 1024→512 → Attention(512ch)
#     │   Block1: 512→256  → Attention(256ch)
#     │   Block2: 256→128
#     │   Block3: 128→64
#     ▼
#  (B, 64, H, W)
#     │
#     │ output (1×1 Conv)
#     ▼
# 预测噪声 (B, 3, H, W)


class AttentionBlock(nn.Module):
    """
    多头自注意力模块，用于捕捉全局依赖关系。
    在低分辨率特征图（16×16、8×8）上使用，帮助模型理解整体结构。
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert self.head_dim * num_heads == channels, "channels 必须能被 num_heads 整除"

        self.gnorm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x

        # GroupNorm 归一化
        x = self.gnorm(x)

        # 生成 Q, K, V
        qkv = self.qkv(x)  # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)  # 各 (B, C, H, W)

        # 重塑为多头格式: (B, num_heads, head_dim, H*W)
        q = q.reshape(B, self.num_heads, self.head_dim, H * W)
        k = k.reshape(B, self.num_heads, self.head_dim, H * W)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W)

        # 缩放点积注意力
        scale = self.head_dim ** -0.5
        attn = torch.softmax(q.transpose(-2, -1) @ k * scale, dim=-1)  # (B, heads, H*W, H*W)
        out = (v @ attn.transpose(-2, -1))  # (B, heads, head_dim, H*W)

        # 恢复形状
        out = out.reshape(B, C, H, W)
        out = self.proj(out)

        return out + residual  # 残差连接


class Block(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, up: bool = False):
        super().__init__()
        # 将时间嵌入映射到与输出通道相同的维度
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        if up:
            # 上采样时输入通道数翻倍（因为要拼接跳跃连接的特征）
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            # 转置卷积用于上采样，尺寸翻倍
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            # 步长为2的卷积用于下采样，尺寸减半
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        # GroupNorm 替代 BatchNorm：在扩散模型中训练更稳定，不依赖 batch 统计量
        num_groups = 32 if out_ch % 32 == 0 else out_ch
        self.gnorm1 = nn.GroupNorm(num_groups, out_ch)
        self.gnorm2 = nn.GroupNorm(num_groups, out_ch)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 第一次卷积
        h = self.gnorm1(self.relu(self.conv1(x)))
        # 时间嵌入处理并注入
        time_emb = self.relu(self.time_mlp(t))  # (B, out_ch)
        # 扩展最后两个维度以匹配特征图形状 (B, out_ch, H, W)
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        # 第二次卷积
        h = self.gnorm2(self.relu(self.conv2(h)))
        # 下采样或上采样
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    """
    将时间步 t（整数）编码为正弦位置嵌入，类似于 Transformer 的位置编码。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        # 计算频率指数
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # 时间步与频率相乘
        embeddings = time[:, None] * embeddings[None, :]
        # 分别计算 sin 和 cos 并拼接
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SimpleUnet(nn.Module):
    """
    简化版 U‑Net，专为 DDPM 设计。
    输入：带噪图像 x_t (batch, 3, H, W) 和时间步 t (batch,)
    输出：预测的噪声 (batch, 3, H, W)
    """

    def __init__(self):
        super().__init__()
        image_channels = 3
        # 下采样通道数 (共5层)
        down_channels = (64, 128, 256, 512, 1024)
        # 上采样通道数 (对称)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3  # 输出噪声的通道数（RGB）
        time_emb_dim = 32  # 时间嵌入的维度

        # 时间嵌入：正弦位置编码 + 线性层 + ReLU
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        # 初始卷积
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # 下采样路径（5个 Block）
        self.downs = nn.ModuleList([
            Block(down_channels[i], down_channels[i+1], time_emb_dim)
            for i in range(len(down_channels) - 1)
        ])

        # 注意力层：在低分辨率处（16×16 和 8×8）插入，捕捉全局结构
        # down_channels: (64, 128, 256, 512, 1024)
        # Block 0: 64→128 (32→16), Block 1: 128→256 (16→8)
        # Block 2: 256→512 (8→4),   Block 3: 512→1024 (4→2)
        # 在 256ch (16×16) 和 512ch (8×8) 处加注意力
        self.attn_down1 = AttentionBlock(256)   # 16×16
        self.attn_down2 = AttentionBlock(512)   # 8×8

        # 上采样路径（5个 Block, up=True）
        self.ups = nn.ModuleList([
            Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True)
            for i in range(len(up_channels) - 1)
        ])

        # 上采样路径中的注意力（对称）
        # up_channels: (1024, 512, 256, 128, 64)
        # Block 0: 1024→512 (2→4), Block 1: 512→256 (4→8)
        # Block 2: 256→128 (8→16), Block 3: 128→64 (16→32)
        # 在 512ch (4×4) 和 256ch (8×8) 处加注意力
        self.attn_up1 = AttentionBlock(512)     # 4×4
        self.attn_up2 = AttentionBlock(256)     # 8×8

        # 最终输出卷积（1×1 卷积，不改变空间尺寸）
        self.output = nn.Conv2d(up_channels[-1], out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # 1. 时间编码
        t = self.time_mlp(timestep)  # (B, time_emb_dim)

        # 2. 初始投影
        x = self.conv0(x)

        # 3. 下采样并存储跳跃连接
        residual_inputs = []
        for i, down in enumerate(self.downs):
            x = down(x, t)
            residual_inputs.append(x)
            # 在 256ch (16×16) 和 512ch (8×8) 后插入注意力
            if i == 1:  # 128→256 之后，256ch, 16×16
                x = self.attn_down1(x)
            elif i == 2:  # 256→512 之后，512ch, 8×8
                x = self.attn_down2(x)

        # 4. 上采样并拼接跳跃连接
        for i, up in enumerate(self.ups):
            residual_x = residual_inputs.pop()
            # 在通道维度上拼接（通道数翻倍）
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
            # 在 512ch (4×4) 和 256ch (8×8) 后插入注意力
            if i == 0:  # 1024→512 之后，512ch, 4×4
                x = self.attn_up1(x)
            elif i == 1:  # 512→256 之后，256ch, 8×8
                x = self.attn_up2(x)

        # 5. 输出
        return self.output(x)


# 快速测试（可选）
if __name__ == "__main__":
    model = SimpleUnet()
    print("U‑Net 参数量:", sum(p.numel() for p in model.parameters()))
    # 模拟一次前向
    dummy_x = torch.randn(2, 3, 32, 32)
    dummy_t = torch.randint(0, 300, (2,))
    out = model(dummy_x, dummy_t)
    print("输出形状:", out.shape)  # 应为 (2, 3, 32, 32)