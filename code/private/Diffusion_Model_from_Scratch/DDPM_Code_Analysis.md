# DDPM (Denoising Diffusion Probabilistic Models) 代码分析报告

## 一、项目概览

本项目实现了一个**从零搭建的 DDPM（去噪扩散概率模型）**，基于 CIFAR-10 数据集进行训练和图像生成。代码结构清晰、模块化良好，共包含 **8 个 Python 源文件**，完整覆盖了数据加载、前向扩散、U-Net 模型、训练循环、采样生成和推理五大核心环节。

### 文件结构总览

```
Diffusion_Model_from_Scratch/
├── dataset.py          # 数据集加载与预处理
├── data_loader.py      # 数据可视化工具（独立脚本）
├── forward.py          # 前向扩散过程与噪声调度
├── unet.py             # U-Net 骨干网络定义
├── train_ddpm.py       # 训练主循环
├── inference.py        # 模型推理脚本
├── sample.py           # 反向去噪采样
├── demo_forward.py     # 前向扩散过程演示
├── models/
│   └── ddpm_model.pth  # 训练好的模型权重
├── outputs/
│   └── ddpm_sample.png # 采样结果图
└── data/
    ├── cifar-10-batches-py/  # CIFAR-10 原始数据
    └── example/              # 示例输出图片
```

---

## 二、各文件详细分析

### 2.1 `dataset.py` — 数据集加载与预处理

**核心功能：**

| 函数/变量 | 功能 |
|-----------|------|
| `load_transformed_dataset()` | 加载 CIFAR-10 训练集+测试集，合并后应用数据增强与归一化 |
| `show_tensor_image(image)` | 将 [-1,1] 的张量还原为 PIL 图像并显示 |
| `IMG_SIZE = 32` | 图像尺寸常量 |
| `BATCH_SIZE = 128` | 批次大小常量 |

**预处理管线：**

```
原始图像 (0-255 uint8)
  → Resize(32×32)
  → RandomHorizontalFlip()          # 数据增强
  → ToTensor()                      # 转为 [0,1] 浮点张量
  → Lambda(lambda x: x*2.0 - 1.0)  # 归一化到 [-1, 1]
```

**设计亮点：**
- 将训练集和测试集**合并**（`ConcatDataset`），最大化训练数据量（共 60,000 张），这在生成模型中是常见做法——我们关心生成质量而非分类精度。
- 归一化到 `[-1, 1]` 而非 `[0, 1]`，与标准高斯噪声 $\epsilon \sim \mathcal{N}(0, I)$ 的数值范围一致，有助于损失函数的梯度稳定性。这是 DDPM 训练的**标配操作**。
- 提供了 `show_tensor_image` 的反归一化工具函数，支持单张和批量图像的可视化。

**不足之处：**
1. `download=True` 在每次运行时都会尝试下载，如果数据已存在可能产生不必要的网络请求（虽然 torchvision 会自动跳过）。
2. 缺少对数据加载失败的异常处理。
3. `show_tensor_image` 中硬编码了 `plt.imshow` 调用，但没有 `plt.show()` 或 `plt.close()`，可能导致内存泄漏（在非交互式环境下）。

---

### 2.2 `data_loader.py` — 数据可视化工具

**核心功能：** 独立的可视化脚本，展示 CIFAR-10 样本图像网格。

**关键代码逻辑：**
```python
data = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=False,
    transform=torchvision.transforms.ToTensor()
)
show_images(data)  # 绘制 20 张样本的网格图
```

**设计亮点：**
- 使用 `plt.savefig()` 而非 `plt.show()`，适配无图形界面的 Linux 服务器环境。
- 通过 `permute(1,2,0)` 正确处理 PyTorch (C,H,W) 到 Matplotlib (H,W,C) 的维度转换。

**不足之处：**
1. 该文件与 `dataset.py` **功能重叠**——两者都加载了 CIFAR-10 数据，但使用了不同的 transform（此处仅为 `ToTensor()`，未归一化到 [-1,1]），存在不一致性。
2. `show_images` 函数中 `save_path` 是全局变量，耦合度高，不利于复用。
3. `num_samples//cols+1` 的行数计算在 `num_samples` 不能被 `cols` 整除时会产生多余的空子图行。
4. 该文件实际上是**独立脚本**而非模块，没有被其他文件导入使用，可考虑合并到 `dataset.py` 中。

---

### 2.3 `forward.py` — 前向扩散过程

**核心功能：** 实现 DDPM 的前向扩散（加噪）过程及相关数学公式的预计算。

**关键组件：**

#### (1) 噪声调度
```python
def linear_beta_schedule(timesteps, start=0.0001, end=0.01):
    return torch.linspace(start, end, timesteps)
```
- 采用**线性调度**：$\beta_t$ 从 $10^{-4}$ 线性增长到 $0.01$，共 $T=1000$ 步。
- 这是原始 DDPM 论文的默认设置。

#### (2) 预计算系数

| 变量 | 数学含义 | 用途 |
|------|---------|------|
| `betas` | $\beta_t$ | 每步噪声方差 |
| `alphas` | $\alpha_t = 1 - \beta_t$ | 信号保留率 |
| `alphas_cumprod` | $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ | 累积信号保留率 |
| `sqrt_alphas_cumprod` | $\sqrt{\bar{\alpha}_t}$ | 前向采样：$x_t$ 的信号分量 |
| `sqrt_one_minus_alphas_cumprod` | $\sqrt{1 - \bar{\alpha}_t}$ | 前向采样：$x_t$ 的噪声分量 |
| `sqrt_recip_alphas` | $1/\sqrt{\alpha_t}$ | 反向采样：预测 $x_0$ |
| `posterior_variance` | $\sigma_t^2$ | 反向采样：后验方差 |

#### (3) 前向扩散采样（闭式解）
```python
def forward_diffusion_sample(x_0, t):
    noise = torch.randn_like(x_0)
    x_t = √ᾱ_t · x_0  +  √(1-ᾱ_t) · noise
    return x_t, noise
```
利用重参数化技巧，**一步到位**地从 $x_0$ 生成任意时间步 $t$ 的 $x_t$，无需迭代 $t$ 步。同时返回添加的真实噪声作为训练标签。

#### (4) `get_index_from_list` 工具函数
从预计算的 (T,) 系数数组中按时间步索引取值，并 reshape 为 `(B,1,1,1)` 以支持与图像张量的广播操作。

**设计亮点：**
- 所有系数在**模块加载时预计算**，避免训练时重复计算，效率高。
- `get_index_from_list` 使用 `.gather()` 而非循环索引，高效且支持 GPU。
- 注释详尽，每个变量都标注了数学含义。

**不足之处：**
1. 所有系数都是**全局变量**（模块级），在多进程训练（如 `DataLoader(num_workers>0)`）时可能被复制多份，浪费内存。
2. 噪声调度仅实现了线性调度，未提供余弦调度等其他选项（原始 DDPM 论文指出余弦调度效果更好）。
3. `posterior_variance` 的计算使用了简化公式 $\sigma_t^2 = \beta_t \cdot \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}$，但 DDPM 论文中也讨论了使用 $\beta_t$ 或 $\tilde{\beta}_t$ 两种选择，这里没有提供灵活性。
4. `forward_diffusion_sample` 的 `device` 参数默认为 `"cpu"`，但在实际使用中（`train_ddpm.py`）会传入 `device`，默认值可能引起混淆。

---

### 2.4 `unet.py` — U-Net 骨干网络

**核心功能：** 专为 DDPM 设计的简化版 U-Net，接收带噪图像 $x_t$ 和时间步 $t$，输出预测的噪声 $\epsilon_\theta(x_t, t)$。

**网络架构：**

```
输入: x_t (B, 3, 32, 32)           t (B,)
        │                              │
   conv0 (3→64)                   SinusoidalPositionEmbeddings
        │                              │
        │                         Linear(32→32) + ReLU
        │                              │
   ┌────▼──────────────────────────────┘
   │  Block0: 64→128  (下采样)  ← t_emb
   │  Block1: 128→256 (下采样)  ← t_emb
   │  Block2: 256→512 (下采样)  ← t_emb
   │  Block3: 512→1024(下采样)  ← t_emb
   │  Block4: 1024→1024(下采样) ← t_emb
   │     (最底层: B, 1024, 1, 1)
   │  Block5: 1024→512 (上采样) ← t_emb + skip
   │  Block6: 512→256  (上采样) ← t_emb + skip
   │  Block7: 256→128  (上采样) ← t_emb + skip
   │  Block8: 128→64   (上采样) ← t_emb + skip
   │  Block9: 64→64    (上采样) ← t_emb + skip
   └────┬──────────────────────────────┘
        │
   output (1×1 Conv, 64→3)
        │
  预测噪声: (B, 3, 32, 32)
```

**关键组件分析：**

#### (1) `SinusoidalPositionEmbeddings`
- 将整数时间步 $t$ 编码为 32 维的正弦位置向量。
- 公式：$\text{PE}(t, 2i) = \sin(t \cdot e^{-2i \cdot \log(10000)/d})$，$\text{PE}(t, 2i+1) = \cos(t \cdot e^{-2i \cdot \log(10000)/d})$
- 源自 Transformer 的位置编码，具有平滑的连续性和良好的外推能力。

#### (2) `Block` — U-Net 基本构建块
每个 Block 包含：
1. **时间注入**：通过 `time_mlp` 将时间嵌入映射到输出通道维度，然后以加法方式注入特征图。
2. **两次卷积**：`Conv2d(3×3) → BatchNorm → ReLU → Conv2d(3×3) → BatchNorm → ReLU`
3. **尺度变换**：
   - 下采样：`Conv2d(4×4, stride=2, padding=1)`，尺寸减半
   - 上采样：`ConvTranspose2d(4×4, stride=2, padding=1)`，尺寸翻倍

#### (3) `SimpleUnet` — 完整 U-Net
- **下采样路径**：5 个 Block，通道数 64→128→256→512→1024，空间尺寸 32→16→8→4→2→1。
- **跳跃连接**：下采样时存储中间特征，上采样时在通道维度拼接（`torch.cat`），保留细粒度空间信息。
- **上采样路径**：5 个 Block，通道数 1024→512→256→128→64→64，空间尺寸 1→2→4→8→16→32。
- **输出层**：1×1 卷积，将 64 通道映射为 3 通道（RGB 噪声预测）。

**设计亮点：**
- 架构简洁高效，总参数量适中（约 50M），适合 32×32 的小图像。
- 时间嵌入通过**加法注入**（而非拼接），计算效率高，是 DDPM U-Net 的标准做法。
- 跳跃连接保留了空间细节，对生成质量至关重要。
- 每个 Block 内的时间 MLP 独立学习，增强了时间条件的表达能力。

**不足之处：**
1. **缺少 Self-Attention 层**：现代 DDPM 实现通常在低分辨率层（如 16×16 或 8×8）插入自注意力机制，以捕捉全局依赖关系。本实现完全没有注意力机制，可能限制生成质量。
2. **缺少 GroupNorm**：使用了 `BatchNorm2d`，在小 batch size 下表现不稳定。DDPM 论文及其后续工作通常使用 **Group Normalization** 替代 BatchNorm。
3. **时间嵌入维度固定为 32**：对于 1000 步的扩散过程，32 维可能稍显不足，通常使用 128 或 256 维。
4. **上采样 Block 的通道处理**：上采样 Block 的 `conv1` 期望输入通道为 `2*in_ch`（因为拼接了跳跃连接），但 `in_ch` 参数传入的是拼接前的通道数，容易造成理解混淆。
5. **最底层无特殊处理**：在最底层（1×1 空间尺寸），特征已高度压缩，通常可以在此处加入额外的自注意力或 MLP 处理。
6. **缺少 Dropout**：训练时未使用 Dropout 正则化，可能导致过拟合（虽然 CIFAR-10 上通常还好）。
7. **输出层无激活函数**：`output` 卷积后直接输出，这在预测噪声时是正确的（噪声值域为 $\mathbb{R}$），但缺少注释说明。

---

### 2.5 `train_ddpm.py` — 训练主循环

**核心功能：** 训练 U-Net 模型预测添加的噪声。

**训练流程：**
```
for epoch in range(100):
    for batch in dataloader:
        1. 随机采样时间步 t ~ Uniform(0, T-1)
        2. 前向扩散：生成 x_t 和真实噪声 noise
        3. 模型预测：noise_pred = model(x_t, t)
        4. 计算 L1 Loss：|noise - noise_pred|
        5. 反向传播 + 参数更新
        6. 每 5 个 epoch 采样一次并保存图片
```

**关键设计决策：**

| 参数 | 值 | 说明 |
|------|-----|------|
| 优化器 | Adam, lr=0.001 | 标准选择 |
| 损失函数 | L1 Loss (MAE) | 原始论文使用 L2 (MSE)，L1 对异常值更鲁棒 |
| 训练轮数 | 100 epochs | 对于 CIFAR-10 偏少，通常需要 200-500 epochs |
| Batch Size | 128 | 适合 32×32 图像 |
| 采样频率 | 每 5 epochs | 便于观察训练进度 |

**设计亮点：**
- 使用 **L1 Loss** 而非 L2 Loss——实践中 L1 有时能产生更清晰的图像，且对异常值更鲁棒。
- 每 5 个 epoch 进行一次采样可视化，便于监控训练进度和生成质量。
- 训练结束后自动保存模型权重到 `ddpm_model.pth`。

**不足之处：**
1. **没有学习率调度**：100 epochs 全程使用固定 lr=0.001。通常应使用余弦退火或阶梯衰减，帮助后期精细收敛。
2. **没有 Exponential Moving Average (EMA)**：EMA 是扩散模型训练的标配技巧，能显著提升采样质量。本实现缺少 EMA 模型。
3. **随机时间步采样使用 `torch.randint`**：虽然均匀采样是标准做法，但某些改进工作（如"重要性采样"）会非均匀采样时间步。
4. **没有梯度裁剪**：训练扩散模型时梯度可能不稳定，添加梯度裁剪（如 `max_norm=1.0`）是常见做法。
5. **没有验证集或 FID 评估**：仅凭肉眼观察采样结果难以客观衡量模型质量。缺少 FID（Fréchet Inception Distance）等量化指标。
6. **采样在训练循环内执行**：`sample_plot_image` 包含 `plt.savefig`，会短暂阻塞训练。更好的做法是异步保存或使用独立的评估脚本。
7. **`BATCH_SIZE` 重复定义**：`dataset.py` 中已定义 `BATCH_SIZE=128`，此处又定义了一次，存在冗余且可能不一致。
8. **没有断点续训功能**：如果训练中断，无法从 checkpoint 恢复，需从头开始。
9. **数据未移动到 device**：`batch[0]` 在传入 `get_loss` 前未显式 `.to(device)`，虽在 `forward_diffusion_sample` 中处理了，但不够清晰。

---

### 2.6 `sample.py` — 反向去噪采样

**核心功能：** 从纯噪声出发，逐步去噪生成图像。

**采样算法（DDPM 标准采样）：**

```
输入: 纯噪声 x_T ~ N(0, I)
for t = T-1, ..., 0:
    1. 预测 x_0:  x̂_0 = (x_t - √(1-ᾱ_t) · ε_θ(x_t, t)) / √ᾱ_t
    2. 计算均值: μ_θ = (1/√α_t) · (x_t - (β_t/√(1-ᾱ_t)) · ε_θ(x_t, t))
    3. 计算方差: σ_t²
    4. 采样:     x_{t-1} = μ_θ + σ_t · z  (t>0 时 z~N(0,I); t=0 时 z=0)
输出: x_0
```

**关键函数：**

| 函数 | 功能 |
|------|------|
| `sample_timestep(model, x, t)` | 执行单步去噪，从 $x_t$ 得到 $x_{t-1}$ |
| `sample_plot_image(model, device, T, IMG_SIZE)` | 完整采样过程 + 可视化（每 T/10 步保存一张中间结果） |

**设计亮点：**
- 使用 `@torch.no_grad()` 装饰器，避免采样时构建计算图，节省内存。
- `torch.clamp(img, -1.0, 1.0)` 将图像值裁剪到合法范围，防止误差累积导致数值溢出。
- 可视化展示了从纯噪声到清晰图像的**完整去噪过程**，直观展示模型行为。

**不足之处：**
1. **采样速度慢**：完整采样需要 1000 步（T=1000），每次都要调用模型。未实现 DDIM 等加速采样方法。
2. **`sample_timestep` 期望 `t` 是标量张量**（使用 `t.item()`），但在 `sample_plot_image` 中传入的是 `torch.full((1,), i)`（形状为 (1,)），设计不一致。虽然能工作（因为只有 1 个元素），但不够健壮。
3. **`posterior_variance_t` 的计算**：代码中使用了预计算的 `posterior_variance`，但未提供使用 `beta_t` 作为方差的选项（DDPM 论文中两种选择在 T 较大时几乎等价，但 T 较小时有差异）。
4. **缺少 Classifier-Free Guidance (CFG)**：现代扩散模型几乎必用 CFG 来提升生成质量，本实现完全没有条件生成能力。
5. **采样结果固定保存为 `ddpm_sample.png`**：多次采样会覆盖之前的结果。
6. **缺少批量采样功能**：每次只能生成 1 张图像。

---

### 2.7 `demo_forward.py` — 前向扩散演示

**核心功能：** 可视化前向扩散过程——展示同一张图片在 10 个不同时间步下的噪声化程度。

**设计亮点：**
- 直观展示了 DDPM 前向过程的本质：**信号逐渐被噪声淹没**。
- 使用 `os.makedirs` 确保保存目录存在，避免文件保存失败。

**不足之处：**
1. 与 `train_ddpm.py` 一样，重新实例化了 `dataloader`，代码重复。
2. 硬编码了保存路径 `./data/example/demo_forward_1.png`。

---

## 三、整体架构评价

### 3.1 优点

| 方面 | 评价 |
|------|------|
| **代码可读性** | ⭐⭐⭐⭐⭐ 注释详尽，变量命名清晰，每个函数都有中文注释说明 |
| **模块化设计** | ⭐⭐⭐⭐ 前向过程、U-Net、训练、采样各自独立，职责分明 |
| **数学正确性** | ⭐⭐⭐⭐⭐ 严格遵循 DDPM 论文公式，系数预计算准确 |
| **教学价值** | ⭐⭐⭐⭐⭐ 从零实现，无复杂框架依赖，非常适合学习 DDPM 原理 |
| **可运行性** | ⭐⭐⭐⭐ 依赖简单（仅 torch + torchvision + matplotlib），可直接运行 |

### 3.2 主要不足与改进建议

#### 🔴 高优先级

1. **缺少 Self-Attention 层**
   - 建议：在 U-Net 的 16×16 和 8×8 分辨率层之间插入多头自注意力模块。
   - 参考实现：
     ```python
     class AttentionBlock(nn.Module):
         def __init__(self, channels):
             super().__init__()
             self.norm = nn.GroupNorm(32, channels)
             self.qkv = nn.Conv2d(channels, channels*3, 1)
             self.proj = nn.Conv2d(channels, channels, 1)
         def forward(self, x):
             B, C, H, W = x.shape
             q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
             # ... scaled dot-product attention ...
     ```

2. **缺少 EMA（指数移动平均）**
   - 建议：在训练循环中维护 EMA 模型，采样时使用 EMA 权重。
   - 实现：
     ```python
     ema_model = copy.deepcopy(model)
     ema_decay = 0.9999
     # 每次参数更新后：
     for ema_param, param in zip(ema_model.parameters(), model.parameters()):
         ema_param.data.mul_(ema_decay).add_(param.data, alpha=1-ema_decay)
     ```

3. **缺少学习率调度**
   - 建议：添加余弦退火或 ReduceLROnPlateau。
   ```python
   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
   ```

#### 🟡 中优先级

4. **BatchNorm → GroupNorm**
   - 建议：将所有 `BatchNorm2d` 替换为 `GroupNorm(num_groups=32)`，提高小 batch 稳定性。

5. **缺少 FID 评估**
   - 建议：定期（如每 10 epochs）生成 10000 张图像计算 FID，量化评估生成质量。

6. **缺少断点续训**
   - 建议：定期保存 checkpoint（包含 model, optimizer, epoch, scheduler 状态）。

7. **代码重复**
   - `data_loader.py` 与 `dataset.py` 功能重叠，建议合并。
   - `BATCH_SIZE` 在 `dataset.py` 和 `train_ddpm.py` 中重复定义。

#### 🟢 低优先级

8. **缺少 DDIM 加速采样**
   - 建议：实现 DDIM 采样器，将采样步数从 1000 减少到 50-100 步。

9. **缺少条件生成**
   - 建议：添加类别条件嵌入，实现 Classifier-Free Guidance。

10. **采样结果覆盖问题**
    - 建议：按时间戳或 epoch 编号命名保存的采样图片。

---

## 四、DDPM 数学公式对照

以下是代码中实现的核心 DDPM 公式：

### 前向过程
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)$$

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### 反向过程
$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)$$

### 训练目标（简化版）
$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|_1 \right]$$

> 注：原始论文使用 L2 损失，本实现使用 L1 损失。

---

## 五、运行指南

### 环境依赖
```bash
conda create -n ddpm python=3.10
conda activate ddpm
pip install torch torchvision matplotlib
```

### 运行步骤
```bash
cd /data1/zhangqiyu/study/Diffusion_Model_from_Scratch

# 1. 查看数据样本
python data_loader.py

# 2. 演示前向扩散过程
python demo_forward.py

# 3. 训练模型
python train_ddpm.py

# 4. 查看采样结果
# 训练过程中每 5 epochs 自动保存 ddpm_sample.png
```

---

## 六、总结

本项目是一个**优秀的 DDPM 教学实现**，代码量精简（总计约 300 行核心代码），但完整覆盖了扩散模型的核心流程。特别适合初学者理解 DDPM 的工作原理。主要改进方向集中在：

1. **架构升级**：添加 Self-Attention、GroupNorm、EMA
2. **训练优化**：学习率调度、梯度裁剪、断点续训
3. **评估完善**：FID 量化评估、条件生成、加速采样

这些改进可以将一个教学级实现提升为接近 SOTA 的研究级实现。
