# 输入图像 (3, 32, 32)
#        │
#    conv1 (3→6, 5×5)     → 提取低层特征（边缘、颜色）
#        │
#    BatchNorm + ReLU + MaxPool(2,2)   → 尺寸缩小，保留显著特征
#        │
#    特征图 (6, 14, 14)
#        │
#    conv2 (6→16, 5×5)    → 组合低层特征，生成中层特征（纹理、形状）
#        │
#    BatchNorm + ReLU + MaxPool(2,2)   → 再次缩小
#        │
#    特征图 (16, 5, 5)
#        │
#    flatten → 全连接层   → 分类决策
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

   def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)      # 增大通道数
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)    # 多加一层
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)           # 32→16→8→4
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

   def forward(self, x):
       x = self.pool(F.relu(self.bn1(self.conv1(x))))
       x = self.pool(F.relu(self.bn2(self.conv2(x))))
       x = self.pool(F.relu(self.bn3(self.conv3(x))))
       x = torch.flatten(x, 1)
       x = F.relu(self.fc1(x))
       x = self.dropout(x)
       x = F.relu(self.fc2(x))
       x = self.fc3(x)
       return x
    
net = Net()
