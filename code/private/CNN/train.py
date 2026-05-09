import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import trainloader, testloader   
from model import Net                             
import time
import datetime
import warnings
warnings.filterwarnings("ignore", message=".*align should be passed.*")# 忽略警告信息

# ---------- 超参数 ----------
BATCH_SIZE = 256          
EPOCHS = 50          
LEARNING_RATE = 0.001
#MOMENTUM = 0.9

# ---------- 设备设置 ----------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

# ---------- 模型实例化 ----------
net = Net().to(device)

# ---------- 损失函数和优化器 ----------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

# ---------- 开始训练前的信息输出 ----------
print("\n========== 训练开始 ==========")
print(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"超参数: Epochs={EPOCHS}, Batch Size={BATCH_SIZE}, Learning Rate={LEARNING_RATE}")
print(f"优化器: Adam")
print(f"训练集批次数: {len(trainloader)}, 测试集批次数: {len(testloader)}\n")

start_time = time.time()

# ---------- 训练循环 ----------
for epoch in range(EPOCHS):
    net.train()                    # 设置为训练模式
    running_loss = 0.0
    total_train = 0
    correct_train = 0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 统计 loss 和训练准确率
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        # 每 2000 个 batch 打印一次（可选保留）
        if i % 2000 == 1999:
            print(f'[{epoch+1}, {i+1}] 中间 loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

    # 计算一个 epoch 的平均训练 loss 和训练准确率
    epoch_loss = running_loss / len(trainloader)   # 最后不足2000的部分也算
    epoch_train_acc = 100 * correct_train / total_train

    # ---------- 每个 epoch 完成后在测试集上评估 ----------
    net.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    test_acc = 100 * correct_test / total_test

    # 打印 epoch 摘要
    print(f'Epoch {epoch+1:3d}/{EPOCHS} | '
          f'Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | '
          f'Test Acc: {test_acc:.2f}%')

print('\n========== 训练完成 ==========')
total_time = time.time() - start_time
print(f"结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"总训练时长: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")

# ---------- 最终测试（已经在 epoch 循环里做过，这里可以重复一次完整报告） ----------
correct = 0
total = 0
net.eval()
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
final_acc = 100 * correct / total
print(f'最终测试集准确率: {final_acc:.2f}%')