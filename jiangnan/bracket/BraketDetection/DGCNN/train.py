import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from dataloader import GeometryDataset
from model import GeometryClassifier

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def test(model, loader):
    model.eval()
    correct_0 = 0
    correct_1 = 0
    total_0 = 0
    total_1 = 0

    for data in loader:
        output = model(data)
        pred = output.argmax(dim=1)

        # 分别计算每一类的正确预测数和总数
        correct_0 += ((pred == 0) & (data.y == 0)).sum().item()
        correct_1 += ((pred == 1) & (data.y == 1)).sum().item()
        total_0 += (data.y == 0).sum().item()
        total_1 += (data.y == 1).sum().item()

    # 分别计算每一类的精确率
    accuracy_0 = correct_0 / total_0 if total_0 > 0 else 0
    accuracy_1 = correct_1 / total_1 if total_1 > 0 else 0
    accuracy = (correct_0 + correct_1) / (total_0 + total_1)
    
    # 返回每一类的精确率
    return accuracy, accuracy_0, accuracy_1

# 超参数设置
batch_size = 4
learning_rate = 0.0001
epochs = 100

# 加载数据
dataset = GeometryDataset(root_dir="data_folder")
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型、优化器和损失函数
model = GeometryClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# 训练过程
for epoch in range(1, epochs + 1):
    loss = train(model, train_loader, optimizer, criterion)
    accuracy, accuracy_0, accuracy_1 = test(model, train_loader)
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"./cpkt/geometry_classifier{epoch}_ACC_{accuracy:.4f}.pth")
    print(f"Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Accuracy0: {accuracy_0:.4f}, Accuracy1: {accuracy_1:.4f}")

torch.save(model.state_dict(), "./cpkt/geometry_classifier.pth")