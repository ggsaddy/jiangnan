import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool, global_max_pool, EdgeConv

class GeometryClassifier(torch.nn.Module):
    def __init__(self):
        super(GeometryClassifier, self).__init__()
        
        # 第一层图卷积，将节点特征从 2 维映射到 64 维
        self.conv1 = GCNConv(in_channels=2, out_channels=64)
        
        # 第二层使用 SAGEConv 并添加残差连接，将节点特征从 64 维映射到 128 维
        self.conv2 = SAGEConv(in_channels=64, out_channels=128)
        
        # 第三层图卷积层，使用 EdgeConv 进一步增加特征复杂度
        self.conv3 = EdgeConv(torch.nn.Sequential(
            torch.nn.Linear(128 * 2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128)
        ))

        # 第四层图卷积层，使用 GCNConv 并将特征映射到 256 维
        self.conv4 = GCNConv(in_channels=128, out_channels=256)
        
        # 全局池化层
        self.global_mean_pool = global_mean_pool
        self.global_max_pool = global_max_pool
        
        # 全连接层分类器
        self.fc1 = torch.nn.Linear(in_features=512, out_features=128)
        self.fc2 = torch.nn.Linear(in_features=128, out_features=64)
        self.fc3 = torch.nn.Linear(in_features=64, out_features=2)
        
        # 用于调整 residual 的线性层（将维度从 64 变为 128）
        self.residual_fc = torch.nn.Linear(64, 128)
    
    def forward(self, data):
        # 获取输入数据的特征
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 第一层GCNConv图卷积并激活，使用边特征
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # 第二层SAGEConv图卷积并激活 + 残差连接，使用边特征
        residual = x  # 保存残差
        x = self.conv2(x, edge_index)
        
        # 将 residual 映射到与 x 相同的维度
        residual = self.residual_fc(residual)
        
        # 使用残差连接
        x = F.relu(x + residual)  # 使用残差连接

        # 第三层 EdgeConv 并激活，EdgeConv 层自动使用边特征
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # 第四层 GCNConv 图卷积并激活，使用边特征
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        
        # 全局池化：均值池化和最大池化
        x_mean = self.global_mean_pool(x, batch)
        x_max = self.global_max_pool(x, batch)
        
        # 合并池化后的特征
        x = torch.cat([x_mean, x_max], dim=1)
        
        # 全连接层并激活
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)  # 加入 Dropout 防止过拟合
        
        x = self.fc2(x)
        x = F.relu(x)
        
        # 输出层，得到二分类的概率
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
