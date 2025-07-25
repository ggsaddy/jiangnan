import os
import torch
from torch_geometric.data import Data, Dataset, DataLoader

class GeometryDataset(Dataset):
    def __init__(self, root_dir):
        # 调用父类初始化
        super().__init__(root=root_dir)
        self.root_dir = root_dir
        self.file_list = []
        
        # 递归遍历根目录，找到所有子文件夹下的 txt 文件
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.txt'):
                    self.file_list.append(os.path.join(root, file))
        print("num_of_data:", len(self.file_list))
        
        # 添加 _indices 属性
        self._indices = None

    def len(self):
        return len(self.file_list)
    
    def get(self, idx):
        file_path = self.file_list[idx]
        
        # 读取文件并解析
        edges = []
        edge_features = []
        nodes = set()
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
            # 文件最后一行是标签
            label = int(lines[-1].strip())
            
            # 处理前面的边数据
            for line in lines[:-1]:
                start_x, start_y, end_x, end_y, length = map(float, line.strip().split())
                
                # 每条边的起始点和终点坐标
                start_node = (start_x, start_y)
                end_node = (end_x, end_y)
                
                # 将节点加入集合，以构建所有的节点
                nodes.add(start_node)
                nodes.add(end_node)
                
                # 记录边信息
                edges.append([start_node, end_node])
                edge_features.append([length])  # 边特征可以包括长度或其他特征
        
        # 将节点映射到索引
        node_list = list(nodes)
        node_index = {node: i for i, node in enumerate(node_list)}
        
        # 构建 PyG 数据格式的边索引
        edge_index = torch.tensor([[node_index[edge[0]], node_index[edge[1]]] for edge in edges], dtype=torch.long).t().contiguous()
        
        # 将节点坐标作为节点特征
        x = torch.tensor(node_list, dtype=torch.float)
        
        # 将边长度作为边特征
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        # 标签作为图的全局属性
        y = torch.tensor([label], dtype=torch.long)
        
        # 创建图数据
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data

# # 使用 GeometryDataset 和 DataLoader
# root_dir = "data_folder"  # 指定数据文件夹路径
# dataset = GeometryDataset(root_dir=root_dir)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# # 遍历数据
# for data in dataloader:
#     print(data)
