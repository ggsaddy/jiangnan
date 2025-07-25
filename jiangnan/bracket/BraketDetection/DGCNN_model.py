import torch
from DGCNN.model import GeometryClassifier
from torch_geometric.data import Data

def filter_by_pretrained_DGCNN_Model(polys, model_path):
    if model_path=="":
        return polys
    # 导入并定义与训练时相同的模型结构
    model = GeometryClassifier()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()  # 将模型设置为评估模式
    
    # 选择设备
    device = torch.device("cpu")
    model = model.to(device)
    
    res = []
    for poly in polys:
        # 计算多边形中心坐标
        num_points = 0
        center_x, center_y = 0.0, 0.0
        
        for seg in poly:
            (start_x, start_y), (end_x, end_y), length = seg.start_point, seg.end_point, seg.length()
            center_x += start_x + end_x
            center_y += start_y + end_y
            num_points += 2  # 每个 seg 有两个点
        
        # 计算中心坐标
        center_x /= num_points
        center_y /= num_points
        
        # 创建节点和边信息列表
        nodes = set()
        edges = []
        edge_features = []
        
        # 处理 poly 中的每个 seg，平移坐标并构建图数据
        for seg in poly:
            (start_x, start_y), (end_x, end_y), length = seg.start_point, seg.end_point, seg.length()
            
            # 平移后的起点和终点
            shifted_start = (start_x - center_x, start_y - center_y)
            shifted_end = (end_x - center_x, end_y - center_y)
            
            # 加入节点集合并记录边信息
            nodes.add(shifted_start)
            nodes.add(shifted_end)
            edges.append([shifted_start, shifted_end])
            edge_features.append([length])  # 边特征为长度
            
        # 将节点映射到索引
        node_list = list(nodes)
        node_index = {node: i for i, node in enumerate(node_list)}
        
        # 构建 PyG 数据格式的边索引和节点特征
        edge_index = torch.tensor([[node_index[edge[0]], node_index[edge[1]]] for edge in edges], dtype=torch.long).t().contiguous()
        x = torch.tensor(node_list, dtype=torch.float)  # 节点特征为节点坐标
        edge_attr = torch.tensor(edge_features, dtype=torch.float)  # 边特征为边长度

        # 构建 PyG 数据对象并转移到设备
        new_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr).to(device)
        
        # 使用模型进行预测
        with torch.no_grad():
            output = model(new_data)
            prediction = output.argmax(dim=1).item()
            
            # 如果预测为类别 1，将整个 poly 添加到结果中
            if prediction == 1:
                res.append(poly)

    return res
