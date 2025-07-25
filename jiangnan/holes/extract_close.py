import json
import math
import numpy as np
import argparse
import os
import time
import ezdxf
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from collections import defaultdict
from load_v2 import DXFConverterV2
from extract_dimen import DimensionExtractor
from draw_dxf import draw_rectangle_in_dxf

class Node:
    def __init__(self, x, y, k=4):
        # 允许两种容差判断 这应该怎么办
        '''
            p1 = (-272.5935, 40262.5374)
            p2 = (-267.5901, 40268.6572)
            219，407
            219，406
        '''
        self.x = int(x)
        self.y = int(y)
        
        # 保存原始坐标用于精确计算距离
        self.orig_x = x
        self.orig_y = y
        
        # 保存容差系数
        self.tolerance = k

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        if other is None:
            return False
        # 使用绝对容差进行比较
        atol = self.tolerance  # 使用容差系数作为绝对容差
        return (abs(self.orig_x - other.orig_x) <= atol and 
                abs(self.orig_y - other.orig_y) <= atol)

    def is_close_to(self, other, atol=None):
        """判断两个节点是否足够接近，使用绝对容差"""
        if other is None:
            return False
        if atol is None:
            atol = self.tolerance
        return (abs(self.orig_x - other.orig_x) <= atol and 
                abs(self.orig_y - other.orig_y) <= atol)

    def __repr__(self):
        return f"({self.x}, {self.y})"

class Graph:
    def __init__(self):
        # 不再使用defaultdict存储图结构
        # 而是使用节点列表和边列表
        self.nodes = []  # 所有节点列表
        self.edges = []  # 所有边列表，每条边是(u, v, entity)的形式
        self.debug_info = {}  # 用于存储调试信息
    
    def add_edge(self, u, v, entity):
        if u is not None and v is not None:
            # 检查节点是否已存在，使用is_close_to而不是==
            u_exists = False
            v_exists = False
            u_idx = -1
            v_idx = -1
            
            for i, node in enumerate(self.nodes):
                if u.is_close_to(node):
                    u_exists = True
                    u_idx = i
                    break
            
            for i, node in enumerate(self.nodes):
                if v.is_close_to(node):
                    v_exists = True
                    v_idx = i
                    break
            
            # 如果节点不存在，则添加到节点列表
            if not u_exists:
                self.nodes.append(u)
                u_idx = len(self.nodes) - 1
            
            if not v_exists:
                self.nodes.append(v)
                v_idx = len(self.nodes) - 1
            
            # 使用实际存储的节点引用
            actual_u = self.nodes[u_idx]
            actual_v = self.nodes[v_idx]
            
            # 检查边是否已存在
            edge_exists = False
            for edge_u, edge_v, _ in self.edges:
                if (actual_u.is_close_to(edge_u) and actual_v.is_close_to(edge_v)) or \
                   (actual_u.is_close_to(edge_v) and actual_v.is_close_to(edge_u)):
                    edge_exists = True
                    break
            
            # 如果边不存在，则添加
            if not edge_exists:
                self.edges.append((actual_u, actual_v, entity))
    
    def get_neighbors(self, node):
        """获取节点的所有邻居"""
        if node is None:
            return []
        
        neighbors = []
        for u, v, _ in self.edges:
            if node.is_close_to(u):
                neighbors.append(v)
            elif node.is_close_to(v):
                neighbors.append(u)
        return neighbors
    
    def get_degree(self, node):
        """获取节点的度"""
        return len(self.get_neighbors(node))
    
    def get_entity_for_edge(self, u, v):
        """获取边(u,v)对应的实体"""
        for node1, node2, entity in self.edges:
            if (u.is_close_to(node1) and v.is_close_to(node2)) or \
               (u.is_close_to(node2) and v.is_close_to(node1)):
                return entity
        return None
    
    def dfs(self, node, visited, parent, path, cycles, debug=False):
        """深度优先搜索找闭合回路，不依赖哈希值"""
        if node is None:
            return

        # 标记当前节点为已访问
        for i, n in enumerate(visited):
            if node.is_close_to(n[0]):
                visited[i] = (n[0], True)
                break
        
        path.append(node)
        
        if debug:
            print(f"DFS访问节点: {node}, 路径: {[n for n in path]}")

        # 获取所有邻居
        neighbors = self.get_neighbors(node)
        
        for neighbor in neighbors:
            if neighbor is None:
                continue
                
            if debug:
                print(f"  检查邻居: {neighbor}")
            
            # 检查邻居是否是父节点
            is_parent = False
            if parent is not None and neighbor.is_close_to(parent):
                is_parent = True
                if debug:
                    print(f"  邻居 {neighbor} 是父节点，跳过")
                continue
            
            # 确定邻居节点的访问状态
            neighbor_visited = False
            neighbor_in_path = False
            path_index = -1
            
            for n, v in visited:
                if neighbor.is_close_to(n):
                    neighbor_visited = v
                    break
            
            # 检查邻居是否在当前路径中
            for idx, p in enumerate(path):
                if neighbor.is_close_to(p):
                    neighbor_in_path = True
                    path_index = idx
                    break
            
            if not neighbor_visited:
                if debug:
                    print(f"  邻居 {neighbor} 未访问过，继续DFS")
                self.dfs(neighbor, visited, node, path, cycles, debug)
            elif not is_parent and neighbor_in_path:
                # 找到闭合环路
                if debug:
                    print(f"  找到环路! 邻居 {neighbor} 已在路径中，位置: {path_index}")
                    
                cycle = path[path_index:]
                
                # 只有当环路至少包含3个节点时才添加
                if len(cycle) >= 3:
                    if debug:
                        print(f"  环路长度 {len(cycle)} >= 3")
                    
                    # 检查是否是已知的环路，避免重复添加
                    is_duplicate = False
                    for existing_cycle in cycles:
                        if len(cycle) == len(existing_cycle):
                            # 检查是否为相同环路的不同起点
                            nodes_match_count = 0
                            for cycle_node in cycle:
                                for existing_node in existing_cycle:
                                    if cycle_node.is_close_to(existing_node):
                                        nodes_match_count += 1
                                        break
                            
                            if nodes_match_count == len(cycle):
                                is_duplicate = True
                                if debug:
                                    print(f"  发现重复环路")
                                break
                                
                    if not is_duplicate:
                        if debug:
                            print(f"  添加有效环路: {cycle}")
                        cycles.append(cycle)
                    elif debug:
                        print(f"  跳过重复环路: {cycle}")
                elif debug:
                    print(f"  环路长度 {len(cycle)} < 3，不是有效环路")

        path.pop()
        if debug:
            print(f"回溯，移除节点 {node} 从路径")

    def find_cycles(self, require_degree_2=True, debug=False):
        """
        查找闭合回路，不依赖哈希值
        
        参数:
        require_degree_2: 是否要求每个节点的度都为2
        debug: 是否输出调试信息
        
        返回:
        List[List[Node]]: 找到的闭合回路列表
        """
        if debug:
            print("\n开始寻找环路:")
            print(f"图中有 {len(self.nodes)} 个节点和 {len(self.edges)} 条边")
            
            # 打印所有节点
            for i, node in enumerate(self.nodes):
                neighbors = self.get_neighbors(node)
                print(f"节点 {i+1}: {node} 连接到 {len(neighbors)} 个邻居")
                for j, n in enumerate(neighbors):
                    print(f"  邻居 {j+1}: {n}")
            
        # 初始化访问状态列表，每个元素是(节点, 是否访问)元组
        visited = [(node, False) for node in self.nodes]
        cycles = []

        # 对每个未访问的节点进行DFS
        for i, (node, is_visited) in enumerate(visited):
            if not is_visited:
                if debug:
                    print(f"\n从节点 {i+1}: {node} 开始DFS")
                # 创建新的visited列表，避免引用问题
                current_visited = [(n, v) for n, v in visited]
                self.dfs(node, current_visited, None, [], cycles, debug)
                
                # 更新主visited列表
                for j, (n, v) in enumerate(current_visited):
                    if v:  # 如果节点在当前DFS中被访问
                        visited[j] = (n, True)
        
        # 如果要求每个点的度都为2
        if require_degree_2:
            valid_cycles = []
            for cycle in cycles:
                is_valid = True
                for node in cycle:
                    if self.get_degree(node) != 2:
                        is_valid = False
                        if debug:
                            print(f"环路无效: 节点 {node} 的度为 {self.get_degree(node)} != 2")
                        break
                if is_valid:
                    valid_cycles.append(cycle)
            cycles = valid_cycles
            
        if debug:
            print(f"\n找到 {len(cycles)} 个环路:")
            for i, cycle in enumerate(cycles):
                print(f"环路 {i+1}: {cycle}")
                
                # 检查环路的连通性
                for j, node in enumerate(cycle):
                    next_node = cycle[(j+1) % len(cycle)]
                    entity = self.get_entity_for_edge(node, next_node)
                    if entity:
                        print(f"  边 {j+1} -> {(j+2) % len(cycle)}: {entity['type']}, 颜色={entity.get('color')}")
                    else:
                        print(f"  边 {j+1} -> {(j+2) % len(cycle)}: 找不到对应实体")
        
        return cycles
    
    def get_cycle_entities(self, cycle):
        """获取环路对应的实体列表"""
        entities = []
        for i in range(len(cycle)):
            node = cycle[i]
            next_node = cycle[(i+1) % len(cycle)]
            entity = self.get_entity_for_edge(node, next_node)
            if entity:
                # 对于arc类型的实体，确保包含mid属性
                entity_copy = entity.copy()
                if entity_copy['type'] == 'arc':
                    # 直接从原始entity中获取mid属性
                    if 'mid' not in entity_copy and 'mid' in entity:
                        entity_copy['mid'] = entity['mid']
                entities.append(entity_copy)
        return entities

class CloseExtractor:
    def __init__(self, json_path: str, tolerance_factor: float = 4.0, scale_factor: float = 1.0, debug: bool = False):
        """
        初始化提取器
        
        参数:
        json_path: allbe.json文件路径
        tolerance_factor: 容差系数
        scale_factor: bbox缩放系数，用于提取dimensions和texts
        debug: 是否启用调试模式
        """
        self.json_path = json_path
        self.tolerance_factor = tolerance_factor
        self.scale_factor = scale_factor
        self.debug = debug
        self.data = self.load_json(json_path)
        
        if self.debug:
            print(f"加载了 {len(self.data['entities'])} 个实体，{len(self.data['dimensions'])} 个尺寸标注，{len(self.data['texts'])} 个文本")
            print(f"缩放系数: {self.scale_factor}")

    def load_json(self, json_path: str) -> dict:
        """加载JSON文件"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def is_point_in_bbox(self, point: List[float], bbox: List[float]) -> bool:
        """判断点是否在检测框内"""
        x, y = point
        x1, y1, x2, y2 = bbox[:4]
        return (min(x1, x2) <= x <= max(x1, x2)) and (min(y1, y2) <= y <= max(y1, y2))

    def filter_entities_by_bbox(self, bbox: List[float]) -> List[Dict]:
        """筛选在bbox内的实体"""
        filtered_entities = []
        
        for entity in self.data['entities']:
            if entity['type'] == 'line':
                if (self.is_point_in_bbox(entity['start'], bbox) or
                    self.is_point_in_bbox(entity['end'], bbox)):
                    filtered_entities.append(entity)
                    
            elif entity['type'] == 'arc':
                if (self.is_point_in_bbox(entity['start'], bbox) or
                    self.is_point_in_bbox(entity['end'], bbox) or
                    self.is_point_in_bbox(entity['center'], bbox)):
                    filtered_entities.append(entity)
                    
            elif entity['type'] == 'circle':
                if self.is_point_in_bbox(entity['center'], bbox):
                    filtered_entities.append(entity)
                    
            elif entity['type'] == 'spline':
                if (self.is_point_in_bbox(entity['start'], bbox) or
                    self.is_point_in_bbox(entity['end'], bbox)):
                    filtered_entities.append(entity)

            elif entity['type'] == 'polyline':
                if (self.is_point_in_bbox(entity['start'], bbox) or
                    self.is_point_in_bbox(entity['end'], bbox)):
                    filtered_entities.append(entity)
                    
            elif entity['type'] == 'ellipse':
                if (self.is_point_in_bbox(entity['start'], bbox) or
                    self.is_point_in_bbox(entity['end'], bbox) or
                    self.is_point_in_bbox(entity['center'], bbox)):
                    filtered_entities.append(entity)
        
        if self.debug:
            print(f"在bbox {bbox} 内找到 {len(filtered_entities)} 个实体（包括line、arc、spline、ellipse等类型）")
            
        return filtered_entities

    def get_all_entities_bbox(self) -> List[float]:
        """获取所有实体的边界框，返回格式为[x1, y1, x2, y2]"""
        all_points = []
        
        for entity in self.data['entities']:
            if entity['type'] == 'line':
                all_points.append(entity['start'])
                all_points.append(entity['end'])
            elif entity['type'] == 'arc':
                all_points.append(entity['start'])
                all_points.append(entity['end'])
                all_points.append(entity['center'])
            elif entity['type'] == 'circle':
                all_points.append(entity['center'])
                # 添加圆的边界点
                radius = entity['radius']
                center = entity['center']
                all_points.append([center[0] - radius, center[1]])
                all_points.append([center[0] + radius, center[1]])
                all_points.append([center[0], center[1] - radius])
                all_points.append([center[0], center[1] + radius])
            elif entity['type'] == 'polyline':
                all_points.append(entity['start'])
                all_points.append(entity['end'])
            elif entity['type'] == 'spline':
                all_points.append(entity['start'])
                all_points.append(entity['end'])
            elif entity['type'] == 'ellipse':
                all_points.append(entity['start'])
                all_points.append(entity['end'])
                all_points.append(entity['center'])
                # 考虑椭圆的主轴和次轴
                if 'major_axis' in entity:
                    major_axis = entity['major_axis']
                    ratio = entity.get('ratio', 1.0)
                    center = entity['center']
                    # 计算长轴的端点
                    major_length = math.sqrt(major_axis[0]**2 + major_axis[1]**2)
                    all_points.append([center[0] + major_axis[0], center[1] + major_axis[1]])
                    # 计算短轴的端点
                    minor_x = -major_axis[1] * ratio
                    minor_y = major_axis[0] * ratio
                    all_points.append([center[0] + minor_x, center[1] + minor_y])
                    all_points.append([center[0] - minor_x, center[1] - minor_y])
        
        if not all_points:
            return [-1e9, -1e9, 1e9, 1e9]  # 默认边界
        
        x_values = [point[0] for point in all_points]
        y_values = [point[1] for point in all_points]
        
        x_min = min(x_values)
        x_max = max(x_values)
        y_min = min(y_values)
        y_max = max(y_values)
        
        return [x_min, y_max, x_max, y_min]  # [左上x, 左上y, 右下x, 右下y]

    def build_graph(self, entities: List[Dict]) -> Graph:
        """构建图"""
        graph = Graph()
        
        if self.debug:
            print(f"\n构建图，使用容差系数 k={self.tolerance_factor}")
        selected_color = [7]
        # 添加所有边到图中
        for entity in entities:
            if entity['color'] not in selected_color:
                continue
            if entity['type'] == 'line':
                start_node = Node(entity['start'][0], entity['start'][1], self.tolerance_factor)
                end_node = Node(entity['end'][0], entity['end'][1], self.tolerance_factor)
                graph.add_edge(start_node, end_node, entity)
                if self.debug:
                    print(f"添加line: 起点={start_node}, 终点={end_node}")
                    
            elif entity['type'] == 'arc':
                start_node = Node(entity['start'][0], entity['start'][1], self.tolerance_factor)
                end_node = Node(entity['end'][0], entity['end'][1], self.tolerance_factor)
                graph.add_edge(start_node, end_node, entity)
                if self.debug:
                    print(f"添加arc: 起点={start_node}, 终点={end_node}")
                    
            elif entity['type'] == 'polyline':
                start_node = Node(entity['start'][0], entity['start'][1], self.tolerance_factor)
                end_node = Node(entity['end'][0], entity['end'][1], self.tolerance_factor)
                graph.add_edge(start_node, end_node, entity)
                if self.debug:
                    print(f"添加polyline: 起点={start_node}, 终点={end_node}")
                    
            elif entity['type'] == 'spline':
                start_node = Node(entity['start'][0], entity['start'][1], self.tolerance_factor)
                end_node = Node(entity['end'][0], entity['end'][1], self.tolerance_factor)
                graph.add_edge(start_node, end_node, entity)
                if self.debug:
                    print(f"添加spline: 起点={start_node}, 终点={end_node}")
                    
            elif entity['type'] == 'ellipse':
                start_node = Node(entity['start'][0], entity['start'][1], self.tolerance_factor)
                end_node = Node(entity['end'][0], entity['end'][1], self.tolerance_factor)
                graph.add_edge(start_node, end_node, entity)
                if self.debug:
                    print(f"添加ellipse: 起点={start_node}, 终点={end_node}")
        
        # 输出图的信息
        if self.debug:
            print(f"\n构建完成的图:")
            node_count = len(graph.nodes)
            edge_count = len(graph.edges)
            print(f"图中有 {node_count} 个节点和 {edge_count} 条边")
            
            for node, neighbors in graph.get_neighbors(None):
                print(f"节点 {node} 连接到 {len(neighbors)} 个邻居: {neighbors}")
        
        return graph

    def extract_closed_components(self, bbox_list: List[List[float]], require_degree_2: bool = True) -> Dict:
        """
        提取闭合连通分量
        
        参数:
        bbox_list: bbox列表
        require_degree_2: 是否要求每个节点的度都为2
        
        返回:
        Dict: 包含闭合连通分量的结果
        """
        result = {
            "closed_components": []
        }
        
        # 如果bbox_list为空，则获取所有实体的边界框
        if not bbox_list:
            bbox = self.get_all_entities_bbox()
            bbox_list = [bbox]
            if self.debug:
                print(f"bbox_list为空，使用所有实体的边界框: {bbox}")
        
        for bbox_idx, bbox in enumerate(bbox_list):
            if self.debug:
                print(f"\n处理bbox {bbox_idx+1}: {bbox}")
                
            # 筛选在bbox内的实体
            filtered_entities = self.filter_entities_by_bbox(bbox)
            
            # 构建图
            graph = self.build_graph(filtered_entities)
            
            # 寻找闭合回路
            cycles = graph.find_cycles(require_degree_2, self.debug)
            
            # 临时存储当前bbox内的所有闭合连通分量
            bbox_components = []
            
            # 为每个闭合回路创建结果
            for cycle_idx, cycle in enumerate(cycles):
                cycle_entities = graph.get_cycle_entities(cycle)
                
                if not cycle_entities:
                    continue
                
                # 计算当前闭合连通分量的实际边界框
                component_bbox = self.calculate_component_bbox(cycle_entities)
                area1 = (component_bbox[2] - component_bbox[0]) * (component_bbox[3] - component_bbox[1])
                area2 = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area1 / area2 < 0.3:
                    continue
                # 使用scale_factor缩放component_bbox来收集dimensions和texts
                scaled_bbox = self.scale_bbox(component_bbox, self.scale_factor)
                filtered_texts = self.filter_texts_by_bbox(scaled_bbox)
                filtered_dimensions = self.filter_dimensions_by_bbox(scaled_bbox)
                
                if self.debug:
                    print(f"  分量 {bbox_idx}_{cycle_idx}: 原始bbox={component_bbox}")
                    print(f"  缩放后bbox={scaled_bbox} (scale_factor={self.scale_factor})")
                    print(f"  找到texts: {len(filtered_texts)}, dimensions: {len(filtered_dimensions)}")
                
                # 收集实体，以及对应的dimensions和texts
                component = {
                    "bbox": bbox,
                    "component_bbox": component_bbox,  # 新增：闭合连通分量自身的边界框
                    "scaled_bbox": scaled_bbox,  # 新增：缩放后的边界框
                    "cycle_id": f"{bbox_idx}_{cycle_idx}",
                    "node_count": len(cycle),
                    "entities": cycle_entities,
                    "dimensions": filtered_dimensions,  # 在缩放后bbox内的dimensions
                    "texts": filtered_texts,  # 在缩放后bbox内的texts
                    "is_panel": False,  # 新增：标记是否为面板结构
                    "panel_entities": []  # 新增：保存面板内部实体
                }
                
                bbox_components.append(component)
            
            # 面板检测：如果当前bbox内有多个闭合连通分量，检测是否为面板结构
            if len(bbox_components) >= 2:
                if self.debug:
                    print(f"  在bbox {bbox_idx+1} 中发现 {len(bbox_components)} 个闭合连通分量，开始面板检测")
                
                # 计算所有分量之间的IoU
                panel_detected = False
                panel_components = []
                
                for i in range(len(bbox_components)):
                    for j in range(i + 1, len(bbox_components)):
                        comp1 = bbox_components[i]
                        comp2 = bbox_components[j]
                        
                        # 计算两个闭合连通分量的IoU
                        iou = calculate_iou(comp1["component_bbox"], comp2["component_bbox"])
                        
                        if self.debug:
                            print(f"    分量 {i+1} 与分量 {j+1} 的IoU: {iou:.4f}")
                        
                        # 如果IoU大于0.9，认为是面板结构
                        if iou > 0.9:
                            panel_detected = True
                            if comp1 not in panel_components:
                                panel_components.append(comp1)
                            if comp2 not in panel_components:
                                panel_components.append(comp2)
                
                if panel_detected:
                    if self.debug:
                        print(f"    检测到面板结构，涉及 {len(panel_components)} 个分量")
                    
                    # 标记所有涉及的分量为面板结构
                    for comp in panel_components:
                        comp["is_panel"] = True
                    
                    # 计算面板分量中bbox最大的那个（按面积计算）
                    max_area = 0
                    max_component = None
                    
                    for comp in panel_components:
                        comp_bbox = comp["component_bbox"]
                        area = (comp_bbox[2] - comp_bbox[0]) * (comp_bbox[3] - comp_bbox[1])
                        if area > max_area:
                            max_area = area
                            max_component = comp
                    
                    if self.debug:
                        print(f"    保留面积最大的分量，面积: {max_area:.2f}")
                    
                    # 将其他面板分量的entities添加到最大分量的panel_entities中
                    if max_component is not None:
                        for comp in panel_components:
                            if comp != max_component:
                                # 将较小分量的entities添加到最大分量的panel_entities中
                                max_component["panel_entities"].extend(comp["entities"])
                                if self.debug:
                                    print(f"    将分量 {comp['cycle_id']} 的 {len(comp['entities'])} 个实体添加到面板内部")
                    
                    # 只保留面积最大的面板分量，其他面板分量不添加到结果中
                    for comp in bbox_components:
                        if comp["is_panel"]:
                            if comp == max_component:
                                result["closed_components"].append(comp)
                                if self.debug:
                                    print(f"    保留面板分量: {comp['cycle_id']}，包含 {len(comp['panel_entities'])} 个内部实体")
                        else:
                            # 非面板分量正常添加
                            result["closed_components"].append(comp)
                else:
                    # 没有检测到面板结构，所有分量都正常添加
                    for comp in bbox_components:
                        result["closed_components"].append(comp)
            else:
                # 只有一个或没有闭合连通分量，直接添加
                for comp in bbox_components:
                    result["closed_components"].append(comp)
        
        if self.debug:
            panel_count = sum(1 for comp in result["closed_components"] if comp["is_panel"])
            print(f"\n找到 {len(result['closed_components'])} 个闭合连通分量，其中 {panel_count} 个为面板结构")
            
        return result

    def rectify_angle(self, angle: float) -> float:
        """角度精度修正"""
        if abs(angle) < 1e-10:
            return 0.0
        if abs(angle - 360) < 1e-10:
            return 360.0
        if abs(angle - 90) < 1e-10:
            return 90.0
        if abs(angle + 90) < 1e-10:
            return 270.0
        if abs(angle - 180) < 1e-10:
            return 180.0
        if abs(angle - 270) < 1e-10:
            return 270.0
        return angle

    def get_arc_points(self, center: List[float], radius: float, 
                      start_angle: float, end_angle: float, 
                      num_points: int = 64) -> List[List[float]]:
        """计算圆弧上的点序列"""
        # 1. 统一精度处理
        radius = round(radius, 6)
        
        # 2. 确保角度是角度制
        if isinstance(start_angle, float) and abs(start_angle) <= 2 * math.pi:
            start_angle = math.degrees(start_angle)
        if isinstance(end_angle, float) and abs(end_angle) <= 2 * math.pi:
            end_angle = math.degrees(end_angle)
            
        # 3. 对角度进行精度修正
        start_angle = self.rectify_angle(start_angle)
        end_angle = self.rectify_angle(end_angle)
        
        # 4. 规范化角度到[0, 360)范围
        start_angle = start_angle % 360
        end_angle = end_angle % 360
        
        # 5. 处理终点角度小于起点角度的情况
        if end_angle < start_angle:
            end_angle += 360
        
        # 6. 处理完整圆的特殊情况
        if abs(end_angle - start_angle - 360) < 1e-10:
            end_angle = start_angle + 360
            
        # 7. 计算角度步长和点坐标
        angle_step = (end_angle - start_angle) / (num_points - 1)
        points = []
        
        for i in range(num_points):
            angle = math.radians(start_angle + i * angle_step)
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append([x, y])
            
        return points

    def visualize(self, result: Dict, output_dir: str = "./visualize"):
        """
        可视化检测结果
        
        参数:
        result: 包含检测结果的字典
        output_dir: 输出目录
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time())
        
        print(f"\n开始可视化 {len(result['closed_components'])} 个闭合连通分量")
        
        # 对每个闭合连通分量进行可视化
        for idx, component in enumerate(result['closed_components']):
            # 创建图像
            fig = plt.figure(figsize=(8, 8), dpi=64)
            ax = fig.add_subplot(111)
            
            # 收集所有坐标点，用于确定绘图范围
            all_points = []
            
            # 处理bbox
            bbox = component['bbox']
            all_points.extend([(bbox[0], bbox[1]), (bbox[2], bbox[3])])
            
            # 处理entities
            for entity in component['entities']:
                if entity['type'] == 'line':
                    all_points.append(entity['start'])
                    all_points.append(entity['end'])
                elif entity['type'] == 'arc':
                    all_points.append(entity['center'])
                    all_points.append(entity['start'])
                    all_points.append(entity['end'])
                elif entity['type'] == 'spline' or entity['type'] == 'ellipse' or entity['type'] == 'polyline':
                    all_points.append(entity['start'])
                    all_points.append(entity['end'])
            
            # 确定坐标范围
            if not all_points:
                continue
                
            all_points = np.array(all_points)
            min_x = np.min(all_points[:, 0])
            max_x = np.max(all_points[:, 0])
            min_y = np.min(all_points[:, 1])
            max_y = np.max(all_points[:, 1])
            
            # 增加边距
            padding = 0.05
            width = max_x - min_x
            height = max_y - min_y
            min_x -= width * padding
            max_x += width * padding
            min_y -= height * padding
            max_y += height * padding
            
            # 计算缩放比例和偏移量
            canvas_size = 512
            scale_x = (canvas_size - 20) / (max_x - min_x)
            scale_y = (canvas_size - 20) / (max_y - min_y)
            scale = min(scale_x, scale_y)
            
            offset_x = 10 - min_x * scale
            offset_y = canvas_size - 10 + min_y * scale  # 注意这里是加，因为y轴方向相反
            
            # 定义坐标转换函数
            def transform_point(x, y):
                px = x * scale + offset_x
                py = offset_y - y * scale
                return px, py
            
            # 绘制bbox（黄色）
            try:
                x1, y1, x2, y2, c = bbox
            except:
                x1, y1, x2, y2 = bbox
            x1, y1 = transform_point(x1, y1)
            x2, y2 = transform_point(x2, y2)
            rect = plt.Rectangle((x1, y2), x2-x1, y1-y2, fill=False, edgecolor='yellow', linewidth=2)
            ax.add_patch(rect)
            
            # 绘制entities（黑色）
            for entity in component['entities']:
                if entity['type'] == 'line':
                    x1, y1 = transform_point(entity['start'][0], entity['start'][1])
                    x2, y2 = transform_point(entity['end'][0], entity['end'][1])
                    ax.plot([x1, x2], [y1, y2], color='black', linewidth=1)
                elif entity['type'] == 'arc':
                    center = entity['center']
                    radius = entity['radius']
                    start_angle = entity['startAngle']
                    end_angle = entity['endAngle']
                    
                    # 获取圆弧上的点
                    arc_points = self.get_arc_points(center, radius, start_angle, end_angle, num_points=32)
                    # 转换坐标并绘制
                    arc_points_transformed = [transform_point(p[0], p[1]) for p in arc_points]
                    xs, ys = zip(*arc_points_transformed)
                    ax.plot(xs, ys, color='black', linewidth=1)
                    
                    # 绘制圆弧的起点和终点
                    start_x, start_y = transform_point(entity['start'][0], entity['start'][1])
                    end_x, end_y = transform_point(entity['end'][0], entity['end'][1])
                    ax.plot(start_x, start_y, 'ro', markersize=4)
                    ax.plot(end_x, end_y, 'go', markersize=4)
                    
                    # 绘制圆心位置
                    center_x, center_y = transform_point(center[0], center[1])
                    ax.plot(center_x, center_y, 'mo', markersize=3)
                elif entity['type'] == 'spline' or entity['type'] == 'ellipse' or entity['type'] == 'polyline':
                    # 将spline和ellipse当作直线处理，连接起点和终点
                    x1, y1 = transform_point(entity['start'][0], entity['start'][1])
                    x2, y2 = transform_point(entity['end'][0], entity['end'][1])
                    ax.plot([x1, x2], [y1, y2], color='black', linewidth=1)
                    
                    # 绘制起点和终点
                    ax.plot(x1, y1, 'ro', markersize=4)
                    ax.plot(x2, y2, 'go', markersize=4)
            
            # 设置图像属性
            ax.set_xlim(0, canvas_size)
            ax.set_ylim(canvas_size, 0)  # 注意y轴方向
            ax.set_aspect('equal')
            ax.axis('off')  # 隐藏坐标轴
            plt.tight_layout()
            
            # 添加标题信息
            panel_text = " (面板结构)" if component.get('is_panel', False) else ""
            plt.title(f"闭合连通分量 {idx+1} - 节点数: {component.get('node_count', 'N/A')}{panel_text}", fontsize=10)
            
            # 保存图像
            filename = f"{output_dir}/component_{idx+1}_{bbox[0]:.0f}-{bbox[1]:.0f}-{bbox[2]:.0f}-{bbox[3]:.0f}_{timestamp}.png"
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            
            if self.debug:
                print(f"  已保存可视化结果到 {filename}")
        
        print(f"完成可视化 {len(result['closed_components'])} 个闭合连通分量")
        
        # 绘制所有连通分量的总体图
        self.visualize_all_components(result, output_dir, timestamp)
        
    def visualize_all_components(self, result: Dict, output_dir: str, timestamp: int):
        """
        将所有闭合连通分量绘制在同一张图上
        
        参数:
        result: 包含检测结果的字典
        output_dir: 输出目录
        timestamp: 时间戳，用于生成唯一文件名
        """
        if not result['closed_components']:
            print("没有闭合连通分量可供绘制")
            return
            
        print("\n开始绘制所有闭合连通分量的总体图")
        
        # 创建图像
        fig = plt.figure(figsize=(12, 12), dpi=100)
        ax = fig.add_subplot(111)
        
        # 收集所有坐标点，用于确定绘图范围
        all_points = []
        
        # 处理所有component
        for component in result['closed_components']:
            # 处理bbox
            bbox = component['bbox']
            all_points.extend([(bbox[0], bbox[1]), (bbox[2], bbox[3])])
            
            # 处理entities
            for entity in component['entities']:
                if entity['type'] == 'line':
                    all_points.append(entity['start'])
                    all_points.append(entity['end'])
                elif entity['type'] == 'arc':
                    all_points.append(entity['center'])
                    all_points.append(entity['start'])
                    all_points.append(entity['end'])
                elif entity['type'] == 'spline' or entity['type'] == 'ellipse' or entity['type'] == 'polyline':
                    all_points.append(entity['start'])
                    all_points.append(entity['end'])
        
        # 确定坐标范围
        if not all_points:
            print("没有坐标点可供绘制")
            return
            
        all_points = np.array(all_points)
        min_x = np.min(all_points[:, 0])
        max_x = np.max(all_points[:, 0])
        min_y = np.min(all_points[:, 1])
        max_y = np.max(all_points[:, 1])
        
        # 增加边距
        padding = 0.05
        width = max_x - min_x
        height = max_y - min_y
        min_x -= width * padding
        max_x += width * padding
        min_y -= height * padding
        max_y += height * padding
        
        # 计算缩放比例和偏移量
        canvas_size = 1000  # 使用更大的画布
        scale_x = (canvas_size - 20) / (max_x - min_x)
        scale_y = (canvas_size - 20) / (max_y - min_y)
        scale = min(scale_x, scale_y)
        
        offset_x = 10 - min_x * scale
        offset_y = canvas_size - 10 + min_y * scale  # 注意这里是加，因为y轴方向相反
        
        # 定义坐标转换函数
        def transform_point(x, y):
            px = x * scale + offset_x
            py = offset_y - y * scale
            return px, py
        
        # 生成不同颜色，每个连通分量一种颜色
        import matplotlib.colors as mcolors
        colors = list(mcolors.TABLEAU_COLORS.values())
        
        # 绘制所有连通分量，使用不同的颜色
        for idx, component in enumerate(result['closed_components']):
            color = colors[idx % len(colors)]
            
            # 绘制bbox
            bbox = component['bbox']
            try:
                x1, y1, x2, y2, c = bbox
            except:
                x1, y1, x2, y2 = bbox
            x1, y1 = transform_point(x1, y1)
            x2, y2 = transform_point(x2, y2)
            rect = plt.Rectangle((x1, y2), x2-x1, y1-y2, fill=False, edgecolor=color, linewidth=1, alpha=0.7)
            ax.add_patch(rect)
            
            # 绘制entities
            for entity in component['entities']:
                if entity['type'] == 'line':
                    x1, y1 = transform_point(entity['start'][0], entity['start'][1])
                    x2, y2 = transform_point(entity['end'][0], entity['end'][1])
                    ax.plot([x1, x2], [y1, y2], color=color, linewidth=1.5)
                elif entity['type'] == 'arc':
                    center = entity['center']
                    radius = entity['radius']
                    start_angle = entity['startAngle']
                    end_angle = entity['endAngle']
                    
                    # 获取圆弧上的点
                    arc_points = self.get_arc_points(center, radius, start_angle, end_angle, num_points=32)
                    # 转换坐标并绘制
                    arc_points_transformed = [transform_point(p[0], p[1]) for p in arc_points]
                    xs, ys = zip(*arc_points_transformed)
                    ax.plot(xs, ys, color=color, linewidth=1.5)
                elif entity['type'] == 'spline' or entity['type'] == 'ellipse' or entity['type'] == 'polyline':
                    # 将spline和ellipse当作直线处理，连接起点和终点
                    x1, y1 = transform_point(entity['start'][0], entity['start'][1])
                    x2, y2 = transform_point(entity['end'][0], entity['end'][1])
                    ax.plot([x1, x2], [y1, y2], color=color, linewidth=1.5)
            
            # 绘制面板内部实体（如果存在）
            if 'panel_entities' in component and component['panel_entities']:
                for entity in component['panel_entities']:
                    if entity['type'] == 'line':
                        x1, y1 = transform_point(entity['start'][0], entity['start'][1])
                        x2, y2 = transform_point(entity['end'][0], entity['end'][1])
                        ax.plot([x1, x2], [y1, y2], color=color, linewidth=1, alpha=0.5, linestyle='--')
                    elif entity['type'] == 'arc':
                        center = entity['center']
                        radius = entity['radius']
                        start_angle = entity['startAngle']
                        end_angle = entity['endAngle']
                        
                        # 获取圆弧上的点
                        arc_points = self.get_arc_points(center, radius, start_angle, end_angle, num_points=32)
                        # 转换坐标并绘制
                        arc_points_transformed = [transform_point(p[0], p[1]) for p in arc_points]
                        xs, ys = zip(*arc_points_transformed)
                        ax.plot(xs, ys, color=color, linewidth=1, alpha=0.5, linestyle='--')
                    elif entity['type'] == 'spline' or entity['type'] == 'ellipse' or entity['type'] == 'polyline':
                        # 将spline和ellipse当作直线处理，连接起点和终点
                        x1, y1 = transform_point(entity['start'][0], entity['start'][1])
                        x2, y2 = transform_point(entity['end'][0], entity['end'][1])
                        ax.plot([x1, x2], [y1, y2], color=color, linewidth=1, alpha=0.5, linestyle='--')
            
            # 添加组件编号标签
            bbox_center_x = (x1 + x2) / 2
            bbox_center_y = (y1 + y2) / 2
            panel_entities_count = len(component.get('panel_entities', []))
            label_text = str(idx+1)
            if panel_entities_count > 0:
                label_text += f"({panel_entities_count})"
            ax.text(bbox_center_x, bbox_center_y, label_text, color=color, fontsize=10, 
                   ha='center', va='center', fontweight='bold', 
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
        
        # 设置图像属性
        ax.set_xlim(0, canvas_size)
        ax.set_ylim(canvas_size, 0)  # 注意y轴方向
        ax.set_aspect('equal')
        ax.axis('off')  # 隐藏坐标轴
        plt.tight_layout()
        
        # 添加标题信息
        panel_count = sum(1 for comp in result['closed_components'] if comp.get('is_panel', False))
        plt.title(f"闭合连通分量总数: {len(result['closed_components'])} (其中面板结构: {panel_count})", fontsize=14)
        
        # 保存图像
        filename = f"{output_dir}/all_components_{timestamp}.png"
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.2, dpi=150)
        plt.close()
        
        print(f"总体图已保存到: {filename}")

    def save_to_json(self, output_path: str, bbox_list: List[List[float]], require_degree_2: bool = True, visualize: bool = False):
        """
        提取闭合连通分量并保存到JSON文件
        
        参数:
        output_path: 输出JSON文件路径
        bbox_list: bbox列表
        require_degree_2: 是否要求每个节点的度都为2
        visualize: 是否进行可视化
        """
        result = self.extract_closed_components(bbox_list, require_degree_2)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        
        print(f"结果已保存到: {output_path}")
        
        # 如果需要可视化，则调用可视化函数
        if visualize:
            self.visualize(result)

    def calculate_component_bbox(self, entities: List[Dict]) -> List[float]:
        """
        计算闭合连通分量的边界框
        
        参数:
        entities: 实体列表
        
        返回:
        List[float]: [x1, y1, x2, y2] 格式的边界框
        """
        if not entities:
            return [0, 0, 0, 0]
        
        all_points = []
        
        for entity in entities:
            if entity['type'] == 'line':
                all_points.append(entity['start'])
                all_points.append(entity['end'])
            elif entity['type'] == 'arc':
                all_points.append(entity['start'])
                all_points.append(entity['end'])
                all_points.append(entity['center'])
                # 添加圆弧的边界点
                center = entity['center']
                radius = entity['radius']
                all_points.append([center[0] - radius, center[1]])
                all_points.append([center[0] + radius, center[1]])
                all_points.append([center[0], center[1] - radius])
                all_points.append([center[0], center[1] + radius])
            elif entity['type'] == 'circle':
                center = entity['center']
                radius = entity['radius']
                all_points.append([center[0] - radius, center[1]])
                all_points.append([center[0] + radius, center[1]])
                all_points.append([center[0], center[1] - radius])
                all_points.append([center[0], center[1] + radius])
            elif entity['type'] == 'spline' or entity['type'] == 'ellipse' or entity['type'] == 'polyline':
                all_points.append(entity['start'])
                all_points.append(entity['end'])
                if entity['type'] == 'ellipse' and 'center' in entity:
                    all_points.append(entity['center'])
        
        if not all_points:
            return [0, 0, 0, 0]
        
        x_values = [point[0] for point in all_points]
        y_values = [point[1] for point in all_points]
        
        x_min = min(x_values)
        x_max = max(x_values)
        y_min = min(y_values)
        y_max = max(y_values)
        
        return [x_min, y_min, x_max, y_max]

    def scale_bbox(self, bbox: List[float], scale_factor: float) -> List[float]:
        """
        根据scale_factor缩放bbox，保持中心点不变
        
        参数:
        bbox: [x1, y1, x2, y2] 格式的边界框
        scale_factor: 缩放系数
        
        返回:
        List[float]: 缩放后的边界框
        """
        # 提取原始边界框坐标
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox[:4]
            conf = 1
        else:
            x1, y1, x2, y2, conf = bbox[:5]

        
        # 计算中心点
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # 计算原始宽度和高度
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        # 计算新的宽度和高度
        new_width = width * scale_factor
        new_height = height * scale_factor
        
        # 计算新的边界框坐标
        new_x1 = center_x - new_width / 2
        new_x2 = center_x + new_width / 2
        new_y1 = center_y - new_height / 2
        new_y2 = center_y + new_height / 2
        
        return [new_x1, new_y1, new_x2, new_y2, conf]

    def filter_texts_by_bbox(self, bbox: List[float]) -> List[Dict]:
        """
        筛选在bbox内的texts
        
        参数:
        bbox: [x1, y1, x2, y2] 格式的边界框
        
        返回:
        List[Dict]: 符合条件的texts列表
        """
        filtered_texts = []
        
        if self.debug:
            print(f"筛选文本，总共有 {len(self.data.get('texts', []))} 个文本")
            print(f"使用bbox: {bbox}")
        
        for text_idx, text in enumerate(self.data.get('texts', [])):
            if text.get('type') in ['text', 'mtext']:
                text_in_bbox = False
                
                # 检查是否有insert字段
                if 'insert' in text:
                    insert_point = text['insert']
                    if self.is_point_in_bbox(insert_point, bbox):
                        text_in_bbox = True
                        if self.debug:
                            print(f"  文本 {text_idx}: '{text.get('content', '')}' 通过insert点 {insert_point} 在bbox内")
                
                # 检查是否有bound字段
                elif 'bound' in text:
                    bound = text['bound']
                    if isinstance(bound, dict) and all(key in bound for key in ['x1', 'y1', 'x2', 'y2']):
                        # 计算文本边界框的中心点
                        center_x = (bound['x1'] + bound['x2']) / 2
                        center_y = (bound['y1'] + bound['y2']) / 2
                        center_point = [center_x, center_y]
                        
                        if self.is_point_in_bbox(center_point, bbox):
                            text_in_bbox = True
                            if self.debug:
                                print(f"  文本 {text_idx}: '{text.get('content', '')}' 通过bound中心点 {center_point} 在bbox内")
                        
                        # 也可以检查文本边界框与目标bbox是否有重叠
                        elif not text_in_bbox:
                            text_bbox = [bound['x1'], bound['y1'], bound['x2'], bound['y2']]
                            # 检查两个矩形是否重叠
                            if (max(text_bbox[0], bbox[0]) < min(text_bbox[2], bbox[2]) and
                                max(text_bbox[1], bbox[1]) < min(text_bbox[3], bbox[3])):
                                text_in_bbox = True
                                if self.debug:
                                    print(f"  文本 {text_idx}: '{text.get('content', '')}' 通过bound重叠检测在bbox内")
                
                if text_in_bbox:
                    filtered_texts.append(text)
                elif self.debug:
                    # 输出为什么文本没有被选中
                    if 'insert' in text:
                        print(f"  文本 {text_idx}: '{text.get('content', '')}' insert点 {text['insert']} 不在bbox内")
                    elif 'bound' in text:
                        bound = text['bound']
                        center_x = (bound['x1'] + bound['x2']) / 2
                        center_y = (bound['y1'] + bound['y2']) / 2
                        print(f"  文本 {text_idx}: '{text.get('content', '')}' bound中心点 [{center_x}, {center_y}] 不在bbox内")
                    else:
                        print(f"  文本 {text_idx}: '{text.get('content', '')}' 没有insert或bound字段")
        
        if self.debug:
            print(f"在缩放后bbox {bbox} 内找到 {len(filtered_texts)} 个文本")
            if filtered_texts:
                for idx, text in enumerate(filtered_texts):
                    print(f"  选中文本 {idx+1}: '{text.get('content', '')}'")
            
        return filtered_texts

    def filter_dimensions_by_bbox(self, bbox: List[float]) -> List[Dict]:
        """
        筛选在bbox内的dimensions
        
        参数:
        bbox: [x1, y1, x2, y2] 格式的边界框
        
        返回:
        List[Dict]: 符合条件的dimensions列表
        """
        filtered_dimensions = []
        
        for dimension in self.data.get('dimensions', []):
            if dimension.get('type') == 'dimension':
                # 检查defpoint1到defpoint5中是否有任何一个在bbox内
                found_point_in_bbox = False
                for i in range(1, 6):
                    defpoint_key = f'defpoint{i}'
                    if defpoint_key in dimension:
                        defpoint = dimension[defpoint_key]
                        if self.is_point_in_bbox(defpoint, bbox):
                            found_point_in_bbox = True
                            break
                
                if found_point_in_bbox:
                    filtered_dimensions.append(dimension)
        
        if self.debug:
            print(f"在缩放后bbox {bbox} 内找到 {len(filtered_dimensions)} 个尺寸标注")
            
        return filtered_dimensions

def calculate_iou(box1: list, box2: list) -> float:
    """
    计算两个边界框的IoU值
    参数:
    box1, box2: [x1, y1, x2, y2] 格式的边界框
    返回:
    float: IoU值
    """
    # Extract coordinates (ignore confidence values if any)
    x1_1, y1_1, x2_1, y2_1 = min(box1[0], box1[2]), min(box1[1], box1[3]), max(box1[0], box1[2]), max(box1[1], box1[3])
    x1_2, y1_2, x2_2, y2_2 = min(box2[0], box2[2]), min(box2[1], box2[3]), max(box2[0], box2[2]), max(box2[1], box2[3])
    
    # Calculate intersection coordinates
    x_left = max(x1_1, x1_2)
    x_right = min(x2_1, x2_2)
    y_top = max(y1_1, y1_2)
    y_bottom = min(y2_1, y2_2)
    
    # Check if there is intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Calculate areas
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou

def main():
    parser = argparse.ArgumentParser(description="提取闭合连通分量并保存为JSON")
    parser.add_argument("--input", "-i", default="allbe.json", help="输入allbe.json文件路径")
    parser.add_argument("--output", "-o", default="close.json", help="输出close.json文件路径")
    parser.add_argument("--bbox", "-b", nargs="+", type=float, action="append", 
                        help="bbox坐标，格式为x1 y1 x2 y2 [conf]，可以多次使用-b添加多个bbox")
    parser.add_argument("--bbox_file", "-bf", help="包含bbox列表的文件路径")
    parser.add_argument("--tolerance", "-t", type=float, default=5, help="容差系数")
    parser.add_argument("--scale_factor", "-s", type=float, default=1.5, help="bbox缩放系数，用于提取dimensions和texts")
    parser.add_argument("--allow_any_degree", "-a", action="store_true", 
                        help="允许任意度数的节点，不要求度为2")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--visualize", "-v", action="store_true", help="可视化结果")
    parser.add_argument("--dxf_path", "-d", type=str, default="xxx.dxf", help="dxf path")
    parser.add_argument("--output_path", type=str, default="xxx.json", help="output path")
    
    args = parser.parse_args()
    
    # 获取bbox列表
    # bbox_list = [[-1371, 38413, 725, 40552, 1]]
    bbox_list = []
    # bbox_list = [[20772, 38846, 23017, 40981, 0.1]]
    
    
    # bbox_list = [[15792, 1751, 18548, 2713, 0.1]]
    # bbox_list = [[20797, 38882, 23041, 40998, 1]]


    # bbox_list = [[48977, 20806, 52508, 24366, 1]]
    # bbox_list = [[-795.4, 35248, 137.6, 36155, 1]]
    # bbox_list = [[32983, 7280, 38910, 14349, 1]]

    # 如果提供了bbox参数，解析bbox
    if args.bbox:
        for bbox in args.bbox:
            if len(bbox) >= 4:
                if len(bbox) == 4:
                    # 添加默认置信度
                    bbox.append(1.0)
                bbox_list.append(bbox)
    
    # 如果提供了bbox文件，从文件中加载bbox
    if args.bbox_file:
        bbox_list = []
        with open(args.bbox_file, 'r') as f:
            for line in f:
                bbox = list(map(float, line.strip().split(',')))
                if len(bbox) >= 4:
                    if len(bbox) == 4:
                        # 添加默认置信度
                        bbox.append(1.0)
                    bbox_list.append(bbox)
    else:
        converter_pred = DXFConverterV2("Holes")
        bbox_list, _ = converter_pred.convert_file(args.dxf_path, args.output_path)
    
    # 创建提取器并保存结果
    extractor = CloseExtractor(args.input, args.tolerance, args.scale_factor, args.debug)
    extractor.save_to_json(args.output, bbox_list, not args.allow_any_degree, args.visualize)
    print(f"提取完成，结果已保存到: {args.output}")

    if not args.bbox_file:
        bbox_list = []
        with open(args.output, "r", encoding='utf-8') as f:
            data = json.load(f)
        for item in data['closed_components']:
            if len(item["bbox"]) == 4:
                bbox = item["bbox"] + [1]
            else:
                bbox = item["bbox"] # 读取到置信度
            ret = {
                "x1": bbox[0],
                "y1": bbox[1],
                "x2": bbox[2],
                "y2": bbox[3],
                "conf": bbox[4]
            }
            bbox_list.append(ret)
        
        # 绘制最终结果
        # final_path = args.dxf_output_path.replace(".dxf", "_final.dxf")
        draw_rectangle_in_dxf(args.dxf_path, "./out_post", bbox_list)
        print(f"闭合连通分量处理完成，结果已保存到 ./out_post")
        
        # 信息提取
        allbe_path = "allbe_detailed.json"  # 包含所有实体信息的文件
        close_path = args.output  # 包含检测到的目标的文件
        output_path = f"extracted_dimensions-{args.dxf_path.split('/')[-1].replace('.dxf', '')}.json"  # 输出文件
        
        # 设置容差
        t1 = 1.0  # 判断点是否在线上的容差
        t2 = 1.0  # 找到参考线的容差
        slope_tolerance = 0.1  # 判断直线平行的斜率容差
        parallel_distance_threshold = 999980.0  # 平行线之间的距离阈值
        midpoint_distance_threshold = 9999400.0  # 线段中点之间的距离阈值
        
        # 创建提取器并运行
        extractor = DimensionExtractor(
            allbe_path=allbe_path,
            close_path=close_path,
            t1=t1,
            t2=t2,
            debug=False,
            slope_tolerance=slope_tolerance,
            parallel_distance_threshold=parallel_distance_threshold,
            midpoint_distance_threshold=midpoint_distance_threshold
        )
        
        # 执行提取并保存结果
        results = extractor.save_results(output_path)
        
        print(f"Dimension提取完成！")
        print(f"处理了 {len(results)} 个检测目标")
        print(f"结果已保存到: {output_path}")
        
        # 统计信息
        total_dimensions = sum(len(r['extracted_dimensions']) for r in results)
        total_texts = sum(len(r['extracted_texts']) for r in results)
        total_reference_lines = sum(len(r['reference_lines']) for r in results)
        total_stiffeners = sum(len(r.get('stiffeners', [])) for r in results)
        
        print(f"总计提取:")
        print(f"  Dimensions: {total_dimensions}")
        print(f"  Texts: {total_texts}")
        print(f"  参考线: {total_reference_lines}")
        print(f"  Stiffeners: {total_stiffeners}")


        bbox_list = []
        handles = set()
        with open(f"extracted_dimensions-{args.dxfname.replace('.dxf', '')}.json", "r", encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            # bboxes
            if len(item["bbox"]) == 4:
                bbox = item["bbox"] + [1]
            else:
                bbox = item["bbox"] 
            ret = {
                "x1": bbox[0],
                "y1": bbox[1],
                "x2": bbox[2],
                "y2": bbox[3],
                "conf": bbox[4]
            }
            bbox_list.append(ret)
            # entities handles
            for entity in item["entities"]:
                handles.add(entity.get("handle", None))
            # extracted_dimensions
            for dimension in item["extracted_dimensions"]:
                handles.add(dimension.get("dimension", {}).get("handle", None))
            # extracted_texts
            for text in item["extracted_texts"]:
                handles.add(text.get("handle", None))
            # reference_lines
            for line in item["reference_lines"]:
                handles.add(line.get("handle", None))
            # stiffeners
            for stiffener in item["stiffeners"]:
                handles.add(stiffener.get("handle", None))

        draw_rectangle_in_dxf(args.dxf_path, "./out_post_final", bbox_list)
        # modify layer name
        print("修改图层到 holes_detected")
        doc = ezdxf.readfile("./out_post_final/{}_Holes.dxf".format(args.dxf_path.split('/')[-1].replace(".dxf", "")))
        msp = doc.modelspace()
        for handle in handles:
            entity = doc.entitydb[handle]
            entity.dxf.layer = "holes_detected"
        doc.saveas("./out_post_final/{}_final.dxf".format(args.dxf_path.split('/')[-1].replace(".dxf", "")))

if __name__ == "__main__":
    main()



'''
添加1ine：起点=（-99367，382634），终点=（-100529，382634）
添加1ine：起点=（-99998，385619），终点=（-98678,384298）
添加1ine：起点=（-102004，384788），终点=（-102004，384109）
添加1ine：起点=（-99980，385637），终点=（-98660，384316）
添加1ine：起点=（-102029，384788），终点=（-102029,384109）
添加1ine：起点=（-99367，382609），终点=（-100529,382609）
添加1ine：起点=（-99146，384767），终点=（-99455，384168)
添加1ine：起点=（-99455，384168），終点=（-100727,384168）
添加1ine：起点=（-98301，385817），终点=（-98167, 385683）
添加1ine：起点=（-98178，385835），终点=（-98231,385888）
添加1ine：起点=（-100751，382880），终点=（-100612，383076)
添加1ine：起点=（-100612，383076），终点=（-100897，383279)
添加1ine：起点=（-100897，383279），终点=（-101182，383481)
'''