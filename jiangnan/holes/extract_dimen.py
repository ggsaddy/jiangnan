import json
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from typing import List, Dict, Tuple
from collections import defaultdict
import re
import pandas as pd
import argparse
from collections import Counter
from load_v2 import DXFConverterV2

class DimensionExtractor:
    def __init__(self, allbe_path: str, close_path: str, t1: float = 1.0, t2: float = 1.0, debug: bool = False, 
                 slope_tolerance: float = 0.1, parallel_distance_threshold: float = 30.0, 
                 midpoint_distance_threshold: float = 200.0):
        """
        初始化Dimension提取器
        
        参数:
        allbe_path: allbe.json文件路径，包含所有实体信息
        close_path: close.json文件路径，包含检测到的目标
        t1: 判断点是否在线上的容差
        t2: 找到参考线的容差
        debug: 是否启用调试模式
        slope_tolerance: 判断直线平行的斜率容差
        parallel_distance_threshold: 平行线之间的距离阈值
        midpoint_distance_threshold: 线段中点之间的距离阈值
        """
        self.t1 = t1
        self.t2 = t2
        self.debug = debug
        self.slope_tolerance = slope_tolerance
        self.parallel_distance_threshold = parallel_distance_threshold
        self.midpoint_distance_threshold = midpoint_distance_threshold
        
        # 加载数据
        self.allbe_data = self.load_json(allbe_path)
        self.close_data = self.load_json(close_path)
        
        if self.debug:
            print(f"加载allbe数据: {len(self.allbe_data)} 个项目")
            
            # 处理不同的close_data结构
            close_components = []
            if isinstance(self.close_data, list):
                close_components = self.close_data
                print(f"加载close数据: {len(close_components)} 个检测目标(直接列表)")
            elif isinstance(self.close_data, dict):
                close_components = self.close_data.get('closed_components', [])
                print(f"加载close数据: {len(close_components)} 个检测目标(closed_components)")
            
            if close_components:
                print(f"第一个检测目标包含字段: {list(close_components[0].keys())}")
        
        # 在所有后续处理之前，先扩展简化的实体
        self.expand_entities()

    def load_json(self, json_path: str) -> dict:
        """加载JSON文件"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def expand_entities(self):
        """
        扩展简化的实体（如polyline、spline）为详细的line段
        根据句柄从allbe_detailed.json中找到对应的所有line类型实体进行替换
        """
        if self.debug:
            print(f"\n========== 开始扩展简化实体 ==========")
        
        # 首先从allbe_data中提取所有实体，建立handle索引
        handle_to_entities = {}
        
        # 处理allbe数据结构
        entities_data = None
        if isinstance(self.allbe_data, list) and len(self.allbe_data) > 0:
            entities_data = self.allbe_data[0]
        elif isinstance(self.allbe_data, dict):
            entities_data = self.allbe_data
        
        if entities_data is None:
            if self.debug:
                print("无法从allbe_data中提取entities_data，跳过实体扩展")
            return
        
        # 建立句柄到实体的映射
        if isinstance(entities_data, list):
            for entity in entities_data:
                if isinstance(entity, dict) and 'handle' in entity:
                    handle = entity['handle']
                    if handle not in handle_to_entities:
                        handle_to_entities[handle] = []
                    handle_to_entities[handle].append(entity)
        elif isinstance(entities_data, dict):
            for key, value in entities_data.items():
                if isinstance(value, list):
                    for entity in value:
                        if isinstance(entity, dict) and 'handle' in entity:
                            handle = entity['handle']
                            if handle not in handle_to_entities:
                                handle_to_entities[handle] = []
                            handle_to_entities[handle].append(entity)
        
        if self.debug:
            print(f"从allbe_data中建立了 {len(handle_to_entities)} 个句柄的映射")
        
        # 处理close_data中的每个闭合连通分量
        close_components = []
        if isinstance(self.close_data, list):
            close_components = self.close_data
        elif isinstance(self.close_data, dict):
            close_components = self.close_data.get('closed_components', [])
        
        total_replaced = 0
        
        for comp_idx, component in enumerate(close_components):
            if 'entities' not in component:
                continue
            
            original_entities = component['entities'][:]
            expanded_entities = []
            component_replaced = 0
            
            if self.debug:
                print(f"\n处理闭合连通分量 {comp_idx + 1}，原始包含 {len(original_entities)} 个实体")
            
            for entity_idx, entity in enumerate(original_entities):
                entity_type = entity.get('type', '')
                entity_handle = entity.get('handle', '')
                
                # 检查是否是需要扩展的实体类型
                if entity_type in ['polyline', 'spline'] and entity_handle:
                    if self.debug:
                        print(f"  发现需要扩展的实体: type={entity_type}, handle={entity_handle}")
                    
                    # 从handle映射中查找对应的line实体
                    if entity_handle in handle_to_entities:
                        related_entities = handle_to_entities[entity_handle]
                        line_entities = [e for e in related_entities if e.get('type') == 'line']
                        
                        if line_entities:
                            if self.debug:
                                print(f"    找到 {len(line_entities)} 个对应的line实体")
                                for i, line_entity in enumerate(line_entities):
                                    print(f"      line {i+1}: start={line_entity.get('start')}, end={line_entity.get('end')}")
                            
                            # 添加所有找到的line实体
                            expanded_entities.extend(line_entities)
                            component_replaced += 1
                        else:
                            if self.debug:
                                print(f"    警告: 句柄 {entity_handle} 没有找到对应的line实体，保留原实体")
                            expanded_entities.append(entity)
                    else:
                        if self.debug:
                            print(f"    警告: 句柄 {entity_handle} 在allbe_data中未找到，保留原实体")
                        expanded_entities.append(entity)
                else:
                    # 对于其他类型的实体，直接添加
                    expanded_entities.append(entity)
            
            # 更新组件的实体列表
            component['entities'] = expanded_entities
            total_replaced += component_replaced
            
            if self.debug:
                print(f"  扩展完成: 替换了 {component_replaced} 个实体，现在包含 {len(expanded_entities)} 个实体")
        
        if self.debug:
            print(f"\n========== 实体扩展完成 ==========")
            print(f"总共处理了 {len(close_components)} 个闭合连通分量")
            print(f"总共替换了 {total_replaced} 个简化实体")
            
            # 统计扩展后的实体类型分布
            type_count = {}
            for component in close_components:
                for entity in component.get('entities', []):
                    entity_type = entity.get('type', 'unknown')
                    type_count[entity_type] = type_count.get(entity_type, 0) + 1
            
            print(f"扩展后的实体类型分布: {type_count}")

    def is_point_on_line(self, point: List[float], line_start: List[float], 
                        line_end: List[float], tolerance: float = None) -> bool:
        """
        判断点是否在线段上，考虑误差容忍
        
        参数:
        point: 待判断的点坐标 [x, y]
        line_start: 线段起点坐标 [x, y]
        line_end: 线段终点坐标 [x, y]
        tolerance: 误差容忍值，默认使用self.t1
        
        返回:
        bool: 点是否在线段上
        """
        if tolerance is None:
            tolerance = self.t1
            
        if not point or not line_start or not line_end:
            return False
            
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # 线段长度的平方
        l2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
        if l2 == 0:
            # 线段起点和终点重合的情况
            return np.sqrt((x - x1) ** 2 + (y - y1) ** 2) <= tolerance
        
        # 计算投影点的参数 t
        t = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / l2
        
        # 如果 t 不在 [0,1] 范围内，点不在线段上
        if t < 0 or t > 1:
            return False
        
        # 计算点到线段的实际距离
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        distance = np.sqrt((x - proj_x) ** 2 + (y - proj_y) ** 2)
        
        return distance <= tolerance

    def get_all_entities_from_allbe(self) -> List[Dict]:
        """从allbe数据中提取所有实体"""
        all_entities = []
        
        # 处理allbe数据结构
        entities_data = None
        if isinstance(self.allbe_data, list) and len(self.allbe_data) > 0:
            entities_data = self.allbe_data[0]
        elif isinstance(self.allbe_data, dict):
            entities_data = self.allbe_data
        
        if entities_data is None:
            return all_entities
            
        if isinstance(entities_data, list):
            for entity in entities_data:
                if entity.get('type') in ['line', 'arc', 'circle', 'lwpolyline']:
                    all_entities.append(entity)
        elif isinstance(entities_data, dict):
            for key, value in entities_data.items():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and item.get('type') in ['line', 'arc', 'circle', 'lwpolyline']:
                            all_entities.append(item)
        
        return all_entities

    def get_defpoints_from_dimension(self, dimension: Dict) -> List[List[float]]:
        """从dimension中提取所有非零的defpoint"""
        defpoints = []
        
        if self.debug:
            print(f"    提取defpoints从dimension (handle: {dimension.get('handle')}):")
        
        for i in range(1, 6):  # defpoint1 到 defpoint5
            key = f'defpoint{i}'
            if key in dimension:
                point = dimension[key]
                # 检查点是否不为[0.0, 0.0]
                if point and len(point) >= 2 and not (point[0] == 0.0 and point[1] == 0.0):
                    defpoints.append(point)
                    if self.debug:
                        print(f"      {key}: {point} ✓")
                elif self.debug:
                    print(f"      {key}: {point} (跳过，为零点或无效)")
            elif self.debug:
                print(f"      {key}: 不存在")
        
        if self.debug:
            print(f"    总共提取到 {len(defpoints)} 个有效defpoint")
        
        return defpoints

    def find_all_entities_by_handle(self, handle: str, all_entities: List[Dict]) -> List[Dict]:
        """
        根据handle查找所有相关的实体
        
        参数:
        handle: 要查找的句柄
        all_entities: 所有实体列表
        
        返回:
        List[Dict]: 具有相同handle的所有实体
        """
        related_entities = []
        
        for entity in all_entities:
            if entity.get('handle') == handle:
                related_entities.append(entity)
        
        if self.debug and related_entities:
            print(f"    找到handle '{handle}' 的 {len(related_entities)} 个相关实体")
            for idx, entity in enumerate(related_entities):
                print(f"      实体 {idx+1}: type={entity.get('type')}, start={entity.get('start')}, end={entity.get('end')}")
        
        return related_entities

    def find_dimensions_in_bbox(self, close_component: Dict) -> Tuple[List[Dict], List[Dict]]:
        """
        处理close组件中的dimensions，找到与bbox实体和参考线的关系
        
        参数:
        close_component: close.json中的一个组件
        
        返回:
        Tuple[List[Dict], List[Dict]]: (匹配的dimensions, 参考线列表)
        """
        matched_dimensions = []
        reference_lines = []
        
        # 尝试不同的bbox字段名
        bbox = close_component.get('scaled_bbox')
        if bbox is None:
            bbox = close_component.get('bbox')
        
        if bbox is None:
            if self.debug:
                print("警告: 在组件中没有找到bbox或scaled_bbox字段")
            return matched_dimensions, reference_lines
        
        bbox_entities = close_component.get('entities', [])
        
        # 尝试不同的dimensions字段名
        dimensions = close_component.get('dimensions', [])
        if not dimensions:
            dimensions = close_component.get('extracted_dimensions', [])
        
        # 获取所有allbe实体用于查找参考线
        all_entities = self.get_all_entities_from_allbe()
        
        # 获取bbox中实体的所有线段
        bbox_lines = []
        for entity in bbox_entities:
            if entity['type'] == 'line':
                bbox_lines.append(entity)
            elif entity['type'] == 'lwpolyline' and 'segments' in entity:
                for segment in entity['segments']:
                    if segment['type'] == 'line':
                        bbox_lines.append({
                            'type': 'line',
                            'start': segment['start'],
                            'end': segment['end'],
                            'color': entity.get('color'),
                            'handle': entity.get('handle'),
                            'layerName': entity.get('layerName'),
                            'linetype': entity.get('linetype')
                        })
        
        if self.debug:
            print(f"\n检查bbox {bbox} 中的dimensions")
            print(f"bbox中有 {len(bbox_lines)} 条线段")
            print(f"close组件中有 {len(dimensions)} 个dimensions")
        
        # 遍历close组件中的dimensions
        for dim_idx, dimension in enumerate(dimensions):
            # 获取dimension中的所有非零defpoint
            defpoints = self.get_defpoints_from_dimension(dimension)
            
            if not defpoints:
                if self.debug:
                    print(f"  dimension {dim_idx} (handle: {dimension.get('handle')}): 没有有效的defpoint")
                continue
            
            if self.debug:
                print(f"  dimension {dim_idx} (handle: {dimension.get('handle')}): 找到 {len(defpoints)} 个defpoint: {defpoints}")
            
            # 检查所有defpoint与线段的关系
            points_on_bbox_lines = []
            points_on_reference_lines = []
            
            for point in defpoints:
                if self.debug:
                    print(f"    检查点 {point} 是否在线段上...")
                
                # 检查点是否在bbox中的线段上
                found_on_bbox_line = False
                for line_idx, line in enumerate(bbox_lines):
                    is_on_line = self.is_point_on_line(point, line['start'], line['end'], self.t1)
                    if self.debug:
                        print(f"      bbox线段 {line_idx} ({line['start']} -> {line['end']}): {is_on_line}")
                    
                    if is_on_line:
                        points_on_bbox_lines.append({
                            'point': point,
                            'handle': line.get('handle'),
                            'layerName': line.get('layerName')
                        })
                        found_on_bbox_line = True
                        if self.debug:
                            print(f"    ✓ 点 {point} 在bbox线段上: {line['start']} -> {line['end']}")
                        break
                
                # 检查是否在参考线上
                if self.debug:
                    print(f"    检查点 {point} 是否在参考线上...")
                for entity in all_entities:
                    if entity['type'] == 'line':
                        # 跳过已经在bbox中的线段
                        is_bbox_line = False
                        for bbox_line in bbox_lines:
                            if (entity.get('handle') == bbox_line.get('handle') or
                                (abs(entity['start'][0] - bbox_line['start'][0]) < self.t2 and
                                 abs(entity['start'][1] - bbox_line['start'][1]) < self.t2 and
                                 abs(entity['end'][0] - bbox_line['end'][0]) < self.t2 and
                                 abs(entity['end'][1] - bbox_line['end'][1]) < self.t2)):
                                is_bbox_line = True
                                break
                        
                        if is_bbox_line:
                            continue
                        
                        if self.is_point_on_line(point, entity['start'], entity['end'], self.t2):
                            points_on_reference_lines.append({
                                'point': point,
                                'handle': entity.get('handle'),
                                'layerName': entity.get('layerName')
                            })
                            if self.debug:
                                print(f"    ✓ 点 {point} 在参考线上: {entity['start']} -> {entity['end']} (handle: {entity.get('handle')})")
                            
                            # 收集参考线（避免重复）
                            if not any(ref_line.get('handle') == entity.get('handle') for ref_line in reference_lines):
                                # 找到参考线时，收集相同handle的所有线段
                                related_entities = self.find_all_entities_by_handle(entity.get('handle'), all_entities)
                                for related_entity in related_entities:
                                    if related_entity['type'] == 'line':
                                        # 避免重复添加
                                        if not any(
                                            ref_line.get('handle') == related_entity.get('handle') and 
                                            abs(ref_line['start'][0] - related_entity['start'][0]) < self.t2 and
                                            abs(ref_line['start'][1] - related_entity['start'][1]) < self.t2 and
                                            abs(ref_line['end'][0] - related_entity['end'][0]) < self.t2 and
                                            abs(ref_line['end'][1] - related_entity['end'][1]) < self.t2
                                            for ref_line in reference_lines
                                        ):
                                            reference_lines.append(related_entity)
                                            if self.debug:
                                                print(f"      添加相同handle的线段: {related_entity['start']} -> {related_entity['end']}")
                            break
            
            # 只保留同时有points_on_bbox_lines和points_on_reference_lines的dimension
            if points_on_bbox_lines and points_on_reference_lines:
                # 构建dimension信息
                dim_info = {
                    'dimension': dimension,
                    'measurement': dimension.get('measurement'),
                    'text': dimension.get('text', ''),
                    'textpos': dimension.get('textpos'),
                    'dimtype': dimension.get('dimtype'),
                    'handle': dimension.get('handle'),
                    'points_on_bbox_lines': points_on_bbox_lines,
                    'points_on_reference_lines': points_on_reference_lines
                }
                matched_dimensions.append(dim_info)
                
                if self.debug:
                    print(f"  ✓ 保留dimension {dim_idx}: {len(points_on_bbox_lines)} 个点在bbox线段上, {len(points_on_reference_lines)} 个点在参考线上")
            else:
                if self.debug:
                    print(f"  ✗ 跳过dimension {dim_idx}: 缺少bbox线段点({len(points_on_bbox_lines)})或参考线点({len(points_on_reference_lines)})")
        
        return matched_dimensions, reference_lines

    def is_text_valid(self, text):
        pattern = r'^(R\d+|FB\d+X\d+.*)$'
        return bool(re.match(pattern, text))

    def process_all(self) -> List[Dict]:
        """处理所有检测目标，提取dimensions和texts"""
        results = []
        
        # 处理close数据中的每个检测目标
        close_components = []
        if isinstance(self.close_data, list):
            close_components = self.close_data
        elif isinstance(self.close_data, dict):
            close_components = self.close_data.get('closed_components', [])
            # 如果没有closed_components，尝试从results字段获取
            if not close_components and 'results' in self.close_data:
                close_components = self.close_data['results']
                if self.debug:
                    print(f"使用results字段，找到 {len(close_components)} 个检测目标")
        
        if not close_components:
            if self.debug:
                print("警告: 在JSON文件中没有找到closed_components或results字段")
            return results
            
        for idx, close_item in enumerate(close_components):
            if self.debug:
                print(f"\n========== 处理检测目标 {idx + 1} ==========")
            
            bbox = close_item['bbox']
            bbox_entities = close_item.get('entities', [])
            texts_temp = close_item.get('texts', [])
            texts = []
            for t in texts_temp:
                # some mtext have no content key
                if self.is_text_valid(t.get('content', "")):
                    texts.append(t)

            if self.debug:
                print(f"bbox: {bbox}")
                print(f"bbox中有 {len(bbox_entities)} 个实体")
                print(f"close组件中有 {len(texts)}  个texts")
                print(f"cycle_id: {close_item.get('cycle_id', 'N/A')}")
                print(f"node_count: {close_item.get('node_count', 'N/A')}")
                print(f"is_panel: {close_item.get('is_panel', False)}")
            
            # 寻找相关的dimensions
            matched_dimensions, reference_lines = self.find_dimensions_in_bbox(close_item)
            
            # 提取bbox中的Stiffener线段
            stiffeners_in_bbox = self.extract_stiffeners_in_bbox(close_item)
            
            # 为arc实体添加equal_with_text字段
            processed_entities = self.add_equal_with_text_to_arcs(bbox_entities, texts)
            
            # 构建结果
            result = {
                'bbox': bbox,
                'confidence': bbox[-1],
                'cycle_id': close_item.get('cycle_id'),
                'node_count': close_item.get('node_count'),
                'is_panel': close_item.get('is_panel', False),
                'entities': processed_entities,
                'panel_entities': close_item.get('panel_entities', []),
                'extracted_dimensions': matched_dimensions,
                'extracted_texts': texts,
                'reference_lines': reference_lines,
                'stiffeners': stiffeners_in_bbox,
                'dimension_count': len(matched_dimensions),
                'text_count': len(texts),
                'stiffener_count': len(stiffeners_in_bbox)
            }
            
            results.append(result)
            
            if self.debug:
                print(f"提取结果: {len(matched_dimensions)} 个dimensions, {len(texts)} 个texts, {len(reference_lines)} 条参考线, {len(stiffeners_in_bbox)} 个stiffeners")
        
        return results

    def add_equal_with_text_to_arcs(self, entities: List[Dict], texts: List[Dict]) -> List[Dict]:
        """为arc实体添加equal_with_text字段"""
        processed_entities = []
        
        # 提取所有以R开头的文本和其数值
        r_texts = []
        for text in texts:
            content = text.get('content', '').strip()
            # 匹配R开头后跟数字的模式，支持R100, R400等格式
            match = re.match(r'^R(\d+(?:\.\d+)?)', content, re.IGNORECASE)
            if match:
                radius_value = float(match.group(1))
                # 获取文本中心点
                text_center = self.get_text_center(text)
                if text_center:
                    r_texts.append({
                        'content': content,
                        'radius_value': radius_value,
                        'center': text_center,
                        'text_info': text
                    })
                    if self.debug:
                        print(f"  找到R文本: {content} -> 半径值: {radius_value}, 位置: {text_center}")
        
        if self.debug:
            print(f"  共找到 {len(r_texts)} 个R开头的文本")
        
        # 处理每个实体
        for entity in entities:
            new_entity = entity.copy()
            
            if entity.get('type') == 'arc':
                arc_center = entity.get('center')
                arc_radius = entity.get('radius')
                
                if arc_center and arc_radius is not None:
                    closest_r_text = None
                    min_distance = float('inf')
                    
                    # 找到最近的R文本
                    for r_text in r_texts:
                        distance = math.sqrt(
                            (arc_center[0] - r_text['center'][0]) ** 2 + 
                            (arc_center[1] - r_text['center'][1]) ** 2
                        )
                        if distance < min_distance:
                            min_distance = distance
                            closest_r_text = r_text
                    
                    if closest_r_text:
                        # 比较半径值，容差为1
                        radius_diff = abs(arc_radius - closest_r_text['radius_value'])
                        tolerance = 1.0
                        
                        if radius_diff <= tolerance:
                            new_entity['equal_with_text'] = True
                        else:
                            new_entity['equal_with_text'] = False
                        
                        # 添加调试信息
                        new_entity['closest_text_content'] = closest_r_text['content']
                        new_entity['closest_text_radius'] = closest_r_text['radius_value']
                        new_entity['distance_to_text'] = min_distance
                        new_entity['radius_difference'] = radius_diff
                        
                        if self.debug:
                            print(f"  Arc半径: {arc_radius}, 最近文本: {closest_r_text['content']} "
                                  f"(半径值: {closest_r_text['radius_value']}), 距离: {min_distance:.2f}, "
                                  f"半径差: {radius_diff:.2f}, 匹配: {new_entity['equal_with_text']}")
                    else:
                        new_entity['equal_with_text'] = None
                        if self.debug:
                            print(f"  Arc半径: {arc_radius}, 未找到R文本")
                else:
                    new_entity['equal_with_text'] = None
            
            processed_entities.append(new_entity)
        
        return processed_entities

    def visualize_results(self, results: List[Dict], output_dir: str = "./visualize_dimensions"):
        """可视化提取结果"""
        print(f"可视化 {len(results)} 个提取结果")
        
        # 创建保存目录
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time())
        
        for idx, result in enumerate(results):
            if not result['extracted_dimensions'] and not result['extracted_texts']:
                if self.debug:
                    print(f"跳过结果 {idx + 1}: 没有提取到dimensions或texts")
                continue
            
            # 创建图像
            fig = plt.figure(figsize=(12, 10), dpi=100)
            ax = fig.add_subplot(111)
            
            # 收集所有坐标点，用于确定绘图范围
            all_points = []
            
            # 处理bbox
            bbox = result['bbox']
            all_points.extend([(bbox[0], bbox[1]), (bbox[2], bbox[3])])
            
            # 收集entities的点
            for entity in result['entities']:
                if entity['type'] == 'line':
                    all_points.extend([entity['start'], entity['end']])
                elif entity['type'] == 'arc':
                    all_points.extend([entity['center'], entity['start'], entity['end']])
                elif entity['type'] == 'lwpolyline' and 'segments' in entity:
                    for segment in entity['segments']:
                        if segment['type'] == 'line':
                            all_points.extend([segment['start'], segment['end']])
            
            # 收集dimensions的points
            for dim_info in result['extracted_dimensions']:
                # 收集bbox线段上的点
                for point_info in dim_info.get('points_on_bbox_lines', []):
                    all_points.append(point_info['point'])
                # 收集参考线上的点
                for point_info in dim_info.get('points_on_reference_lines', []):
                    all_points.append(point_info['point'])
                # 收集textpos
                if dim_info.get('textpos'):
                    all_points.append(dim_info['textpos'])
            
            # 收集参考线的点
            for ref_line in result['reference_lines']:
                all_points.extend([ref_line['start'], ref_line['end']])
            
            # 收集Stiffener线段的点
            for stiffener in result.get('stiffeners', []):
                all_points.extend([stiffener['start'], stiffener['end']])
            
            # 收集文本位置
            for text_info in result['extracted_texts']:
                if text_info.get('position'):
                    all_points.append(text_info['position'])
                if text_info.get('bound'):
                    bound = text_info['bound']
                    if isinstance(bound, dict):
                        all_points.extend([(bound.get('x1', 0), bound.get('y1', 0)), 
                                         (bound.get('x2', 0), bound.get('y2', 0))])
            
            if not all_points:
                continue
            
            # 确定坐标范围
            all_points = np.array(all_points)
            min_x = np.min(all_points[:, 0])
            max_x = np.max(all_points[:, 0])
            min_y = np.min(all_points[:, 1])
            max_y = np.max(all_points[:, 1])
            
            # 增加边距
            padding = 0.1
            width = max_x - min_x
            height = max_y - min_y
            min_x -= width * padding
            max_x += width * padding
            min_y -= height * padding
            max_y += height * padding
            
            # 绘制bbox（黄色边框）
            x1, y1, x2, y2 = bbox[:4]  # 只取前4个值，忽略可能的第5个值
            rect = plt.Rectangle((x1, y2), x2-x1, y1-y2, fill=False, edgecolor='yellow', linewidth=3, label='Detection Bbox')
            ax.add_patch(rect)
            
            # 绘制bbox中的entities（黑色）
            entities_labeled = False
            for entity in result['entities']:
                if entity['type'] == 'line':
                    ax.plot([entity['start'][0], entity['end'][0]], 
                           [entity['start'][1], entity['end'][1]], 
                           color='black', linewidth=2, alpha=0.8,
                           label='Bbox Entities' if not entities_labeled else "")
                    entities_labeled = True
                elif entity['type'] == 'arc':
                    # 绘制圆弧
                    center = entity['center']
                    radius = entity['radius']
                    start_angle = entity['startAngle'] 
                    end_angle = entity['endAngle']
                    
                    # 使用matplotlib的Arc类绘制弧线
                    from matplotlib.patches import Arc
                    arc = Arc((center[0], center[1]), 2*radius, 2*radius,
                             angle=0, theta1=start_angle, theta2=end_angle,
                             color='black', linewidth=2, alpha=0.8)
                    ax.add_patch(arc)
                    
                    # 可选：标记圆弧的起点和终点
                    if 'start' in entity and 'end' in entity:
                        ax.plot(entity['start'][0], entity['start'][1], 'ko', markersize=3, alpha=0.8)
                        ax.plot(entity['end'][0], entity['end'][1], 'ko', markersize=3, alpha=0.8)
                elif entity['type'] == 'lwpolyline' and 'segments' in entity:
                    for segment in entity['segments']:
                        if segment['type'] == 'line':
                            ax.plot([segment['start'][0], segment['end'][0]], 
                                   [segment['start'][1], segment['end'][1]], 
                                   color='black', linewidth=2, alpha=0.8,
                                   label='Bbox Entities' if not entities_labeled else "")
                            entities_labeled = True
            
            # 绘制参考线（绿色）
            for ref_idx, ref_line in enumerate(result['reference_lines']):
                ax.plot([ref_line['start'][0], ref_line['end'][0]], 
                       [ref_line['start'][1], ref_line['end'][1]], 
                       color='green', linewidth=2, alpha=0.7, 
                       label='Reference Line' if ref_idx == 0 else "")
            
            # 绘制Stiffener线段（橙色）
            stiffeners_labeled = False
            for stiffener in result.get('stiffeners', []):
                ax.plot([stiffener['start'][0], stiffener['end'][0]], 
                       [stiffener['start'][1], stiffener['end'][1]], 
                       color='orange', linewidth=2, alpha=0.8,
                       label='Stiffener' if not stiffeners_labeled else "")
                stiffeners_labeled = True
            
            # 绘制dimension信息
            points_on_ref_line_labeled = False  # 用于确保蓝色点标签只添加一次
            points_on_bbox_line_labeled = False  # 用于确保红色点标签只添加一次
            dimension_line_labeled = False  # 用于确保dimension线标签只添加一次
            
            for dim_idx, dim_info in enumerate(result['extracted_dimensions']):
                # 绘制dimension线段（红色线）
                dimension = dim_info['dimension']
                defpoints = []
                for i in range(1, 6):
                    point_key = f'defpoint{i}'
                    if point_key in dimension:
                        point = dimension[point_key]
                        if point and len(point) >= 2 and not (point[0] == 0.0 and point[1] == 0.0):
                            defpoints.append(point)
                
                # 连接defpoints绘制dimension线段
                if len(defpoints) >= 2:
                    # 绘制主要的dimension线段（连接前两个点）
                    ax.plot([defpoints[0][0], defpoints[1][0]], 
                           [defpoints[0][1], defpoints[1][1]], 
                           color='red', linewidth=2, alpha=0.8, 
                           label='Dimension Line' if not dimension_line_labeled else "")
                    dimension_line_labeled = True
                    
                    # 如果有第三个点，也连接它（通常是dimension的延伸线）
                    if len(defpoints) >= 3:
                        ax.plot([defpoints[0][0], defpoints[2][0]], 
                               [defpoints[0][1], defpoints[2][1]], 
                               color='red', linewidth=1, alpha=0.6, linestyle='--')
                
                # 绘制在bbox线段上的点（红色圆圈）
                for point_info in dim_info['points_on_bbox_lines']:
                    point = point_info['point']
                    ax.plot(point[0], point[1], 'ro', markersize=8, 
                           label='Points on Bbox Line' if not points_on_bbox_line_labeled else "")
                    points_on_bbox_line_labeled = True
                
                # 绘制在参考线上的点（蓝色圆圈）
                for point_info in dim_info['points_on_reference_lines']:
                    point = point_info['point']
                    ax.plot(point[0], point[1], 'bo', markersize=8, 
                           label='Points on Reference Line' if not points_on_ref_line_labeled else "")
                    points_on_ref_line_labeled = True
                
                # 绘制dimension连线（红色虚线）
                points_on_bbox = [p['point'] for p in dim_info['points_on_bbox_lines']]
                points_on_ref = [p['point'] for p in dim_info['points_on_reference_lines']]
                
                if points_on_bbox and points_on_ref:
                    for bbox_point in points_on_bbox:
                        for ref_point in points_on_ref:
                            ax.plot([bbox_point[0], ref_point[0]], 
                                   [bbox_point[1], ref_point[1]], 
                                   'r--', linewidth=1, alpha=0.6)
                
                # 添加dimension文本标注
                if dim_info.get('textpos'):
                    text_pos = dim_info['textpos']
                    measurement = dim_info.get('measurement', 0)
                    text_content = dim_info.get('text') or f"D{dim_idx+1}: {measurement:.2f}"
                    ax.annotate(text_content, 
                              xy=(text_pos[0], text_pos[1]), 
                              fontsize=8, color='red',
                              bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            # 绘制参考线（绿色）
            for ref_idx, ref_line in enumerate(result['reference_lines']):
                ax.plot([ref_line['start'][0], ref_line['end'][0]], 
                       [ref_line['start'][1], ref_line['end'][1]], 
                       color='green', linewidth=2, alpha=0.7, 
                       label='Reference Line' if ref_idx == 0 else "")
            
            # 绘制提取的文本（紫色）
            text_to_arc_line_labeled = False
            compatible_text_labeled = False
            incompatible_text_labeled = False
            
            for text_idx, text_info in enumerate(result['extracted_texts']):
                text_center = self.get_text_center(text_info)
                if text_center:
                    # 根据兼容性设置颜色
                    is_compatible = text_info.get('is_compatible', False)
                    color = 'green' if is_compatible else 'red'
                    marker_color = 'go' if is_compatible else 'ro'
                    
                    # 根据兼容性添加标签
                    if is_compatible and not compatible_text_labeled:
                        label = 'Compatible Text'
                        compatible_text_labeled = True
                    elif not is_compatible and not incompatible_text_labeled:
                        label = 'Incompatible Text'
                        incompatible_text_labeled = True
                    else:
                        label = ""
                    
                    ax.plot(text_center[0], text_center[1], marker_color, markersize=8, label=label)
                    
                    # 显示文本内容和兼容性信息
                    content = text_info.get('content', '')
                    compatibility_status = "✓" if is_compatible else "✗"
                    label_text = f"T{text_idx+1}: {content} {compatibility_status}"
                    
                    ax.annotate(label_text, 
                              xy=(text_center[0], text_center[1]), 
                              fontsize=8, color=color,
                              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
                    
                    # 如果有关联的arc，绘制连线
                    if 'closest_arc_center' in text_info:
                        arc_center = text_info['closest_arc_center']
                        ax.plot([text_center[0], arc_center[0]], 
                               [text_center[1], arc_center[1]], 
                               color='purple', linewidth=1, linestyle='--', alpha=0.5,
                               label='Text-Arc Connection' if not text_to_arc_line_labeled else "")
                        text_to_arc_line_labeled = True
            
            # 设置图像属性
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            
            # 添加标题信息
            title = (f"Extracted Res {idx+1} - "
                    f"Dimensions: {len(result['extracted_dimensions'])}, "
                    f"Texts: {len(result['extracted_texts'])}, "
                    f"Ref Lines: {len(result['reference_lines'])}, "
                    f"Stiffeners: {len(result.get('stiffeners', []))}")
            plt.title(title, fontsize=12)
            
            # 保存图像
            filename = f"{output_dir}/dimension_extract_{idx+1}_{bbox[0]:.0f}-{bbox[1]:.0f}-{bbox[2]:.0f}-{bbox[3]:.0f}_{timestamp}.png"
            plt.savefig(filename, bbox_inches='tight', dpi=150)
            plt.close()
            
            print(f"  已保存可视化结果到 {filename}")
        
        print(f"完成可视化 {len(results)} 个结果")

    def get_text_center(self, text_info: Dict) -> List[float]:
        """获取文本的中心点坐标"""
        if 'position' in text_info:
            position = text_info['position']
            if isinstance(position, list) and len(position) >= 2:
                return position[:2]
        
        # 如果没有position字段，尝试从其他字段计算中心点
        if 'insertion_point' in text_info:
            insertion_point = text_info['insertion_point']
            if isinstance(insertion_point, list) and len(insertion_point) >= 2:
                return insertion_point[:2]
        
        # 如果有bound字段，计算边界框的中心点
        if 'bound' in text_info:
            bound = text_info['bound']
            if isinstance(bound, dict) and all(key in bound for key in ['x1', 'y1', 'x2', 'y2']):
                center_x = (bound['x1'] + bound['x2']) / 2
                center_y = (bound['y1'] + bound['y2']) / 2
                return [center_x, center_y]
        
        # 如果都没有，返回None
        return None

    def is_point_in_bbox(self, point: List[float], bbox: List[float]) -> bool:
        """
        检查点是否在bbox内
        
        参数:
        point: 点坐标 [x, y]
        bbox: 边界框 [x1, y1, x2, y2]，其中(x1,y1)和(x2,y2)是对角线上的两个点
        
        返回:
        bool: 点是否在bbox内
        """
        if not point or len(point) < 2 or not bbox or len(bbox) < 4:
            return False
        
        x, y = point[:2]
        x1, y1, x2, y2 = bbox[:4]
        
        # 确保min <= max，不假设哪个是左上角或右下角
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)
        
        is_inside = min_x <= x <= max_x and min_y <= y <= max_y
        
        if self.debug and is_inside:
            print(f"        点 {point} 在bbox [{min_x:.2f}, {min_y:.2f}, {max_x:.2f}, {max_y:.2f}] 内")
        elif self.debug:
            print(f"        点 {point} 不在bbox [{min_x:.2f}, {min_y:.2f}, {max_x:.2f}, {max_y:.2f}] 内")
        
        return is_inside

    def calculate_line_slope(self, start_point: List[float], end_point: List[float]) -> float:
        """
        计算直线的斜率
        
        参数:
        start_point: 起点坐标 [x, y]
        end_point: 终点坐标 [x, y]
        
        返回:
        float: 斜率值，如果是垂直线返回float('inf')
        """
        if not start_point or not end_point or len(start_point) < 2 or len(end_point) < 2:
            return None
        
        x1, y1 = start_point[:2]
        x2, y2 = end_point[:2]
        
        dx = x2 - x1
        dy = y2 - y1
        
        # 处理垂直线的情况，使用更合理的阈值
        vertical_threshold = 1e-6  # 更严格的垂直线判断
        if abs(dx) < vertical_threshold:
            return float('inf')
        
        slope = dy / dx
        return slope

    def are_lines_parallel(self, line1_start: List[float], line1_end: List[float], 
                          line2_start: List[float], line2_end: List[float], 
                          tolerance: float = None) -> bool:
        """
        判断两条直线是否平行
        
        参数:
        line1_start, line1_end: 第一条直线的起点和终点
        line2_start, line2_end: 第二条直线的起点和终点
        tolerance: 斜率容差，默认使用self.slope_tolerance
        
        返回:
        bool: 是否平行
        """
        if tolerance is None:
            tolerance = self.slope_tolerance
        
        slope1 = self.calculate_line_slope(line1_start, line1_end)
        slope2 = self.calculate_line_slope(line2_start, line2_end)
        
        if slope1 is None or slope2 is None:
            return False
        
        # 两条线都是垂直线（斜率为无穷）
        if slope1 == float('inf') and slope2 == float('inf'):
            return True
        
        # 检查是否都是近似垂直的线（斜率绝对值很大）
        large_slope_threshold = 1.0 / tolerance  # 当容差为0.1时，阈值为10
        
        slope1_is_large = slope1 == float('inf') or abs(slope1) > large_slope_threshold
        slope2_is_large = slope2 == float('inf') or abs(slope2) > large_slope_threshold
        
        if slope1_is_large and slope2_is_large:
            # 两条线都是近似垂直的，认为平行
            if self.debug:
                print(f"          垂直线段平行: slope1={slope1}, slope2={slope2}")
            return True
        
        # 如果一条是垂直/近似垂直，另一条不是，则不平行
        if slope1_is_large != slope2_is_large:
            return False
        
        # 两条线都不是垂直线，比较斜率差值
        slope_diff = abs(slope1 - slope2)
        is_parallel = slope_diff <= tolerance
        
        if self.debug and is_parallel:
            print(f"          线段平行: slope1={slope1:.4f}, slope2={slope2:.4f}, diff={slope_diff:.4f}")
        
        return is_parallel

    def calculate_distance_between_parallel_lines(self, line1_start: List[float], line1_end: List[float],
                                                  line2_start: List[float], line2_end: List[float]) -> float:
        """
        计算两条平行线之间的距离（点到直线的距离）
        
        参数:
        line1_start, line1_end: 第一条直线的起点和终点
        line2_start, line2_end: 第二条直线的起点和终点
        
        返回:
        float: 两条平行线之间的距离
        """
        if not all([line1_start, line1_end, line2_start, line2_end]):
            return float('inf')
        
        x1, y1 = line1_start[:2]
        x2, y2 = line1_end[:2]
        x3, y3 = line2_start[:2]
        
        # 检查第一条线段是否为点（长度为0）
        dx = x2 - x1
        dy = y2 - y1
        line_length_squared = dx * dx + dy * dy
        
        if line_length_squared < 1e-12:  # 线段长度几乎为0
            # 如果第一条线退化为点，计算点到点的距离
            if self.debug:
                print(f"            警告: 第一条线段退化为点 ({x1}, {y1})")
            return math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
        
        # 计算第一条直线的方程 Ax + By + C = 0
        # 对于直线 (x1,y1) 到 (x2,y2)
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        
        # 检查分母是否为0
        denominator = math.sqrt(A * A + B * B)
        if denominator < 1e-12:
            if self.debug:
                print(f"            警告: 直线方程系数异常，A={A}, B={B}")
            return math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
        
        # 计算点 (x3,y3) 到直线的距离
        distance = abs(A * x3 + B * y3 + C) / denominator
        
        return distance

    def calculate_midpoint_distance(self, line1_start: List[float], line1_end: List[float],
                                   line2_start: List[float], line2_end: List[float]) -> float:
        """
        计算两条线段中点之间的距离
        
        参数:
        line1_start, line1_end: 第一条线段的起点和终点
        line2_start, line2_end: 第二条线段的起点和终点
        
        返回:
        float: 两条线段中点之间的距离
        """
        if not all([line1_start, line1_end, line2_start, line2_end]):
            return float('inf')
        
        # 计算第一条线段的中点
        mid1_x = (line1_start[0] + line1_end[0]) / 2
        mid1_y = (line1_start[1] + line1_end[1]) / 2
        
        # 计算第二条线段的中点
        mid2_x = (line2_start[0] + line2_end[0]) / 2
        mid2_y = (line2_start[1] + line2_end[1]) / 2
        
        # 计算中点之间的距离
        distance = math.sqrt((mid2_x - mid1_x) ** 2 + (mid2_y - mid1_y) ** 2)
        
        return distance

    def are_lines_parallel_and_close(self, line1_start: List[float], line1_end: List[float], 
                                    line2_start: List[float], line2_end: List[float], 
                                    slope_tolerance: float = None,
                                    parallel_distance_threshold: float = 30.0,
                                    midpoint_distance_threshold: float = 200.0) -> Tuple[bool, Dict]:
        """
        判断两条直线是否平行且距离合适
        
        参数:
        line1_start, line1_end: 第一条直线的起点和终点
        line2_start, line2_end: 第二条直线的起点和终点
        slope_tolerance: 斜率容差，默认使用self.slope_tolerance
        parallel_distance_threshold: 平行线之间的距离阈值
        midpoint_distance_threshold: 中点距离阈值
        
        返回:
        Tuple[bool, Dict]: (是否满足条件, 详细信息字典)
        """
        if slope_tolerance is None:
            slope_tolerance = self.slope_tolerance
        
        # 首先检查是否平行
        is_parallel = self.are_lines_parallel(line1_start, line1_end, line2_start, line2_end, slope_tolerance)
        
        info = {
            'is_parallel': is_parallel,
            'parallel_distance': None,
            'midpoint_distance': None,
            'meets_distance_criteria': False
        }
        
        if not is_parallel:
            return False, info
        
        # 计算平行线之间的距离
        parallel_distance = self.calculate_distance_between_parallel_lines(
            line1_start, line1_end, line2_start, line2_end)
        info['parallel_distance'] = parallel_distance
        
        # 计算中点距离
        midpoint_distance = self.calculate_midpoint_distance(
            line1_start, line1_end, line2_start, line2_end)
        info['midpoint_distance'] = midpoint_distance
        
        # 检查距离条件
        meets_distance_criteria = (parallel_distance <= parallel_distance_threshold and 
                                 midpoint_distance <= midpoint_distance_threshold)
        info['meets_distance_criteria'] = meets_distance_criteria
        
        if self.debug and meets_distance_criteria:
            print(f"          平行且距离合适: 平行距离={parallel_distance:.2f}≤{parallel_distance_threshold}, "
                  f"中点距离={midpoint_distance:.2f}≤{midpoint_distance_threshold}")
        elif self.debug and is_parallel:
            print(f"          平行但距离不合适: 平行距离={parallel_distance:.2f}>{parallel_distance_threshold} "
                  f"或中点距离={midpoint_distance:.2f}>{midpoint_distance_threshold}")
        
        return meets_distance_criteria, info

    def extract_stiffeners_in_bbox(self, close_component: Dict) -> List[Dict]:
        """
        提取bbox中的Stiffener线段
        检查Stiffener线段的start或end点是否在bbox内，
        并且检查Stiffener是否与bbox中任意一个line类型实体平行
        
        参数:
        close_component: close.json中的一个组件
        
        返回:
        List[Dict]: 在bbox内且与bbox中line平行的Stiffener线段列表
        """
        # 尝试不同的bbox字段名
        bbox = close_component.get('scaled_bbox')
        if bbox is None:
            bbox = close_component.get('bbox')
        
        if bbox is None:
            if self.debug:
                print("警告: 在组件中没有找到bbox或scaled_bbox字段")
            return []
        
        bbox_entities = close_component.get('entities', [])
        stiffeners_in_bbox = []
        
        # 提取bbox中的所有line类型线段
        bbox_lines = []
        for entity in bbox_entities:
            if entity['type'] == 'line':
                bbox_lines.append({
                    'start': entity['start'],
                    'end': entity['end'],
                    'type': 'line',
                    'source': 'direct_line',
                    'handle': entity.get('handle'),
                    'color': entity.get('color'),
                    'layerName': entity.get('layerName'),
                    'linetype': entity.get('linetype')
                })
            elif entity['type'] == 'lwpolyline' and 'segments' in entity:
                for segment in entity['segments']:
                    if segment['type'] == 'line':
                        bbox_lines.append({
                            'start': segment['start'],
                            'end': segment['end'],
                            'type': 'line',
                            'source': 'lwpolyline_segment',
                            'handle': entity.get('handle'),  # 从父lwpolyline实体获取句柄
                            'color': entity.get('color'),
                            'layerName': entity.get('layerName'),
                            'linetype': entity.get('linetype')
                        })
        
        if self.debug:
            print(f"    提取到bbox中 {len(bbox_lines)} 条line类型线段用于平行判断")
        
        # 从allbe数据中获取Stiffener
        stiffeners = None
        if isinstance(self.allbe_data, list) and len(self.allbe_data) > 0:
            entities_data = self.allbe_data[0]
            if isinstance(entities_data, dict) and 'Stiffener' in entities_data:
                stiffeners = entities_data['Stiffener']
        elif isinstance(self.allbe_data, dict) and 'Stiffener' in self.allbe_data:
            stiffeners = self.allbe_data['Stiffener']
        
        if not stiffeners:
            if self.debug:
                print(f"    警告: 在allbe数据中未找到Stiffener字段")
            return stiffeners_in_bbox
        
        if self.debug:
            print(f"    检查 {len(stiffeners)} 个Stiffener线段是否在bbox {bbox} 内且与bbox中line平行")
            print(f"    bbox范围: x=[{min(bbox[0], bbox[2]):.2f}, {max(bbox[0], bbox[2]):.2f}], y=[{min(bbox[1], bbox[3]):.2f}, {max(bbox[1], bbox[3]):.2f}]")
        
        for stiffener_idx, stiffener in enumerate(stiffeners):
            if stiffener.get('type') != 'line':
                continue
            
            start_point = stiffener.get('start')
            end_point = stiffener.get('end')
            
            if not start_point or not end_point:
                continue
            
            # 检查start或end点是否在bbox内
            start_in_bbox = self.is_point_in_bbox(start_point, bbox)
            end_in_bbox = self.is_point_in_bbox(end_point, bbox)
            
            # 只有至少一个端点在bbox内的stiffener才进行平行判断
            if start_in_bbox or end_in_bbox:
                # 检查是否与bbox中任意一个line平行，并找到距离最近的那一条
                closest_parallel_line = None
                min_parallel_distance = float('inf')
                
                for bbox_line_idx, bbox_line in enumerate(bbox_lines):
                    is_parallel_and_close, distance_info = self.are_lines_parallel_and_close(
                        start_point, end_point, 
                        bbox_line['start'], bbox_line['end'],
                        slope_tolerance=self.slope_tolerance,
                        parallel_distance_threshold=self.parallel_distance_threshold,
                        midpoint_distance_threshold=self.midpoint_distance_threshold
                    )
                    
                    if is_parallel_and_close:
                        # 检查这条平行线是否比之前找到的更近
                        current_parallel_distance = distance_info['parallel_distance']
                        if current_parallel_distance < min_parallel_distance:
                            min_parallel_distance = current_parallel_distance
                            # 将bbox_line的句柄信息加入到distance_info中
                            enhanced_distance_info = distance_info.copy()
                            enhanced_distance_info['bbox_line_handle'] = bbox_line.get('handle')
                            enhanced_distance_info['bbox_line_layer'] = bbox_line.get('layerName')
                            enhanced_distance_info['bbox_line_color'] = bbox_line.get('color')
                            enhanced_distance_info['bbox_line_linetype'] = bbox_line.get('linetype')
                            
                            closest_parallel_line = {
                                'index': bbox_line_idx,
                                'start': bbox_line['start'],
                                'end': bbox_line['end'],
                                'source': bbox_line['source'],
                                'handle': bbox_line.get('handle'),
                                'layerName': bbox_line.get('layerName'),
                                'color': bbox_line.get('color'),
                                'linetype': bbox_line.get('linetype'),
                                'distance_info': enhanced_distance_info
                            }
                            if self.debug:
                                print(f"        发现更近的平行线 {bbox_line_idx}: 距离 {current_parallel_distance:.2f}, 句柄: {bbox_line.get('handle')}")
                
                if closest_parallel_line is not None:
                    # 添加关于哪些端点在bbox内以及平行信息的详细信息
                    stiffener_info = stiffener.copy()
                    stiffener_info['start_in_bbox'] = start_in_bbox
                    stiffener_info['end_in_bbox'] = end_in_bbox
                    stiffener_info['is_parallel_to_bbox_line'] = True
                    stiffener_info['parallel_with_line'] = closest_parallel_line
                    stiffener_info['parallel_distance'] = closest_parallel_line['distance_info']['parallel_distance']
                    stiffener_info['midpoint_distance'] = closest_parallel_line['distance_info']['midpoint_distance']
                    stiffener_info['meets_distance_criteria'] = closest_parallel_line['distance_info']['meets_distance_criteria']
                    # 新增：添加匹配的bbox_line句柄信息到顶层便于访问
                    stiffener_info['matched_bbox_line_handle'] = closest_parallel_line['distance_info']['bbox_line_handle']
                    stiffener_info['matched_bbox_line_layer'] = closest_parallel_line['distance_info']['bbox_line_layer']
                    
                    stiffeners_in_bbox.append(stiffener_info)
                    
                    if self.debug:
                        stiffener_slope = self.calculate_line_slope(start_point, end_point)
                        bbox_line_slope = self.calculate_line_slope(closest_parallel_line['start'], closest_parallel_line['end'])
                        distance_info = closest_parallel_line['distance_info']
                        print(f"      ✓ Stiffener {stiffener_idx} (handle: {stiffener.get('handle')}): "
                              f"start={start_point} (in_bbox: {start_in_bbox}), "
                              f"end={end_point} (in_bbox: {end_in_bbox}), "
                              f"平行于最近的bbox line {closest_parallel_line['index']} "
                              f"(句柄: {closest_parallel_line.get('handle')}, "
                              f"斜率: stiffener={stiffener_slope:.4f}, bbox_line={bbox_line_slope:.4f}), "
                              f"平行距离={distance_info['parallel_distance']:.2f}, "
                              f"中点距离={distance_info['midpoint_distance']:.2f}")
                elif self.debug:
                    print(f"      ✗ Stiffener {stiffener_idx} (handle: {stiffener.get('handle')}): "
                          f"在bbox内但不与任何bbox line平行或距离不合适")
            elif self.debug:
                print(f"      ✗ Stiffener {stiffener_idx} (handle: {stiffener.get('handle')}): "
                      f"不在bbox内")
        
        if self.debug:
            print(f"    提取到 {len(stiffeners_in_bbox)} 个满足条件的Stiffener线段 (在bbox内、平行且距离合适)")
        
        return stiffeners_in_bbox

    def save_results(self, output_path: str):
        """处理并保存所有结果"""
        results = self.process_all()
        
        # 可视化结果
        if self.debug:
            print("开始可视化结果...")
        self.visualize_results(results)
        
        # 保存结果到JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        if self.debug:
            print(f"结果已保存到: {output_path}")
        
        return results

    def generate_excel_reports(self, results: List[Dict], output_dir: str = "./"):
        """
        生成三个Excel报告文件：
        1. 尺寸标注信息.xlsx
        2. 文本信息.xlsx  
        3. 面板.xlsx
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 生成尺寸标注信息.xlsx
        dimension_data = []
        non_standard_hole_id = 0  # 非标孔检出ID，从0开始递增
        
        for result in results:
            current_hole_id = non_standard_hole_id
            non_standard_hole_id += 1
            
            # 处理每个dimension
            for dim_info in result['extracted_dimensions']:
                # 获取基本信息
                dimension_handle = dim_info.get('handle', '')
                measurement_value = round(dim_info.get('measurement', 0))  # 取int
                
                # 获取标注边句柄（bbox线段上的点）
                bbox_line_handles = []
                for point_info in dim_info.get('points_on_bbox_lines', []):
                    bbox_line_handles.append(point_info.get('handle', ''))
                
                # 获取参考边句柄和点信息，去除重复句柄
                reference_line_data = {}  # 使用字典来去重，key为handle，value为点信息列表
                for point_info in dim_info.get('points_on_reference_lines', []):
                    handle = point_info.get('handle', '')
                    point = point_info.get('point', [])
                    if handle not in reference_line_data:
                        reference_line_data[handle] = []
                    reference_line_data[handle].append(point)
                
                # 为每个唯一的参考线句柄创建一行记录
                for ref_handle, points_list in reference_line_data.items():
                    # 检查是否为参考点：如果该句柄有多个点，则"是否为参考点"为否，反之为是
                    is_reference_point = "是" if len(points_list) == 1 else "否"
                    reference_point_coord = points_list[0] if is_reference_point == "是" else ""
                    
                    # 获取对应的标注边句柄（如果有多个，使用第一个；如果没有，为空）
                    bbox_handle = bbox_line_handles[0] if bbox_line_handles else ""
                    
                    dimension_data.append({
                        '非标孔检出ID': current_hole_id,
                        '标注句柄': dimension_handle,
                        '标注值': measurement_value,
                        '标注边句柄': bbox_handle,
                        '参考边句柄': ref_handle,
                        '是否为参考点': is_reference_point,
                        '参考点坐标': str(reference_point_coord) if reference_point_coord else ""
                    })
        
        # 保存尺寸标注信息
        if dimension_data:
            dimension_df = pd.DataFrame(dimension_data)
            dimension_file = os.path.join(output_dir, "尺寸标注信息.xlsx")
            dimension_df.to_excel(dimension_file, index=False)
            print(f"已生成尺寸标注信息到: {dimension_file}")
        
        # 2. 生成文本信息.xlsx
        text_data = []
        non_standard_hole_id = 0  # 重置ID计数器
        
        for result in results:
            current_hole_id = non_standard_hole_id
            non_standard_hole_id += 1
            
            # 处理每个文本
            for text_info in result['extracted_texts']:
                text_handle = text_info.get('handle', '')
                text_content = text_info.get('content', '')
                
                text_data.append({
                    '非标孔检出ID': current_hole_id,
                    '标注句柄': text_handle,
                    '标注文本内容': text_content
                })
        
        # 保存文本信息
        if text_data:
            text_df = pd.DataFrame(text_data)
            text_file = os.path.join(output_dir, "文本信息.xlsx")
            text_df.to_excel(text_file, index=False)
            print(f"已生成文本信息到: {text_file}")
        
        # 3. 生成孔边形材.xlsx
        panel_data = []
        non_standard_hole_id = 0  # 重置ID计数器
        
        # 使用全局集合来去重，避免重复的句柄组合（跨所有非标孔检出ID）
        global_unique_stiffener_combinations = set()
        
        for result in results:
            current_hole_id = non_standard_hole_id
            non_standard_hole_id += 1
            
            # 处理每个stiffener
            for stiffener_info in result.get('stiffeners', []):
                bbox_line_handle = stiffener_info.get('matched_bbox_line_handle', '')
                stiffener_handle = stiffener_info.get('handle', '')
                
                # 创建组合的唯一标识
                combination_key = (bbox_line_handle, stiffener_handle)
                
                # 只有当组合在全局范围内是唯一的时候才添加
                if combination_key not in global_unique_stiffener_combinations:
                    global_unique_stiffener_combinations.add(combination_key)
                    panel_data.append({
                        '非标孔检出ID': current_hole_id,
                        '非标孔边界句柄': bbox_line_handle,
                        '孔边型材句柄': stiffener_handle
                    })
        
        # 保存孔边形材信息
        if panel_data:
            panel_df = pd.DataFrame(panel_data)
            panel_file = os.path.join(output_dir, "孔边形材.xlsx")
            panel_df.to_excel(panel_file, index=False)
            print(f"已生成孔边形材信息到: {panel_file}")
        
        # 4. 生成面板.xlsx
        panel_text_data = []
        non_standard_hole_id = 0  # 重置ID计数器
        
        for result in results:
            current_hole_id = non_standard_hole_id
            non_standard_hole_id += 1
            
            # 处理每个文本，只保存FB开头的文本
            for text_info in result['extracted_texts']:
                text_content = text_info.get('content', '')
                # 只保存FB开头的文本
                if text_content.startswith('FB'):
                    panel_text_data.append({
                        '非标孔检出ID': current_hole_id,
                        '面板文本参数': text_content
                    })
        
        # 保存面板信息
        if panel_text_data:
            panel_text_df = pd.DataFrame(panel_text_data)
            panel_text_file = os.path.join(output_dir, "面板.xlsx")
            panel_text_df.to_excel(panel_text_file, index=False)
            print(f"已生成面板信息到: {panel_text_file}")
        
        # 输出统计信息
        print(f"\n========== Excel报告生成完成 ==========")
        print(f"处理了 {non_standard_hole_id} 个非标孔检出目标")
        print(f"生成了 {len(dimension_data)} 条尺寸标注记录")
        print(f"生成了 {len(text_data)} 条文本记录")
        print(f"生成了 {len(panel_data)} 条孔边形材记录")
        print(f"生成了 {len(panel_text_data)} 条面板记录")


if __name__ == "__main__":
    # 可选择是否运行测试
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dxfpath', type=str, default="/Users/ieellee/Documents/FDU/ship/holes_detection/shadow.dxf", help="dxf path")
    args = parser.parse_args()

    dxf_path = args.dxfpath
    output_path = './final.json'
    selected_layer = "Holes"
    
    converter = DXFConverterV2(selected_layer)
    bboxes, hatch_bboxes = converter.convert_file(dxf_path, output_path)
    print(f"bboxes = {bboxes}")
    print(f"hatch_bboxes = {hatch_bboxes}")
    with open("bbox_temp.txt", "w") as f:
        for bbox in bboxes:
            f.write(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{bbox[4]}\n")
    
    
    
    debug = False
    
    # 主要处理逻辑：从最终DXF中导出bbox，然后重新进行extract和DimensionExtractor
    print("开始从最终DXF中提取bbox并重新处理...")
    
    # 1. 从最终DXF中导出bbox（假设bbox在检测结果层）
    bbox_list_from_dxf = []
    for bbox in bboxes:
        if len(bbox) >= 5:  # 确保有置信度
            bbox_list_from_dxf.append([bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]])
        else:
            bbox_list_from_dxf.append([bbox[0], bbox[1], bbox[2], bbox[3], 1.0])  # 默认置信度1.0
    
    print(f"从DXF中提取到 {len(bbox_list_from_dxf)} 个bbox")
    
    # 2. 重新生成JSON文件（如果需要）
    json_output_path = dxf_path.replace('.dxf', '.json')
    print(f"对应的JSON文件路径: {json_output_path}")
    
    # 3. 提取所有实体信息
    print("开始提取所有实体...")
    from extract_allbe import EntityExtractor as AllbeExtractor
    from extract_allbe_detailed import EntityExtractor as AllbeDetailedExtractor
    from extract_close import CloseExtractor
    
    # 提取allbe信息
    allbe_extractor = AllbeExtractor(json_output_path)
    allbe_data = allbe_extractor.extract_all()
    allbe_json_path = f"allbe_final_{dxf_path.split('/')[-1].replace('.dxf', '')}.json"
    with open(allbe_json_path, "w", encoding='utf-8') as f:
        json.dump(allbe_data, f, ensure_ascii=False, indent=4)
    print(f"allbe数据已保存到: {allbe_json_path}")
    
    # 提取详细实体信息
    allbe_detailed_extractor = AllbeDetailedExtractor(json_output_path)
    allbe_detailed_data = allbe_detailed_extractor.extract_all()
    allbe_detailed_json_path = f"allbe_detailed_final_{dxf_path.split('/')[-1].replace('.dxf', '')}.json"
    with open(allbe_detailed_json_path, "w", encoding='utf-8') as f:
        json.dump(allbe_detailed_data, f, ensure_ascii=False, indent=4)
    print(f"allbe详细数据已保存到: {allbe_detailed_json_path}")
    
    # 4. 重新进行extract_bbox（使用CloseExtractor）
    print("开始提取闭合连通分量...")
    close_extractor = CloseExtractor(allbe_json_path, scale_factor=1.5, tolerance_factor=10, debug=False)
    close_json_path = f"close_final_{dxf_path.split('/')[-1].replace('.dxf', '')}.json"
    close_extractor.save_to_json(close_json_path, bbox_list_from_dxf, require_degree_2=False, visualize=True)
    print(f"闭合连通分量数据已保存到: {close_json_path}")
    
    # 5. 重新进行DimensionExtractor
    print("开始提取尺寸信息...")
    
    # 设置提取参数
    t1 = 1.0  # 判断点是否在线上的容差
    t2 = 1.0  # 找到参考线的容差
    slope_tolerance = 0.1  # 判断直线平行的斜率容差
    parallel_distance_threshold = 999980.0  # 平行线之间的距离阈值
    midpoint_distance_threshold = 9999400.0  # 线段中点之间的距离阈值
    
    # 创建DimensionExtractor实例
    extractor = DimensionExtractor(
        allbe_path=allbe_detailed_json_path,
        close_path=close_json_path,
        t1=t1,
        t2=t2,
        debug=False,  # 设置为False以避免过多输出
        slope_tolerance=slope_tolerance,
        parallel_distance_threshold=parallel_distance_threshold,
        midpoint_distance_threshold=midpoint_distance_threshold
    )
    
    # 执行提取并保存结果
    final_output_path = f"extracted_dimensions_final_{dxf_path.split('/')[-1].replace('.dxf', '')}.json"
    results = extractor.save_results(final_output_path)
    
    # 生成Excel报告
    print("生成Excel报告...")
    extractor.generate_excel_reports(results, output_dir="./")
    
    print(f"最终处理完成！")
    print(f"处理了 {len(results)} 个检测目标")
    print(f"结果已保存到: {final_output_path}")
    
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
    
    
    
    if debug:
        # 输入文件路径
        allbe_path = "allbe_detailed.json"  # 包含所有实体信息的文件
        close_path = "close.json"  # 包含检测到的目标的文件
        output_path = "extracted_dimensions.json"  # 输出文件
        
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
            debug=debug,
            slope_tolerance=slope_tolerance,
            parallel_distance_threshold=parallel_distance_threshold,
            midpoint_distance_threshold=midpoint_distance_threshold
        )
        
        # 执行提取并保存结果
        results = extractor.save_results(output_path)
        
        # 生成Excel报告
        extractor.generate_excel_reports(results)
        
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
