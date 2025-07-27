'''
这个脚本的目的是提取出结构化数据中的信息，然后保存为json文件。

输出的json文件格式应该为：

{
    "entities": [
        {
            'type': 'arc',
            'start': points[0],
            'end': points[-1],
            'center': center,
            'radius': radius,
            'startAngle': start_angle % 360,
            'endAngle': end_angle % 360,
            'color': entity['color'],
            'handle': entity['handle']
        },
        {
            'type': 'line',
            'start': entity['start'],
            'end': entity['end'],
            'color': entity['color'],
            'handle': entity.get('handle'),
            'layerName': entity.get('layerName'),
            'linetype': entity.get('linetype', 'Continuous')
        },
        {......},
        {......},
    ],
    "dimensions": [
    
    ],
    "texts": [
    
    ],
    "Stiffener": [
        // 图层名称为"Stiffener_Visible"或"Stiffener_Invisible"的实体
    ],
    "Discontinuous": [
        // linetype不是"Continuous"的实体
    ]
}

如果遇到lwpolyline, 则需要提取出其中所有的line和arc加入到相应分类中，如果遇到insert，也需要提取出所有的line和arc。
dimensions也需要全部提取，提取方法参考statistic_holes.py

分类规则：
1. 如果图层名称是"Stiffener_Visible"或"Stiffener_Invisible"，则加入到"Stiffener"中
2. 如果linetype不是"Continuous"，则加入到"Discontinuous"中  
3. 其他情况加入到"entities"中

保存的文件名称叫allbe.json
'''

import json
import math
import numpy as np
import argparse
from typing import List, Dict, Tuple

class EntityExtractor:
    def __init__(self, json_path: str, debug=False):
        """
        初始化提取器
        
        参数:
        json_path: DXF转换后的JSON文件路径
        debug: 是否启用调试模式
        """
        self.debug = debug
        self.json_data = self.load_json(json_path)
        self.entities = self.json_data[0]
        self.blocks = self.json_data[1]
        
        if self.debug:
            print(f"初始化提取器: 加载了 {len(self.entities)} 个实体和 {len(self.blocks)} 个块")

    def load_json(self, json_path: str) -> dict:
        """加载JSON文件"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

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

    def transform_block_point(self, point: List[float], 
                            insert_point: List[float],
                            scales: List[float],
                            rotation_angle: float) -> List[float]:
        """转换块中的点坐标"""
        scaled_x = point[0] * scales[0]
        scaled_y = point[1] * scales[1]
        
        angle = math.radians(rotation_angle)
        rotated_x = scaled_x * math.cos(angle) - scaled_y * math.sin(angle)
        rotated_y = scaled_x * math.sin(angle) + scaled_y * math.cos(angle)
        
        final_x = rotated_x + insert_point[0]
        final_y = rotated_y + insert_point[1]
        
        return [final_x, final_y]

    def transform_insert_point(self, x, y, insert: list, rotation: float, scales: list):
        """转换插入点的坐标"""
        rr = rotation / 180 * math.pi
        cosine = math.cos(rr)
        sine = math.sin(rr)
        ipx, ipy = ((cosine * x * scales[0] - sine * y * scales[1])) + insert[0], \
                   ((sine * x * scales[0] + cosine * y * scales[1])) + insert[1]
        return ipx, ipy

    def process_entity(self, entity: Dict) -> List[Dict]:
        """处理单个实体，将其转换为标准格式"""
        results = []
        entity_type = entity['type']
        
        if entity_type == 'line':
            results.append({
                'type': 'line',
                'start': entity['start'],
                'end': entity['end'],
                'color': entity.get('color'),
                'handle': entity.get('handle'),
                'layerName': entity.get('layerName'),
                'linetype': entity.get('linetype', 'Continuous')
            })
        
        elif entity_type == 'arc':
            # points = self.get_arc_points(entity['center'], entity['radius'], entity['startAngle'], entity['endAngle'])
            results.append({
                'type': 'arc',
                'start': entity['start_point'],
                'end': entity['end_point'],
                'mid': entity['mid_point'],
                'center': entity['center'],
                'radius': entity['radius'],
                'startAngle': entity['startAngle'] % 360,
                'endAngle': entity['endAngle'] % 360,
                'color': entity.get('color'),
                'handle': entity.get('handle'),
                'layerName': entity.get('layerName'),
                'linetype': entity.get('linetype', 'Continuous')
            })
            
        elif entity_type == 'circle':
            results.append({
                'type': 'circle',
                'center': entity['center'],
                'radius': entity['radius'],
                'color': entity.get('color'),
                'handle': entity.get('handle'),
                'layerName': entity.get('layerName'),
                'linetype': entity.get('linetype', 'Continuous')
            })
            
        elif entity_type == 'ellipse':
            # 获取基本参数
            center = entity['center']
            major_axis = entity['major_axis']
            ratio = entity['ratio']
            start_theta = entity.get('calc_start_theta', 0)
            end_theta = entity.get('calc_end_theta', 2 * math.pi)
            
            # 计算长轴长度
            major_length = math.sqrt(major_axis[0]**2 + major_axis[1]**2)
            
            # 计算长轴和短轴的长度
            a = major_length
            b = major_length * ratio
            
            # 计算椭圆上的起点和终点
            # start_x = center[0] + a * math.cos(start_theta)
            # start_y = center[1] + a * math.sin(start_theta) * ratio
            # end_x = center[0] + a * math.cos(end_theta)
            # end_y = center[1] + a * math.sin(end_theta) * ratio
            # 长轴方向单位向量
            ux = major_axis[0] / major_length
            uy = major_axis[1] / major_length

            # 起点坐标
            start_x = center[0] + a * math.cos(start_theta) * ux - b * math.sin(start_theta) * uy
            start_y = center[1] + a * math.cos(start_theta) * uy + b * math.sin(start_theta) * ux

            # 终点坐标
            end_x = center[0] + a * math.cos(end_theta) * ux - b * math.sin(end_theta) * uy
            end_y = center[1] + a * math.cos(end_theta) * uy + b * math.sin(end_theta) * ux
            
            results.append({
                'type': 'ellipse',
                'start': [start_x, start_y],
                'end': [end_x, end_y],
                'center': center,
                'major_axis': major_axis,
                'ratio': ratio,
                'startTheta': start_theta,
                'endTheta': end_theta,
                'color': entity.get('color'),
                'handle': entity.get('handle'),
                'layerName': entity.get('layerName'),
                'linetype': entity.get('linetype', 'Continuous')
            })
            # print("######### ELLIPSE ENTITY ################")
            # print({
            #     'type': 'ellipse',
            #     'start': [start_x, start_y],
            #     'end': [end_x, end_y],
            #     'center': center,
            #     'major_axis': major_axis,
            #     'ratio': ratio,
            #     'startTheta': start_theta,
            #     'endTheta': end_theta,
            #     'color': entity.get('color'),
            #     'handle': entity.get('handle')
            # })
            # print("#########################")

        elif entity_type == 'lwpolyline':
            if 'vertices' in entity and 'verticesType' in entity:
                vertices = entity['vertices']
                types = entity['verticesType']
                for i in range(len(vertices)):
                    if types[i] == 'line':
                        results.append({
                            'type': 'line',
                            'start': [vertices[i][0], vertices[i][1]],
                            'end': [vertices[i][2], vertices[i][3]],
                            'color': entity.get('color'),
                            'handle': entity.get('handle'),
                            'layerName': entity.get('layerName'),
                            'linetype': entity.get('linetype', 'Continuous')
                        })
                    elif types[i] == 'arc':
                        center = [vertices[i][0], vertices[i][1]]
                        start_angle = vertices[i][2]
                        end_angle = vertices[i][3]
                        radius = vertices[i][4]
                        start_pt = vertices[i][5]
                        end_pt = vertices[i][6]
                        mid_pt = vertices[i][7]
                        # points = self.get_arc_points(center, radius, start_angle, end_angle)
                        results.append({
                            'type': 'arc',
                            'start': start_pt,
                            'end': end_pt,
                            'mid': mid_pt,
                            'center': center,
                            'radius': radius,
                            'startAngle': start_angle % 360,
                            'endAngle': end_angle % 360,
                            'color': entity.get('color'),
                            'handle': entity.get('handle'),
                            'layerName': entity.get('layerName'),
                            'linetype': entity.get('linetype', 'Continuous')
                        })
        
        elif entity_type == 'polyline':
            vertices = entity['vertices']
            if len(vertices) == 1:
                return results
            else:
                # for ver in [vertices[0], vertices[-1]]:
                # 添加所有的line

                results.append({
                    'type': 'polyline',
                    'start': [vertices[0][0], vertices[0][1]],
                    'end': [vertices[-1][0], vertices[-1][1]],
                    'color': entity.get('color'),
                    'handle': entity.get('handle'),
                    'layerName': entity.get('layerName'),
                    'linetype': entity.get('linetype', 'Continuous')
                }) 
                # for i in range(len(vertices) - 1):
                #     results.append({
                #         'type': 'line',
                #         'start': [vertices[i][0], vertices[i][1]],
                #         'end': [vertices[i+1][0], vertices[i+1][1]],
                #         'color': entity.get('color'),
                #         'handle': entity.get('handle')
                #     }) 
        
        elif entity_type == 'spline':
            vertices = entity['vertices']
            if len(vertices) == 1:
                return results
            else:
                # for ver in [vertices[0], vertices[-1]]:
                results.append({
                    'type': 'spline',
                    'start': [vertices[0][0], vertices[0][1]],
                    'end': [vertices[-1][0], vertices[-1][1]],
                    'color': entity.get('color'),
                    'handle': entity.get('handle'),
                    'layerName': entity.get('layerName'),
                    'linetype': entity.get('linetype', 'Continuous')
                }) 
                # for i in range(len(vertices) - 1):
                #     results.append({
                #         'type': 'line',
                #         'start': [vertices[i][0], vertices[i][1]],
                #         'end': [vertices[i+1][0], vertices[i+1][1]],
                #         'color': entity.get('color'),
                #         'handle': entity.get('handle')
                #     }) 
        
        else:
            raise NotImplemented("ERROR")

        return results

    def process_dimension(self, entity: Dict) -> Dict:
        """处理维度信息"""
        if entity['type'] in ['dimension', 'text', 'mtext']:
            result = {
                'type': entity['type'],
                'color': entity.get('color'),
                'handle': entity.get('handle'),
                'layerName': entity.get('layerName')
            }
            
            # 复制特定类型的信息
            if entity['type'] == 'dimension':
                # 维度测量值和文本
                if 'measurement' in entity:
                    result['measurement'] = entity['measurement']
                if 'text' in entity:
                    result['text'] = entity['text']
                if 'textpos' in entity:
                    result['textpos'] = entity['textpos']
                if 'dimtype' in entity:
                    result['dimtype'] = entity['dimtype']
                
                # 定义点
                for i in range(1, 6):
                    key = f'defpoint{i}'
                    if key in entity:
                        result[key] = entity[key]
            
            elif entity['type'] == 'text':
                if 'content' in entity:
                    result['content'] = entity['content']
                if 'position' in entity:
                    result['position'] = entity['position']
                if 'bound' in entity:
                    result['bound'] = entity['bound']
            
            elif entity['type'] == 'mtext':
                if 'text' in entity:
                    result['text'] = entity['text']
                if 'position' in entity:
                    result['position'] = entity['position']
                if 'bound' in entity:
                    result['bound'] = entity['bound']
            
            return result
        
        return None

    def process_block(self, block_name: str, insert_point: List[float], 
                    scales: List[float], rotation: float) -> List[Dict]:
        """处理块中的实体"""
        results = []
        
        if block_name in self.blocks:
            block_entities = self.blocks[block_name]
            
            for entity in block_entities:
                if entity['type'] == 'insert':
                    # 处理嵌套块
                    nested_insert = entity['insert']
                    nested_scales = entity['scales']
                    nested_rotation = entity['rotation']
                    
                    # 转换嵌套块的插入点到父块的坐标系
                    transformed_insert_x, transformed_insert_y = self.transform_insert_point(
                        nested_insert[0], nested_insert[1], 
                        insert_point, rotation, scales
                    )
                    
                    # 计算复合变换
                    compound_scales = [scales[0] * nested_scales[0], scales[1] * nested_scales[1]]
                    compound_rotation = rotation + nested_rotation
                    
                    # 递归处理嵌套块
                    nested_results = self.process_block(
                        entity['blockName'],
                        [transformed_insert_x, transformed_insert_y],
                        compound_scales,
                        compound_rotation
                    )
                    results.extend(nested_results)
                    
                elif entity['type'] == 'line':
                    # 转换线段坐标
                    start = self.transform_block_point(
                        entity['start'], insert_point, scales, rotation)
                    end = self.transform_block_point(
                        entity['end'], insert_point, scales, rotation)
                    
                    results.append({
                        'type': 'line',
                        'start': start,
                        'end': end,
                        'color': entity.get('color'),
                        'handle': entity.get('handle'),
                        'layerName': entity.get('layerName'),
                        'linetype': entity.get('linetype', 'Continuous')
                    })
                    
                elif entity['type'] == 'arc':
                    # 转换圆弧参数
                    center = self.transform_block_point(entity['center'], insert_point, scales, rotation) # for debug usage
                    start_point = self.transform_block_point(entity['start_point'], insert_point, scales, rotation)
                    end_point = self.transform_block_point(entity['end_point'], insert_point, scales, rotation)
                    mid_point = self.transform_block_point(entity['mid_point'], insert_point, scales, rotation)
                    center = self.transform_block_point(entity['center'], insert_point, scales, rotation)
                    radius = entity['radius'] * scales[0]  # 假设x和y的缩放比例相同
                    start_angle = self.rectify_angle(entity['startAngle'] + rotation) % 360
                    end_angle = self.rectify_angle(entity['endAngle'] + rotation) % 360

                    if abs(center[0] - mid_point[0]) <= 5000:
                        pass
                    else:
                        raise NotImplementedError(f"{abs(center[0] - mid_point[0])}")
                    
                    # 计算转换后的点
                    # points = self.get_arc_points(center, radius, start_angle, end_angle)
                    results.append({
                        'type': 'arc',
                        'start': start_point,
                        'end': end_point,
                        'mid': mid_point,
                        'center': center,
                        'radius': radius,
                        'startAngle': start_angle,
                        'endAngle': end_angle,
                        'color': entity.get('color'),
                        'handle': entity.get('handle'),
                        'layerName': entity.get('layerName'),
                        'linetype': entity.get('linetype', 'Continuous')
                    })
                    
                elif entity['type'] == 'circle':
                    # 转换圆的参数
                    center = self.transform_block_point(
                        entity['center'], insert_point, scales, rotation)
                    radius = entity['radius'] * scales[0]  # 假设x和y的缩放比例相同
                    
                    results.append({
                        'type': 'circle',
                        'center': center,
                        'radius': radius,
                        'color': entity.get('color'),
                        'handle': entity.get('handle'),
                        'layerName': entity.get('layerName'),
                        'linetype': entity.get('linetype', 'Continuous')
                    })
                    
                elif entity['type'] == 'ellipse':
                    # 获取基本参数
                    center = self.transform_block_point(
                        entity['center'], insert_point, scales, rotation)
                    
                    # 转换长轴向量
                    major_x = entity['major_axis'][0] * scales[0]
                    major_y = entity['major_axis'][1] * scales[1]
                    angle = math.radians(rotation)
                    transformed_major_axis = [
                        major_x * math.cos(angle) - major_y * math.sin(angle),
                        major_x * math.sin(angle) + major_y * math.cos(angle)
                    ]
                    
                    ratio = entity['ratio']
                    start_theta = entity.get('calc_start_theta', 0) + rotation
                    end_theta = entity.get('calc_end_theta', 2 * math.pi) + rotation
                    
                    # 计算长轴长度
                    major_length = math.sqrt(transformed_major_axis[0]**2 + transformed_major_axis[1]**2)
                    
                    # 计算长轴和短轴的长度
                    a = major_length
                    b = major_length * ratio
                    
                    # 计算椭圆上的起点和终点
                    # start_x = center[0] + a * math.cos(start_theta)
                    # start_y = center[1] + a * math.sin(start_theta) * ratio
                    # end_x = center[0] + a * math.cos(end_theta)
                    # end_y = center[1] + a * math.sin(end_theta) * ratio
                    
                    # 长轴方向单位向量
                    ux = transformed_major_axis[0] / major_length
                    uy = transformed_major_axis[1] / major_length

                    # 起点坐标
                    start_x = center[0] + a * math.cos(start_theta) * ux - b * math.sin(start_theta) * uy
                    start_y = center[1] + a * math.cos(start_theta) * uy + b * math.sin(start_theta) * ux

                    # 终点坐标
                    end_x = center[0] + a * math.cos(end_theta) * ux - b * math.sin(end_theta) * uy
                    end_y = center[1] + a * math.cos(end_theta) * uy + b * math.sin(end_theta) * ux
            

                    results.append({
                        'type': 'ellipse',
                        'start': [start_x, start_y],
                        'end': [end_x, end_y],
                        'center': center,
                        'major_axis': transformed_major_axis,
                        'ratio': ratio,
                        'startTheta': start_theta,
                        'endTheta': end_theta,
                        'color': entity.get('color'),
                        'handle': entity.get('handle'),
                        'layerName': entity.get('layerName'),
                        'linetype': entity.get('linetype', 'Continuous')
                    })
                    # print("#########################")
                    # print({
                    #     'type': 'ellipse',
                    #     'start': [start_x, start_y],
                    #     'end': [end_x, end_y],
                    #     'center': center,
                    #     'major_axis': transformed_major_axis,
                    #     'ratio': ratio,
                    #     'startTheta': start_theta,
                    #     'endTheta': end_theta,
                    #     'color': entity.get('color'),
                    #     'handle': entity.get('handle')
                    # })
                    # print("#########################")
                    
                elif entity['type'] == 'polyline' or entity['type'] == 'spline':
                    # 处理多段线或样条曲线
                    vertices = entity['vertices']
                    if len(vertices) > 1:
                        # 转换起点和终点坐标
                        start = self.transform_block_point(
                            [vertices[0][0], vertices[0][1]], insert_point, scales, rotation)
                        end = self.transform_block_point(
                            [vertices[-1][0], vertices[-1][1]], insert_point, scales, rotation)
                        
                        results.append({
                            'type': entity['type'],
                            'start': start,
                            'end': end,
                            'color': entity.get('color'),
                            'handle': entity.get('handle'),
                            'layerName': entity.get('layerName'),
                            'linetype': entity.get('linetype', 'Continuous')
                        })
                    
                elif entity['type'] == 'lwpolyline':
                    # 处理多段线的每个顶点
                    if 'vertices' in entity and 'verticesType' in entity:
                        vertices = entity['vertices']
                        types = entity['verticesType']
                        
                        for i, vtype in enumerate(types):
                            if vtype == 'line':
                                # 转换线段坐标
                                start_x, start_y = self.transform_insert_point(
                                    vertices[i][0], vertices[i][1],
                                    insert_point, rotation, scales
                                )
                                end_x, end_y = self.transform_insert_point(
                                    vertices[i][2], vertices[i][3],
                                    insert_point, rotation, scales
                                )
                                
                                results.append({
                                    'type': 'line',
                                    'start': [start_x, start_y],
                                    'end': [end_x, end_y],
                                    'color': entity.get('color'),
                                    'handle': entity.get('handle'),
                                    'layerName': entity.get('layerName'),
                                    'linetype': entity.get('linetype', 'Continuous')
                                })
                                
                            elif vtype == 'arc':
                                # 转换圆弧参数
                                center_x, center_y = self.transform_insert_point(
                                    vertices[i][0], vertices[i][1],
                                    insert_point, rotation, scales
                                )
                                start_angle = self.rectify_angle(vertices[i][2] + rotation) % 360
                                end_angle = self.rectify_angle(vertices[i][3] + rotation) % 360
                                radius = vertices[i][4] * scales[0]
                                start_x, start_y = self.transform_insert_point(
                                    vertices[i][5][0], vertices[i][5][1],
                                    insert_point, rotation, scales
                                )
                                end_x, end_y = self.transform_insert_point(
                                    vertices[i][6][0], vertices[i][6][1],
                                    insert_point, rotation, scales
                                )
                                mid_x, mid_y = self.transform_insert_point(
                                    vertices[i][7][0], vertices[i][7][1],
                                    insert_point, rotation, scales
                                )
                                # 计算转换后的点
                                # points = self.get_arc_points([center_x, center_y], radius, 
                                #                           start_angle, end_angle)
                                
                                results.append({
                                    'type': 'arc',
                                    'start': [start_x, start_y],
                                    'end': [end_x, end_y],
                                    'mid': [mid_x, mid_y],
                                    'center': [center_x, center_y],
                                    'radius': radius,
                                    'startAngle': start_angle,
                                    'endAngle': end_angle,
                                    'color': entity.get('color'),
                                    'handle': entity.get('handle'),
                                    'layerName': entity.get('layerName'),
                                    'linetype': entity.get('linetype', 'Continuous')
                                })
                
                elif entity['type'] == 'text':
                    # 转换文本位置
                    if 'position' in entity:
                        position = self.transform_block_point(
                            entity['position'], insert_point, scales, rotation)
                        results.append({
                            'type': 'text',
                            'content': entity.get('content', ''),
                            'position': position,
                            'color': entity.get('color'),
                            'handle': entity.get('handle'),
                            'layerName': entity.get('layerName'),
                            'linetype': entity.get('linetype', 'Continuous')
                        })
                
                elif entity['type'] == 'hatch':
                    # hatch实体通常用于填充，我们可以提取其边界信息
                    # 这里只是标记该实体存在，具体处理可以根据需要扩展
                    results.append({
                        'type': 'hatch',
                        'color': entity.get('color'),
                        'handle': entity.get('handle'),
                        'layerName': entity.get('layerName'),
                        'linetype': entity.get('linetype', 'Continuous')
                    })
                
                else:
                    # 跳过未知的实体类型
                    pass
        
        return results

    def classify_entity_by_layer_and_linetype(self, entities: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        根据图层名称和线型对实体进行分类
        
        返回:
        - regular_entities: 普通实体列表
        - stiffener_entities: Stiffener图层实体列表  
        - discontinuous_entities: 非连续线型实体列表
        """
        regular_entities = []
        stiffener_entities = []
        discontinuous_entities = []
        
        for entity in entities:
            layer_name = entity.get('layerName', '')
            line_type = entity.get('linetype', 'Continuous')
            
            # 检查是否为Stiffener图层
            is_stiffener = layer_name in ['Stiffener_Visible', 'Stiffener_Invisible']
            
            # 检查是否为非连续线型
            is_discontinuous = line_type != 'Continuous'
            
            if is_stiffener:
                stiffener_entities.append(entity)
            elif is_discontinuous:
                discontinuous_entities.append(entity)
            else:
                regular_entities.append(entity)
                
        return regular_entities, stiffener_entities, discontinuous_entities

    def extract_all(self) -> Dict:
        """提取所有实体、维度和文本信息"""
        result = {
            "entities": [],
            "dimensions": [],
            "texts": [],
            "Stiffener": [],
            "Discontinuous": []
        }
        not_processed_type = set()
        all_entities = []  # 临时存储所有实体
        
        # 处理主实体
        for entity in self.entities:
            # 处理基本图形实体
            if entity['type'] in ['line', 'arc', 'circle', 'lwpolyline', 'polyline', 'spline', 'ellipse']: # 'ellipse'
                entities = self.process_entity(entity)
                all_entities.extend(entities)
            
                
            # 处理插入块
            elif entity['type'] == 'insert':
                block_entities = self.process_block(
                    entity['blockName'],
                    entity['insert'],
                    entity['scales'],
                    entity['rotation']
                )
                all_entities.extend(block_entities)
                
            # 处理维度和文本
            elif entity['type'] in ['dimension', 'text', 'mtext']:
                processed = self.process_dimension(entity)
                if processed:
                    if entity['type'] == 'dimension':
                        result["dimensions"].append(processed)
                    else:  # text 或 mtext
                        result["texts"].append(processed)
            else:
                not_processed_type.add(entity['type'])
        
        # 对所有实体进行分类
        regular_entities, stiffener_entities, discontinuous_entities = self.classify_entity_by_layer_and_linetype(all_entities)
        
        result["entities"] = regular_entities
        result["Stiffener"] = stiffener_entities
        result["Discontinuous"] = discontinuous_entities
        
        print("Not processed type = ", not_processed_type)
        
        if self.debug:
            print(f"提取完成，共有实体 {len(result['entities'])} 个，Stiffener实体 {len(result['Stiffener'])} 个，Discontinuous实体 {len(result['Discontinuous'])} 个，尺寸标注 {len(result['dimensions'])} 个，文本 {len(result['texts'])} 个")
        
        return result

    def save_to_json(self, output_path: str):
        """提取所有信息并保存到JSON文件"""
        data = self.extract_all()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        if self.debug:
            print(f"结果已保存到: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="提取结构化数据并保存为JSON")
    parser.add_argument("--input", "-i", default="postpro_debug1.json", help="输入JSON文件路径")
    parser.add_argument("--output", "-o", default="allbe.json", help="输出JSON文件路径")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    
    args = parser.parse_args()
    
    extractor = EntityExtractor(args.input, debug=args.debug)
    extractor.save_to_json(args.output)
    print(f"提取完成，结果已保存到: {args.output}")

if __name__ == "__main__":
    main()