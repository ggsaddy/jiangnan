from PIL import Image, ImageDraw
import json
import math
import numpy as np
import os
from tqdm import tqdm 
from glob import glob 
import time

class DXFRenderer:
    def __init__(self, max_size=1024, min_size=None, padding_ratio=0.05, patch_size=1024, overlap=0.5, auto_size=False, factor=0.05, width=2):
        """
        初始化渲染器
        max_size: 输出图像的最大边长
        min_size: 输出图像的最小边长，如果为None则等于max_size
        padding_ratio: 边距比例
        patch_size: 滑动窗口大小
        overlap: 重叠比例
        width: 统一的线条粗细
        """
        self.max_size = max_size
        self.min_size = min_size if min_size is not None else max_size
        self.padding_ratio = padding_ratio
        self.padding = int(min(self.max_size, self.min_size) * padding_ratio)  # 使用较小的边作为padding参考
        self.patch_size = patch_size
        self.overlap = overlap
        self.width = width  # 统一的线条粗细参数
        
        # 初始化图像和绘图对象
        self.image = None
        self.draw = None
        
        # 坐标转换参数
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.world_width = 0
        self.world_height = 0
        self.canvas_width = 0  # 新增：画布实际宽度
        self.canvas_height = 0  # 新增：画布实际高度

        self.auto_size = auto_size # automatically calculate max_size and min_size

        self.factor = factor
    
    def get_dxf_color(self, color_index):
        """
        将DXF颜色索引转换为RGB颜色
        color_index: DXF颜色索引(0-255)
        返回: (R,G,B)元组
        """
        # 特殊颜色处理
        if color_index == 0 or color_index == 256:  # BYBLOCK
            return (255, 255, 255)
        elif color_index == 7:  # 白色
            return (255, 255, 255)
        elif color_index == 251:  # 灰色
            return (128, 128, 128)
        
        # 标准AutoCAD颜色索引转换
        # 这里可以添加更多的颜色映射
        color_map = {
            1: (255, 0, 0),     # 红色
            2: (255, 255, 0),   # 黄色
            3: (0, 255, 0),     # 绿色
            4: (0, 255, 255),   # 青色
            5: (0, 0, 255),     # 蓝色
            6: (255, 0, 255),   # 洋红
            8: (128, 128, 128), # 深灰色
            9: (192, 192, 192), # 浅灰色
        }
        
        return color_map.get(color_index, (255, 255, 255))  # 默认返回白色

    def rectify_angle(self, angle):
        """
        对角度进行精度修正
        """
        # 处理接近0的情况
        if abs(angle) < 1e-10:
            return 0.0
        # 处理接近360的情况
        if abs(angle - 360) < 1e-10:
            return 360.0
        # 处理接近90度的情况
        if abs(angle - 90) < 1e-10:
            return 90.0
        # 处理接近-90度的情况
        if abs(angle + 90) < 1e-10:
            return 270.0  # 转换为0-360范围内
        # 处理接近180度的情况
        if abs(angle - 180) < 1e-10:
            return 180.0
        # 处理接近270度的情况
        if abs(angle - 270) < 1e-10:
            return 270.0
        
        return angle

    def get_arc_points(self, center, radius, start_angle, end_angle, num_points=64):
        """
        计算圆弧上的点序列
        
        在CAD/JSON中的角度定义：
        - 0度：x轴正方向
        - 90度：y轴正方向
        - 180度：x轴负方向
        - 270度：y轴负方向
        - 方向：逆时针为正
        """
        # 1. 统一精度处理
        radius = round(radius, 6)  # 将半径四舍五入到合理精度
        # if debug:
        #     print(f"Before: start_angle = {start_angle}, end_angle = {end_angle}")        
        
        # 2. 确保角度是角度制
        if isinstance(start_angle, float):
            if abs(start_angle) <= 2 * math.pi:
                start_angle = math.degrees(start_angle)
        if isinstance(end_angle, float):
            if abs(end_angle) <= 2 * math.pi:
                end_angle = math.degrees(end_angle)
        
        # 3. 规范化角度到[0, 360)范围
        start_angle = self.rectify_angle(start_angle)
        end_angle = self.rectify_angle(end_angle)
        start_angle = start_angle % 360
        end_angle = end_angle % 360

        
        # 4. 处理终点角度小于起点角度的情况
        # 例如：从270度到0度应该理解为从270度到360度
        if end_angle < start_angle:
            end_angle += 360
        if abs(end_angle - start_angle - 360) < 1e-10:
            end_angle = start_angle + 360

        # if debug:
        #     print(f"After: start_angle = {start_angle}, end_angle = {end_angle}")        
        # 5. 生成圆弧点
        angle_step = (end_angle - start_angle) / (num_points - 1)
        points = []
        
        for i in range(num_points):
            angle = math.radians(start_angle + i * angle_step)
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((x, y))
        
        return points
        
    def find_boundaries(self, entities):
        """找出所有图形的边界范围，忽略文本对象"""
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        
        for entity in entities:
            if entity['type'] == "text":
                continue
            if entity['type'] == "dimension":
                continue
            if 'bound' in entity:
                bound = entity['bound']
                min_x = min(min_x, bound['x1'])
                min_y = min(min_y, bound['y1'])
                max_x = max(max_x, bound['x2'])
                max_y = max(max_y, bound['y2'])
                
        return min_x, min_y, max_x, max_y

    def setup_transform(self, min_x, min_y, max_x, max_y):
        """设置坐标转换参数，处理缩放和偏移，并决定最终画布大小"""
        self.world_width = max_x - min_x
        self.world_height = max_y - min_y
        
        # 计算世界坐标系中的宽高比
        aspect_ratio = self.world_width / self.world_height
        
        # 根据宽高比决定最终画布尺寸
        if aspect_ratio >= 1:  # 宽大于等于高
            self.canvas_width = self.max_size
            self.canvas_height = max(int(self.max_size / aspect_ratio), self.min_size)
        else:  # 高大于宽
            self.canvas_height = self.max_size
            self.canvas_width = max(int(self.max_size * aspect_ratio), self.min_size)
        
        # 计算可用区域（考虑padding）
        available_width = self.canvas_width - 2 * self.padding
        available_height = self.canvas_height - 2 * self.padding
        
        # 计算缩放比例
        width_scale = available_width / self.world_width
        height_scale = available_height / self.world_height
        self.scale = min(width_scale, height_scale)
        
        # 计算偏移量使图形居中
        scaled_width = self.world_width * self.scale
        scaled_height = self.world_height * self.scale
        
        self.offset_x = self.padding + (available_width - scaled_width) / 2 - min_x * self.scale
        self.offset_y = self.canvas_height - self.padding - (available_height - scaled_height) / 2 + min_y * self.scale

        # 更新元数据
        self.metadata = {
            'scale': self.scale,
            'offset_x': self.offset_x,
            'offset_y': self.offset_y,
            'world_width': self.world_width,
            'world_height': self.world_height,
            'canvas_width': self.canvas_width,
            'canvas_height': self.canvas_height,
            'padding': self.padding,
            'patch_size': self.patch_size,
            'overlap': self.overlap
        }


    def transform_point(self, x, y):
        """将世界坐标转换为图片坐标"""
        px = x * self.scale + self.offset_x
        py = self.offset_y - y * self.scale
        return px, py


    def transform_insert_point(self, x, y, insert: list, rotation: float, scales: list):
        rr = rotation / 180 * math.pi
        cosine = math.cos(rr)
        sine = math.sin(rr)
        # x,y=(p[0]*scales[0]+100)/200,(p[1]*scales[1]+100)/200
        ipx, ipy = ((cosine * x * scales[0] - sine * y * scales[1])) + insert[0], \
                   ((sine * x * scales[0] + cosine * y * scales[1])) + insert[1]
        return ipx, ipy
    

    def is_clockwise(self, points):
        """
        判断平面上的点集是顺时针还是逆时针排列
        
        参数:
        points: 形如 [[x1,y1], [x2,y2], ...] 的点集列表
        
        返回值:
        True: 顺时针排列
        False: 逆时针排列
        """
        # 确保至少有3个点
        if len(points) < 3:
            return True
            raise ValueError("需要至少3个点来判断旋转方向")
        
        # 计算所有相邻向量的叉积之和
        # 对于平面向量，叉积可以简化为：(x2-x1)(y3-y1) - (y2-y1)(x3-x1)
        total_cross_product = 0
        
        # 遍历所有点，将最后一个点与第一个点连接形成闭环
        n = len(points)
        for i in range(n):
            j = (i + 1) % n  # 下一个点的索引
            k = (i + 2) % n  # 下下个点的索引
            
            # 计算两个相邻向量的叉积
            cross_product = ((points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) - 
                            (points[j][1] - points[i][1]) * (points[k][0] - points[i][0]))
            
            total_cross_product += cross_product
        
        # 如果叉积和为正，说明是顺时针方向
        # 如果叉积和为负，说明是逆时针方向
        return total_cross_product > 0

    def transform_points(self, points):
        """转换一系列点的坐标"""
        return [self.transform_point(x, y) for x, y in points]
    
    def draw_line(self, start, end, color, width=None):
        """绘制直线段"""
        if width is None:
            width = self.width
        x1, y1 = self.transform_point(start[0], start[1])
        x2, y2 = self.transform_point(end[0], end[1])
        rgb_color = self.get_dxf_color(color)
        self.draw.line([(x1, y1), (x2, y2)], fill=rgb_color, width=width)
        
    def draw_arc(self, center, radius, start_angle, end_angle, color, width=None):
        """使用点序列绘制圆弧"""
        if width is None:
            width = self.width
        
        points = self.get_arc_points(center, radius, start_angle, end_angle)
        transformed_points = self.transform_points(points)
        rgb_color = self.get_dxf_color(color)

        # if debug:
        #     print("**" * 20)
        #     print(f"center = {center}, radius = {radius}, start_angle = {start_angle}, end_angle = {end_angle}")
        #     print(f"points = {transformed_points[:3]}")
        #     print("**" * 20)
        
        # 使用线段连接点序列来绘制圆弧
        for i in range(len(transformed_points) - 1):
            self.draw.line([transformed_points[i], transformed_points[i + 1]], 
                         fill=rgb_color, width=width)
    
    def draw_text(self, position, text, height, color):
        """绘制文本"""
        x, y = self.transform_point(position[0], position[1])
        font_size = int(height * self.scale)
        rgb_color = self.get_dxf_color(color)
        self.draw.text((x, y), text, fill=rgb_color)

    def draw_circle_entity(self, entity, width=None):
        """
        使用PIL的ellipse方法直接绘制圆形实体
        
        参数说明：
        entity: 圆形实体数据，包含中心点和半径信息
        width: 线条宽度，默认使用self.width
        """
        if width is None:
            width = self.width
            
        # 获取原始数据
        center = entity['center']
        radius = entity['radius']
        color = entity['color']
        
        # 转换中心点坐标
        center_x, center_y = self.transform_point(center[0], center[1])
        # 计算缩放后的半径
        scaled_radius = radius * self.scale
        
        # if debug:
        #     print(f"Transformed: center=({center_x}, {center_y}), radius={scaled_radius}")
        
        # 计算边界框坐标
        # x0 = center_x - scaled_radius
        # y0 = center_y - scaled_radius
        # x1 = center_x + scaled_radius
        # y1 = center_y + scaled_radius

        x0 = min(center_x - scaled_radius, center_x + scaled_radius)
        x1 = max(center_x - scaled_radius, center_x + scaled_radius)
        y0 = min(center_y - scaled_radius, center_y + scaled_radius)
        y1 = max(center_y - scaled_radius, center_y + scaled_radius)
        
        # 使用PIL的ellipse方法绘制圆形
        self.draw.ellipse([x0, y0, x1, y1], 
                        fill=None,  # 不填充
                        outline=self.get_dxf_color(color),  # 使用实体的颜色
                        width=width)  # 使用指定的线宽

    # def draw_ellipse_entity(self, entity, width=2):
    #     """
    #     绘制椭圆实体
        
    #     参数说明：
    #     entity: 包含椭圆信息的实体字典，包含以下字段：
    #         - center: 中心点坐标 [x, y]
    #         - major_axis: 长轴向量 [dx, dy]
    #         - ratio: 短轴与长轴的比例
    #         - color: 颜色索引
    #     width: 线条宽度
    #     """
    #     # 获取基本参数
    #     center = entity['center']
    #     major_axis = entity['major_axis']
    #     ratio = entity['ratio']
    #     color = entity['color']
        
    #     # 计算长轴长度
    #     major_length = math.sqrt(major_axis[0]**2 + major_axis[1]**2)
        
    #     # 计算长轴旋转角度（相对于x轴）
    #     rotation = math.degrees(math.atan2(major_axis[1], major_axis[0]))
        
    #     # 计算长轴和短轴的一半长度（实际显示尺寸）
    #     a = major_length
    #     b = major_length * ratio
        
    #     # 转换中心点坐标到PIL坐标系统
    #     center_x, center_y = self.transform_point(center[0], center[1])
        
    #     # 计算椭圆的边界框
    #     x0 = center_x - a * self.scale
    #     y0 = center_y - b * self.scale
    #     x1 = center_x + a * self.scale
    #     y1 = center_y + b * self.scale
        
    #     # 使用PIL的ellipse方法绘制椭圆
    #     self.draw.ellipse([x0, y0, x1, y1], 
    #                     outline=self.get_dxf_color(color),
    #                     width=width)
    def draw_ellipse_entity(self, entity, width=None):
        """
        绘制椭圆实体
        
        参数说明：
        entity: 包含椭圆信息的实体字典，包含以下字段：
            - center: 中心点坐标 [x, y]
            - major_axis: 长轴向量 [dx, dy]
            - ratio: 短轴与长轴的比例
            - color: 颜色索引
            - calc_start_theta: 起始角度
            - calc_end_theta: 结束角度
        width: 线条宽度，默认使用self.width
        """
        if width is None:
            width = self.width
            
        # 获取基本参数
        center = entity['center']
        major_axis = entity['major_axis']
        ratio = entity['ratio']
        color = entity['color']
        start_theta = entity['calc_start_theta']
        end_theta = entity['calc_end_theta']
        
        # 计算长轴长度
        major_length = math.sqrt(major_axis[0]**2 + major_axis[1]**2)
        
        # 计算长轴和短轴的一半长度（实际显示尺寸）
        a = major_length
        b = major_length * ratio
        
        # 转换中心点坐标到PIL坐标系统
        center_x, center_y = self.transform_point(center[0], center[1])
        start_theta = 2 * math.pi  - start_theta  # 将角度从左下角坐标系转换
        end_theta =  2 * math.pi  - end_theta      # 将角度从左下角坐标系转换

        # 0 -》 90  conver 0 -》 -90
        # 90 - 180 conver 270 - 180
        # 180 - 270 conv 180 - 90
        # 270 - 360 conv 90 0
    
        # 判断是否为完整椭圆
        if abs(start_theta - end_theta) < 0.1:
            # 使用PIL的ellipse方法绘制完整椭圆
            x0 = center_x - a * self.scale
            y0 = center_y - b * self.scale
            x1 = center_x + a * self.scale
            y1 = center_y + b * self.scale
            
            self.draw.ellipse([x0, y0, x1, y1], 
                              outline=self.get_dxf_color(color),
                              width=width)
        else:
            # 使用小线段绘制不完整椭圆
            num_points = 100  # 控制绘制的点数
            points = []
            for i in range(num_points + 1):
                theta = start_theta + (end_theta - start_theta) * (i / num_points)
                x = center_x + a * self.scale * math.cos(theta)
                y = center_y + b * self.scale * math.sin(theta)
                points.append((x, y))
            
            # 绘制线段连接点序列
            for i in range(len(points) - 1):
                self.draw.line([points[i], points[i + 1]], 
                               fill=self.get_dxf_color(color), 
                               width=width)

    def draw_spline_entity(self, entity, width=None):
        """
        绘制样条曲线
        
        参数说明：
        entity: 样条曲线实体数据，包含以下字段：
            - vertices: 控制点列表，每个点是[x, y]格式
            - color: 颜色索引
        width: 线条宽度，默认使用self.width
        
        实现说明：
        1. 首先对控制点进行坐标转换
        2. 在每两个控制点之间插入额外的点以实现平滑效果
        3. 使用直线段连接所有点以近似样条曲线
        """
        if width is None:
            width = self.width
            
        # 获取控制点和颜色信息
        vertices = entity['vertices']
        color = entity['color']
        
        # 控制点数量检查
        if len(vertices) < 2:
            return
            
        # 将控制点转换为PIL坐标系统
        transformed_points = self.transform_points(vertices)
        rgb_color = self.get_dxf_color(color)
        
        # 定义插值点数量：每两个控制点之间插入的点数
        # 数量越大，曲线越平滑，但性能消耗越大
        num_segments = 20
        
        # 生成插值点
        interpolated_points = []
        for i in range(len(transformed_points) - 1):
            x1, y1 = transformed_points[i]
            x2, y2 = transformed_points[i + 1]
            
            # 在两个控制点之间生成过渡点
            for t in range(num_segments + 1):
                t_param = t / num_segments
                # 使用线性插值计算中间点的位置
                x = x1 + (x2 - x1) * t_param
                y = y1 + (y2 - y1) * t_param
                interpolated_points.append((x, y))
        
        # 使用PIL的line方法绘制连续的线段
        for i in range(len(interpolated_points) - 1):
            self.draw.line(
                [interpolated_points[i], interpolated_points[i + 1]],
                fill=rgb_color,
                width=width
            )

    def render_lwpolyline(self, entity):
        """
        渲染多段线，正确处理直线段和圆弧段的连接
        """
        vertices = entity['vertices']
        types = entity['verticesType']
        is_closed = entity['isClosed']
        color = entity['color']
        last_point = None


        starts_with_arc = (types[0] == 'arc')
        
        for i in range(len(vertices)):
            current_type = types[i]
            if current_type == 'line':
                self.draw_line([vertices[i][0], vertices[i][1]], [vertices[i][2], vertices[i][3]], color, width=self.width)
                # if debug:
                # print(f"Line from {[vertices[i][0], vertices[i][1]]} to {[vertices[i][2], vertices[i][3]]}")
                # print(f"color = {color}")
            elif current_type == 'arc':
                # 如果是圆弧段
                current_vertex = vertices[i]
                center = current_vertex[:2]
                start_angle = current_vertex[2]
                end_angle = current_vertex[3]
                radius = current_vertex[4]
                # if debug:
                # print(f"Arc center = {center}, radius = {radius}, start_angle = {start_angle}, end_angle = {end_angle}")
                # arc_points = self.get_arc_points(center, radius, start_angle, end_angle)
                # print(f"color = {color}")
                

                self.draw_arc(center, radius, start_angle, end_angle, color, width=self.width)
      
    def draw_arc_entity(self, entity, width=None):
        """
        绘制圆弧实体
        """
        if width is None:
            width = self.width
            
        center = entity['center']
        radius = entity['radius']
        start_angle = entity['startAngle']
        end_angle = entity['endAngle']
        color = entity['color']
        start_angle = int(start_angle)
        end_angle = int(end_angle)
        # if debug:
        #     print(f"center = {center}, start_angle = {start_angle}, end_angle = {end_angle}")
        #     if start_angle < 6:
        #         start_angle = 5.9
        self.draw_arc(center, radius, start_angle, end_angle, color, width)

    def transform_block_entity(self, entity, block_data):
        """
        转换块实体中的坐标，考虑插入点、缩放和旋转
        
        参数:
        entity: 块中的实体数据
        block_data: 插入块的数据(包含插入点、缩放、旋转等信息)
        """
        # 提取插入块的参数
        insert_point = block_data['insert']
        scales = block_data['scales']
        rotation = block_data['rotation']
        
        # 创建一个新的实体副本
        transformed_entity = entity.copy()
        
        # 根据实体类型转换相应的坐标
        if entity['type'] == 'line':
            transformed_entity['start'] = self.transform_block_point(
                entity['start'], insert_point, scales, rotation)
            transformed_entity['end'] = self.transform_block_point(
                entity['end'], insert_point, scales, rotation)
            
        elif entity['type'] == 'arc':
            transformed_entity['center'] = self.transform_block_point(
                entity['center'], insert_point, scales, rotation)
            
            # 处理半径精度
            transformed_entity['radius'] = abs(round(entity['radius'] * scales[0], 6))
            
            # 处理角度变换
            start_angle = self.rectify_angle(entity['startAngle'] + rotation)
            end_angle = self.rectify_angle(entity['endAngle'] + rotation)
            # start_angle = self.rectify_angle(entity['startAngle'])
            # end_angle = self.rectify_angle(entity['endAngle'])
        
            # start_angle = entity['startAngle'] + rotation
            # end_angle = entity['endAngle'] + rotation
            
            # 规范化角度到[0, 360)范围
            transformed_entity['startAngle'] = start_angle % 360
            transformed_entity['endAngle'] = end_angle % 360
            # if (360 - transformed_entity['startAngle']) < 0.5:
            #     transformed_entity['startAngle']
        
        return transformed_entity

    def transform_block_point(self, point, insert_point, scales, rotation_angle):
        """
        转换块中的点坐标
        
        参数:
        point: 原始点坐标 [x, y]
        insert_point: 插入点 [x, y]
        scales: 缩放比例 [sx, sy]
        rotation_angle: 旋转角度(度)
        """
        # 1. 应用缩放
        scaled_x = point[0] * scales[0]
        scaled_y = point[1] * scales[1]
        
        # 2. 应用旋转
        angle = math.radians(rotation_angle)
        rotated_x = scaled_x * math.cos(angle) - scaled_y * math.sin(angle)
        rotated_y = scaled_x * math.sin(angle) + scaled_y * math.cos(angle)
        
        # 3. 应用平移
        final_x = rotated_x + insert_point[0]
        final_y = rotated_y + insert_point[1]
        
        return [final_x, final_y]

    
    def render_block(self, insert_entity, blocks):
        """
        渲染插入的块，处理坐标系和角度转换
        
        参数:
        insert_entity: 插入实体的数据
        blocks: 块定义字典
        """
        block_name = insert_entity['blockName']
        print(f"block_name = {block_name}")
        if block_name not in blocks:
            return
            
        insert = insert_entity['insert']
        scales = insert_entity['scales']
        rotation = float(insert_entity['rotation'])
        block_entities = blocks[block_name]
        
        for entity in block_entities:
            if entity['color'] not in [3, 7]:
                continue
            if entity.get("linetype", None):
                if entity['linetype'] != 'Continuous':
                    continue
                
            # 递归处理嵌套块
            if entity['type'] == "insert":
                self.render_block(entity, blocks)
                continue
                
            # 创建转换后的实体副本
            transformed_entity = entity.copy()
            
            # 根据实体类型转换坐标
            if entity['type'] == 'line':
                transformed_entity['start'] = self.transform_insert_point(entity['start'][0], entity['start'][1], insert, rotation, scales)
                transformed_entity['end'] = self.transform_insert_point(entity['end'][0], entity['end'][1], insert, rotation, scales)
                self.draw_line(transformed_entity['start'], transformed_entity['end'], entity['color'], width=self.width)
                
            elif entity['type'] == 'lwpolyline':
                # 转换所有顶点
                # print(entity['type'])
                for i in range(len(entity['vertices'])):
                    if entity['verticesType'][i] == 'line':
                        transformed_entity['vertices'][i][0:2] = self.transform_insert_point(entity['vertices'][i][0], entity['vertices'][i][1], insert, rotation, scales)
                        transformed_entity['vertices'][i][2:4] = self.transform_insert_point(entity['vertices'][i][2], entity['vertices'][i][3], insert, rotation, scales)
                        # print(f"Line vertice {transformed_entity}")
                    elif entity['verticesType'][i] == 'arc':
                        # 转换圆弧中心点
                        transformed_entity['vertices'][i][0:2] = self.transform_insert_point(entity['vertices'][i][0], entity['vertices'][i][1], insert, rotation, scales)
                        # 调整角度
                        transformed_entity['vertices'][i][2] = entity['vertices'][i][2] + rotation
                        transformed_entity['vertices'][i][3] = entity['vertices'][i][3] + rotation
                        # 调整半径
                        transformed_entity['vertices'][i][4] *= scales[0]  # 假设x和y的缩放比例相同
                        # print(f"Arc vertice {transformed_entity}")
                self.render_lwpolyline(transformed_entity)
                
            elif entity['type'] == 'arc':
                transformed_entity['center'] = self.transform_insert_point(entity['center'][0], entity['center'][1], insert, rotation, scales)
                transformed_entity['radius'] *= scales[0]  # 假设x和y的缩放比例相同
                transformed_entity['startAngle'] = entity['startAngle'] + rotation
                transformed_entity['endAngle'] = entity['endAngle'] + rotation
                self.draw_arc_entity(transformed_entity)
                
            elif entity['type'] == 'circle':
                transformed_entity['center'] = self.transform_insert_point(entity['center'][0], entity['center'][1], insert, rotation, scales)
                transformed_entity['radius'] *= scales[0]  # 假设x和y的缩放比例相同
                self.draw_circle_entity(transformed_entity)
                
            elif entity['type'] == 'ellipse':
                transformed_entity['center'] = self.transform_insert_point(entity['center'][0], entity['center'][1], insert, rotation, scales)
                # 转换长轴向量
                major_x = entity['major_axis'][0] * scales[0]
                major_y = entity['major_axis'][1] * scales[1]
                angle = math.radians(rotation)
                transformed_entity['major_axis'] = [
                    major_x * math.cos(angle) - major_y * math.sin(angle),
                    major_x * math.sin(angle) + major_y * math.cos(angle)
                ]
                transformed_entity['calc_start_theta'] = entity['calc_start_theta'] + rotation
                transformed_entity['calc_end_theta'] = entity['calc_end_theta'] + rotation
                self.draw_ellipse_entity(transformed_entity)
                
            elif entity['type'] in ['spline', 'polyline']:
                # 转换所有控制点
                transformed_entity['vertices'] = [
                    self.transform_insert_point(v[0], v[1], insert, rotation, scales)
                    for v in entity['vertices']
                ]
                self.draw_spline_entity(transformed_entity)

    def render_entities(self, entities, blocks, bbox=None):
        """渲染所有实体"""
        for entity in entities:
            
            if entity['color'] not in [3, 7] and entity['type'] != 'insert': # [3, 7]
                # print("========================")
                # print(entity)
                # print("========================")
                continue
            if entity.get("linetype", None):
                if entity['linetype'] != 'Continuous':
                    continue
            # print(entity['type'])
            # 根据实体类型调用相应的绘制方法
            if entity['type'] == 'line':
                self.draw_line(entity['start'], entity['end'], entity['color'], width=self.width)
            elif entity['type'] == 'lwpolyline':
                self.render_lwpolyline(entity)
            elif entity['type'] == 'arc':
                self.draw_arc_entity(entity)
            elif entity['type'] == 'text':
                continue
                # self.draw_text(entity['insert'], entity['content'], entity['height'], entity['color'])
            elif entity['type'] == 'insert':
                # print("insert detect")
                self.render_block(entity, blocks)
            elif entity['type'] == 'circle':
                self.draw_circle_entity(entity)
            elif entity['type'] == 'ellipse':
                self.draw_ellipse_entity(entity)
            elif entity['type'] == 'spline' or entity['type'] == 'polyline':
                self.draw_spline_entity(entity)
            elif entity['type'] == 'dimension':
                continue
            else:
                print(f"TODO: uncoded type = {entity['type']}")
    
    def render(self, json_path, output_path, bbox=None):
        """主渲染函数，使用新的画布尺寸"""
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        entities = json_data[0]
        blocks = json_data[1]
        
        # 找出边界
        if bbox is None:
            min_x, min_y, max_x, max_y = self.find_boundaries(entities)
        else:
            min_x, min_y, max_x, max_y = bbox
        
        if self.auto_size:
            wid_ = max(max_x - min_x, max_y - min_y)
            hei_ = min(max_x - min_x, max_y - min_y)
            # self.max_size = int((((1024 / 40000 * max(wid_, hei_)) // 1024) + 1) * 1024)
            # self.min_size = int((((1024 / 40000 * min(wid_, hei_)) // 1024) + 1) * 1024)
            self.max_size = max(int(max(wid_, hei_) * self.factor), self.patch_size)
            self.min_size = max(int(min(wid_, hei_) * self.factor), self.patch_size)
            print(f"Using auto reference size, max_size = {self.max_size}, min_size = {self.min_size}")
            time.sleep(3)

        
        # 设置变换参数
        self.setup_transform(min_x, min_y, max_x, max_y)
        
        # 使用计算出的画布尺寸创建图像

        self.image = Image.new('RGB', (int(self.canvas_width), int(self.canvas_height)), 'black')
        self.draw = ImageDraw.Draw(self.image)
        
        # 渲染所有实体
        self.render_entities(entities, blocks, bbox)
        
        path_ = os.path.join(os.path.dirname(output_path), "sliding")
        os.makedirs(path_, exist_ok=True)
        path_ = os.path.join(path_, f"{os.path.basename(json_path).split('.')[0]}_sliding_{int(min_x)}-{int(min_y)}-{int(max_x)}-{int(max_y)}_{self.max_size}-{self.min_size}")
        os.makedirs(path_, exist_ok=True)
        
        # 滑动窗口处理
        if self.patch_size >= max(self.canvas_width, self.canvas_height):
            print(f"no sliding window, saving image at {path_}")
            self.image.save(os.path.join(path_, "patch_0_0.png"))
        else:
            print("Processing with sliding window")
            stride = int(self.patch_size * (1 - self.overlap))
            patch_index = []
            
            # 计算边界
            right_boundary = self.canvas_width - self.patch_size
            bottom_boundary = self.canvas_height - self.patch_size
            
            # 生成滑动窗口位置
            for x in range(0, self.canvas_width - self.patch_size + 1, stride):
                for y in range(0, self.canvas_height - self.patch_size + 1, stride):
                    # 确保不超出边界
                    current_x = min(x, right_boundary)
                    current_y = min(y, bottom_boundary)
                    patch_index.append([current_x, current_y])
            
            # 处理边界patch，确保覆盖所有区域
            for x in range(0, self.canvas_width - self.patch_size + 1, stride):
                if bottom_boundary > 0:
                    patch_index.append([x, bottom_boundary])
            
            for y in range(0, self.canvas_height - self.patch_size + 1, stride):
                if right_boundary > 0:
                    patch_index.append([right_boundary, y])
            
            # 确保添加右下角patch
            if right_boundary > 0 and bottom_boundary > 0:
                patch_index.append([right_boundary, bottom_boundary])
            
            # 移除重复的patch位置
            patch_index = list(set(tuple(p) for p in patch_index))
            
            # 保存所有patch
            for x, y in tqdm(patch_index, desc="Processing patches:..."):
                self.tmp = self.image.crop([x, y, x + self.patch_size, y + self.patch_size])
                path_t = os.path.join(path_, f"{int(min_x)}-{int(min_y)}-{int(max_x)}-{int(max_y)}_patch_{x}_{y}.png")
                self.tmp.save(path_t)
            
            # 保存完整图像
            self.image.save(os.path.join(path_, "whole.png"))
        
        # 保存元数据
        with open(os.path.join(path_, "meta_data.json"), 'w') as f:
            json.dump(self.metadata, f, indent=4)

        return self.max_size, self.min_size

def convert_png2dxf_coord(png_x, png_y, patch_x, patch_y, meta_path):

    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    scale = metadata['scale']
    offset_x = metadata['offset_x']
    offset_y = metadata['offset_y']
    overlap = metadata['overlap']
    padding = metadata['padding']
    patch_size = metadata['patch_size']
    stride = int(patch_size * (1 - overlap))
    
    global_x = patch_x + png_x
    global_y = patch_y + png_y
    

    dxf_x = ((global_x - padding) - offset_x) / scale
    dxf_y = (offset_y - (global_y - padding)) / scale
    
    return dxf_x, dxf_y


if __name__ == "__main__":
    ''' 
    22 insert TODO: 这个先不管
    24 insert TODO: 这个先不管
    30 TODO: angle problem
    '''
    debug = False
    test_one_file = True
    using_bbox = False
    black_list = [10, 22, 24, 33, 46, 76]

    json_path = "holes_dimention.json"
    output_path = "holes_dimention.json"
    renderer = DXFRenderer(auto_size=True)
    renderer.render(json_path, output_path)

    # if (debug or test_one_file) and (not using_bbox):
    #     # json_path = "/Users/ieellee/Downloads/codebase/ship/target_json/target30.json"
    #     json_path = "/Users/ieellee/Documents/FDU/ship/holes_detection/data1114_Holes_gt.json"
    #     output_path = json_path.replace("target_json", "target_png").replace(".json", ".png")
    #     renderer = DXFRenderer(max_size=1024*6, padding_ratio=0.05, patch_size=1024, overlap=0.5)
    #     renderer.render(json_path, output_path)
    # elif using_bbox:
    #     print("Using BBOX")
    #     json_path = "/Users/ieellee/Documents/FDU/ship/holes_detection/data1114_Holes_gt.json"
    #     output_path = json_path.replace("target_json", "target_png").replace(".json", ".png")
    #     renderer = DXFRenderer(max_size=1024*6, min_size=1024, padding_ratio=0.05, patch_size=1024, overlap=0.5)

    #     # with open(json_path, 'r', encoding='utf-8') as f:
    #     #     bbox_data = json.load(f)
    #     # for bbox in bbox_data:
    #     #     bbox = [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']] # xmin ymin xmax ymax

    #     # [262090.2323954443, -288192.0, 267726.9001478128, -284296.1701940134] -> 1024
    #     renderer.render(json_path, output_path, bbox=[61604, 48772, 158131, 104492])
    # else:
    #     for json_path in glob("./target_json/target*.json"):
    #         name = int(json_path.split("/")[-1].split(".")[0].replace("target", ""))
    #         if name in black_list:
    #             print(f"pass id {name}")
    #             continue
    #         output_path = json_path.replace("target_json", "target_png").replace(".json", ".png")
    #         renderer = DXFRenderer(max_size=96, padding_ratio=0.05, patch_size=96, overlap=0.5)
    #         renderer.render(json_path, output_path)

    '''
        200000, 20000
    '''