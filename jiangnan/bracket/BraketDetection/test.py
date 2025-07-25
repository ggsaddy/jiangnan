from element import *
import matplotlib.pyplot as plt
from utils import *
from infoextraction import *
import numpy as np
import os
from plot_geo import *
from draw_dxf import *
from config import *


def output_training_data(polys, training_data_output_folder, name):
    # 如果输出文件夹不存在，则创建
    os.makedirs(training_data_output_folder, exist_ok=True)
    
    for i, poly in enumerate(polys):
        # 计算多边形中心坐标
        num_points = 0
        center_x, center_y = 0.0, 0.0
        
        for seg in poly:
            (start_x, start_y), (end_x, end_y), length = seg.start_point, seg.end_point, seg.length()
            center_x += start_x + end_x
            center_y += start_y + end_y
            num_points += 2  # 每个seg有两个点
        
        # 计算中心坐标
        center_x /= num_points
        center_y /= num_points

        # 创建文件并输出每个线段信息
        output_file = os.path.join(training_data_output_folder, f"{name}_{i}.txt")
        with open(output_file, "w") as f:
            for seg in poly:
                (start_x, start_y), (end_x, end_y), length = seg.start_point, seg.end_point, seg.length()
                
                # 平移线段的起点和终点
                shifted_start_x = start_x - center_x
                shifted_start_y = start_y - center_y
                shifted_end_x = end_x - center_x
                shifted_end_y = end_y - center_y
                
                # 将平移后的线段信息写入文件，保留两位小数
                f.write(f"{shifted_start_x:.2f} {shifted_start_y:.2f} {shifted_end_x:.2f} {shifted_end_y:.2f} {length:.2f}\n")
            

def process_json_files(folder_path, output_foler, training_data_output_folder, training_img_output_folder):
    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"路径 {folder_path} 不存在或不是一个文件夹。")
        return
    
    # 遍历文件夹中的每个文件
    for filename in os.listdir(folder_path):
        # 检查文件是否是JSON文件
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_foler, name)
            # training_data_output_path = os.path.join(training_data_output_folder, name)
            print(f"正在处理文件: {file_path}")
            
            # 打开并读取JSON文件内容
            try:
                process_json_data(file_path, output_path, training_data_output_folder, training_img_output_folder, name)  # 对数据进行操作
            except json.JSONDecodeError as e:
                print(f"解析JSON文件 {file_path} 时出错: {e}")
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")

def process_json_data(json_path, output_path, training_data_output_folder, training_img_output_folder, name):
    segmentation_config=SegmentationConfig()
    segmentation_config.json_path = json_path
    segmentation_config.line_image_path = os.path.join(output_path, "line.png")
    segmentation_config.poly_image_dir = os.path.join(output_path, "poly_image")
    segmentation_config.poly_info_dir = os.path.join(output_path, "poly_info")
    segmentation_config.res_image_path = os.path.join(output_path, "res.png")
    segmentation_config.dxf_output_folder = os.path.join(output_path)

    try:
        # os.makedirs(segmentation_config.poly_image_dir, exist_ok=True)
        os.makedirs(segmentation_config.poly_info_dir, exist_ok=True)
    except Exception as e:
        print(f"创建文件夹时出错: {e}")

    if segmentation_config.verbose:
        print("读取json文件")
    #文件中线段元素的读取和根据颜色过滤
    elements,segments,ori_segments,stiffeners=readJson(json_path,segmentation_config)
    ori_block=build_initial_block(ori_segments,segmentation_config)
    #将线进行适当扩张
    
    texts ,dimensions=findAllTextsAndDimensions(elements)
    ori_dimensions=dimensions
    dimensions=processDimensions(dimensions)
    texts=processTexts(texts)
    if segmentation_config.verbose:
        print("json文件读取完毕")
    

    #找出所有包含角隅孔圆弧的基本环
    ppolys, new_segments, point_map,star_pos_map,cornor_holes,text_map=findClosedPolys_via_BFS(elements,texts,dimensions,segments,segmentation_config)
    # output_training_data(ppolys, training_data_output_folder, name)

    # output_training_img(ppolys, new_segments, training_img_output_folder, name)
    #结构化输出每个肘板信息
    polys_info = []
    classi_res = []
    idxs=[]
    print("正在输出结构化信息...")
    for i, poly in enumerate(ppolys):
        try:
            segments_nearby=ori_block.segments_near_poly(poly)
            res = outputPolyInfo(poly, segments_nearby, segmentation_config, point_map, i, star_pos_map, cornor_holes,texts,dimensions,text_map,stiffeners)
        except Exception as e:
            res=None
            print(e)
        if res is not None:
            polys_info.append(res[0])
            classi_res.append(res[1])
            idxs.append(i)

    print("结构化信息输出完毕，保存于:", segmentation_config.poly_info_dir)
    if segmentation_config.mode=="dev":
        outputRes(ori_segments, point_map, polys_info, segmentation_config.res_image_path,segmentation_config.draw_intersections,segmentation_config.draw_segments,segmentation_config.line_image_drawPolys)

    #将检测到的肘板标注在原本的dxf文件中
    bboxs = []
    for poly_refs in polys_info:
        max_x = float('-inf')
        min_x = float('inf')
        max_y = float('-inf')
        min_y = float('inf')
        for seg in poly_refs:
            # 提取起点和终点的横纵坐标
            x_coords = [seg.start_point[0], seg.end_point[0]]
            y_coords = [seg.start_point[1], seg.end_point[1]]

            # 更新最大最小值
            max_x = max(max_x, *x_coords)
            min_x = min(min_x, *x_coords)
            max_y = max(max_y, *y_coords)
            min_y = min(min_y, *y_coords)

        bbox = [[min_x, min_y], [max_x, max_y]]
        bboxs.append(bbox)
    
    dxf_path = os.path.splitext(segmentation_config.json_path)[0] + '.dxf'
    dxf_output_folder = segmentation_config.dxf_output_folder
    draw_rectangle_in_dxf(dxf_path, dxf_output_folder, bboxs, classi_res,idxs)


if __name__ == '__main__':
    folder_path = "./data/split"
    output_folder = "./output"
    training_data_output_folder = "./DGCNN/data_folder"
    training_img_output_folder = "./training_img"
    process_json_files(folder_path, output_folder, training_data_output_folder, training_img_output_folder)
