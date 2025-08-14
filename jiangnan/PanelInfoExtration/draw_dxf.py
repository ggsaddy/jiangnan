import os 
import numpy as np 
import ezdxf
import json


def draw_rectangle_in_dxf(file_path, folder, bbox_list, classi_res,idxs, free_edge_handles,non_free_edge_handles,all_handles,not_all_handles,removed_handles,delete_bracket_ids):
    folder = os.path.normpath(os.path.abspath(folder))
    os.makedirs(folder, exist_ok=True)

    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()

    if "Braket" not in doc.layers:
        doc.layers.add("Braket", color=30)

    for idx, (bbox, classification) in enumerate(zip(bbox_list, classi_res)):
        # 不对需要删除的肘板进行输出
        if idxs[idx] in delete_bracket_ids:
            continue
        
        # TODO:删除对应id的json和txt输出

        if classification=='Unclassified':
            continue
        if ',' in classification and 'ustd' not in classification:
            continue
        x1 = bbox[0][0] - 20
        y1 = bbox[0][1] - 20
        x2 = bbox[1][0] + 20
        y2 = bbox[1][1] + 20

        # 使用多段线绘制矩形
        rectangle_points = [
            (x1, y1),  # Top-left
            (x2, y1),  # Top-right
            (x2, y2),  # Bottom-right
            (x1, y2),  # Bottom-left
        ]
        msp.add_lwpolyline(rectangle_points, close=True, dxfattribs={"layer": "Braket"})

        # 添加文本
        if classification != "Unclassified":
            text = msp.add_text(classification, dxfattribs={"layer": "Braket", "height": 50})
            text.dxf.insert = ((x1 + x2) / 2, y2)
        text2 = msp.add_text(f"poly_id {idxs[idx]}", dxfattribs={"layer": "Braket", "height": 50})
        text2.dxf.insert = ((x1 + x2) / 2, y1)


    all_free_layer="精确匹配_自由边"
    all_non_free_layer="精确匹配_非自由边"
    not_all_free_layer="模糊匹配_自由边"
    not_all_non_free_layer="模糊匹配_非自由边"
    if all_free_layer not in doc.layers:
        doc.layers.add(all_free_layer, color=7)
    if all_non_free_layer not in doc.layers:
        doc.layers.add(all_non_free_layer, color=7)
    if not_all_free_layer not in doc.layers:
        doc.layers.add(not_all_free_layer, color=7)
    if not_all_non_free_layer not in doc.layers:
        doc.layers.add(not_all_non_free_layer, color=7)
    for e in msp:
        if e.dxf.handle in free_edge_handles and e.dxf.handle in all_handles:
            e.dxf.layer = all_free_layer
        if e.dxf.handle in free_edge_handles and e.dxf.handle in not_all_handles:
            e.dxf.layer = not_all_free_layer
        if e.dxf.handle in non_free_edge_handles and e.dxf.handle in all_handles:
            e.dxf.layer = all_non_free_layer
        if e.dxf.handle in non_free_edge_handles and e.dxf.handle in not_all_handles:
            e.dxf.layer = not_all_non_free_layer
    ref_line_layer="引线"
    if ref_line_layer not in doc.layers:
        doc.layers.add(ref_line_layer,color=7)
    
    for e in msp:
        if e.dxf.handle in removed_handles:
            e.dxf.layer=ref_line_layer


    file_name = os.path.basename(file_path)[:-4]
    doc.saveas(os.path.join(folder, f"{file_name}_braket.dxf"))

    print(f"Braket detect result drawn in {file_name}_braket")
    print("Done....")


def write_bboxes_with_ids(filename, bboxes,ids,count):
    """
    将包围盒数量、ID和坐标写入文件
    
    参数:
        filename: 文件名
        bboxes: 包围盒列表，每个元素是一个字典，包含id和坐标(x_min, x_max, y_min, y_max)
    """
    with open(filename, 'w') as f:
        # 写入包围盒数量
        f.write(f"Bounding Box Count: {count}\n")
        # 写入每个包围盒的坐标
        for i, bbox in enumerate(bboxes):
            x_min, x_max, y_min, y_max = bbox
            f.write(f"ID: {ids[i]}, x_min: {x_min}, x_max: {x_max}, "
                   f"y_min: {y_min}, y_max: {y_max}\n")

def read_bboxes_with_ids(filename):
    """
    从文件中读取包围盒数量、ID和坐标
    
    参数:
        filename: 文件名
        
    返回:
        count: 包围盒数量
        bboxes: 包围盒列表，每个元素是一个包含ID和坐标的字典
    """
    count = 0
    bboxes = []
    ids=[]
    
    with open(filename, 'r') as f:
        # 读取第一行获取数量
        first_line = f.readline().strip()
        if first_line.startswith("Bounding Box Count:"):
            count = int(first_line.split(":")[1].strip())
        
        # 读取后续行获取每个包围盒的ID和坐标
        for line in f:
            if line.startswith("ID:"):
                parts = line.strip().split(", ")
                bbox = {}
                for part in parts:
                    key, value = part.split(": ")
                    bbox[key.lower()] = float(value) if key not in ['ID'] else value
                bboxes.append((bbox['x_min'],bbox['x_max'],bbox['y_min'],bbox['y_max']))
                ids.append(bbox['id'])
    
    return count, bboxes,ids