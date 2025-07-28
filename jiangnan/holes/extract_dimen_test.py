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
from holes.load_v2 import DXFConverterV2
from holes.extract_dimen import DimensionExtractor
import logging

def extract_dimen_test(dxf_path):
    # 可选择是否运行测试
    '''parser = argparse.ArgumentParser()
    
    parser.add_argument('--dxfpath', type=str, default="/Users/ieellee/Documents/FDU/ship/holes_detection/shadow.dxf", help="dxf path") #唯一输入参数
    args = parser.parse_args()

    dxf_path = args.dxfpath'''

    logging.info("extract_dimen_test() 被调用了！")

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
    from holes.extract_allbe import EntityExtractor as AllbeExtractor
    from holes.extract_allbe_detailed import EntityExtractor as AllbeDetailedExtractor
    from holes.extract_close import CloseExtractor
    
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