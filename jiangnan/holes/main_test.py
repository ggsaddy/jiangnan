import os
import cv2
import json 
import time 
import shutil
import argparse
import numpy as np
from glob import glob 
from tqdm import tqdm 
from ultralytics import YOLO
from holes.load import dxf2json
from holes.convert2png_v2 import DXFRenderer
from holes.yolo_test import predict_image, visualize_predictions, convert_png2dxf_coord ,nms, predict_image_tta
from holes.draw_dxf import draw_rectangle_in_dxf, yoloxyxy2dxfxyxy
from holes.load_v2 import DXFConverterV2
from holes.evaluate import evaluate, convert, analyze_confidence_thresholds, calculate_iou, calculate_overlap_rate
# from statistic_holes import EntityAnalyzer
from holes.filter_bbox import load_data_and_get_main_bbox
from holes.extract_dimen import DimensionExtractor
# 新增导入模块
from holes.extract_allbe import EntityExtractor as AllbeExtractor
from holes.extract_allbe_detailed import EntityExtractor as AllbeDetailedExtractor
from holes.extract_close import CloseExtractor

from types import SimpleNamespace
import logging

logging.basicConfig(level=logging.INFO)

def safe_filename(filename):
    """
    确保文件名在不同操作系统下都可用，处理特殊字符
    """
    import re
    # 替换不安全字符
    unsafe_chars = r'[<>:"/\\|?*\x00-\x1f]'
    safe_name = re.sub(unsafe_chars, '_', filename)
    # 处理Windows保留名称
    reserved_names = ['CON', 'PRN', 'AUX', 'NUL'] + [f'COM{i}' for i in range(1, 10)] + [f'LPT{i}' for i in range(1, 10)]
    name_without_ext = os.path.splitext(safe_name)[0]
    if name_without_ext.upper() in reserved_names:
        safe_name = f"_{safe_name}"
    return safe_name

def clear_space(base_path):
    """清理工作空间，基于指定的base_path"""
    sliding_path = os.path.join(base_path, 'sliding')
    # runs_path = os.path.join(base_path, 'runs')
    runs_path = os.path.join('./', 'runs')

    shutil.rmtree(sliding_path, ignore_errors=True)
    shutil.rmtree(runs_path, ignore_errors=True)

def save_debug_file(data, filepath, debug=True):
    """
    保存调试文件的通用方法
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if filepath.endswith('.json'):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    else:
        # 处理txt文件等
        with open(filepath, 'w', encoding='utf-8') as f:
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, (list, tuple)) and len(item) >= 5:
                        x1, y1, x2, y2, conf = item[:5]
                        f.write(f"{x1},{y1},{x2},{y2},{conf}\n")
                    else:
                        f.write(f"{item}\n")
            else:
                f.write(str(data))
                
    print(f"调试文件已保存到: {filepath}")
    
# pyinstaller -F main.py --add-data "best.pt;."
# pyinstaller -F --clean main.py --add-data "best.pt:." --log-level=DEBUG --exclude-module matplotlib.backends --exclude-module jinja2

'''
python main.py --dxfname holes_dimention1.dxf --json_path holes_dimention1.json --auto_size
python main.py --dxfname data1114_v2.dxf --json_path data1114_v2.json --max_size 4096 --min_size 4096 --segment_bbox --clear
python main.py --dxfname holes_dimention_panel1.dxf --json_path holes_dimention_panel1.json --auto_size
python main.py --dxfname holes_dimention.dxf --json_path holes_dimention.json --auto_size
python main.py --dxfname holes_dimention_v2.dxf --json_path holes_dimention_v2.json --auto_size
python main.py --dxfname postpro.dxf --json_path postpro.json --auto_size

TODO: 推理后处理，不添加边上的框.

'''
def get_default_args():

    return {
        'debug': False,
        'clear': True,
        'dxfpath': './',
        'dxfname': 'data1114_v2.dxf',
        'json_path': 'data1114_v2.json',
        'auto_size': True,
        'factor': 0.16,
        'max_size': 1024,
        'min_size': 1024,
        'padding_ratio': 0.05,
        'patch_size': 2560,
        'overlap': 0.5,
        'segment_bbox': True,
        'model_path': r'C:\Users\aa666aa666\Desktop\jiangnan-1\jiangnan\holes\test_only\best.pt',
        'dxf_output_path': './out',
        'evaluate_only': False,
        'dxf_path_gt': './data1114_Holes_gt.dxf',
        'output_path_gt': './data1114_Holes_gt.json',
        'selected_layer_gt': 'Holes',
        'abandon_layer_gt': '开孔识别结果',
        'dxf_path_pred': './data1114_Holes_pred.dxf',
        'output_path_pred': './data1114_Holes_pred.json',
        'selected_layer_pred': 'Holes',
        'inference_only': False,
        'conf': 0.5
    }

def merge_args(user_args: dict) -> SimpleNamespace:

    default_args = get_default_args()
    default_args.update(user_args)
    return SimpleNamespace(**default_args)

def main_test(args: SimpleNamespace):

    logging.info("main_test() 被调用了！")
    # 将dxfpath转换为绝对路径作为工作目录
    work_dir = os.path.abspath(args.dxfpath)
    args.dxfpath = work_dir
    
    # 确保工作目录存在
    os.makedirs(work_dir, exist_ok=True)
    
    # 更新所有相对路径为基于work_dir的绝对路径
    dxf_file_path = os.path.join(work_dir, args.dxfname)
    json_file_path = os.path.join(work_dir, args.json_path)
    
    # 处理文件名中的特殊字符
    safe_dxf_name = safe_filename(args.dxfname)
    base_name = safe_filename(os.path.splitext(args.dxfname)[0])
    
    if args.evaluate_only:
        converter_gt = DXFConverterV2(args.selected_layer_gt)
        bboxes_gt, bboxes_hatch = converter_gt.convert_file(args.dxf_path_gt, args.output_path_gt)

        converter_pred = DXFConverterV2(args.selected_layer_pred)
        bboxes_pred, _ = converter_pred.convert_file(args.dxf_path_pred, args.output_path_pred)
        if len(args.abandon_layer_gt) > 0:
            converter_ab = DXFConverterV2(args.abandon_layer_gt)
            bboxes_ab, _ = converter_ab.convert_file(args.dxf_path_gt, args.output_path_gt)
        else:
            bboxes_ab = None
        # bboxes1 = [[265061.9118044584,-287044.62146601186,265766.6937838789,-287778.4531252597,0.9316078424453735]]
        # bboxes2 = [[265061.9118044584,-287044.62146601186,265766.6937838789,-287778.4531252597,0.9316078424453735]]
        results = evaluate(nms(convert(bboxes_pred), iou_threshold=0.9), convert(bboxes_gt), convert(bboxes_ab), convert(bboxes_hatch))
        analyze_confidence_thresholds(os.path.join(work_dir, "eval_results_all.json"))
        # print(f"Evaluation results: {results}")

        # 读取最佳评估结果
        eval_results_path = os.path.join(work_dir, "eval_results.txt")
        with open(eval_results_path, "r") as f:
            best_results = json.load(f)
        
        best_conf = best_results["best_confidence_threshold"]
        best_iou = best_results["best_iou_threshold"]
        
        # 根据最佳置信度阈值过滤预测结果
        filtered_bboxes = []
        for bbox in bboxes_pred:
            if len(bbox) >= 5 and bbox[4] >= best_conf:  # 过滤置信度
                # 检查是否与舍弃框重叠
                should_keep = True
                if bboxes_ab is not None:
                    for abandon_box in bboxes_ab:
                        if calculate_iou(bbox, abandon_box) > 0.1:
                            should_keep = False
                            break
                if should_keep:
                    filtered_bboxes.append(bbox)
        
        # 添加对bboxes_hatch的过滤功能
        if bboxes_hatch is not None and len(bboxes_hatch) > 0:
            original_count = len(filtered_bboxes)
            final_filtered_bboxes = []
            
            for pred_box in filtered_bboxes:
                should_filter = False
                max_overlap_rate = 0.0
                
                # 检查当前预测框与所有hatch框的overlap rate
                for hatch_box in bboxes_hatch:
                    overlap_rate = calculate_overlap_rate(pred_box, hatch_box)
                    max_overlap_rate = max(max_overlap_rate, overlap_rate)
                    
                    # 如果overlap rate > 0.2，则过滤掉该预测框
                    if overlap_rate > 0.2:
                        should_filter = True
                        break
                
                if not should_filter:
                    final_filtered_bboxes.append(pred_box)
            
            filtered_bboxes = final_filtered_bboxes
            hatch_filtered_count = len(filtered_bboxes)
            print(f"Hatch过滤: 原始预测框数量 {original_count} -> 过滤后数量 {hatch_filtered_count} (使用overlap_rate > 0.2过滤)")
        
        # 使用NMS进一步过滤重复检测
        if len(filtered_bboxes) > 0:
            # 转换格式以便使用NMS
            converted_bboxes = convert(filtered_bboxes)
            if converted_bboxes is not None:
                nms_filtered_bboxes = nms(converted_bboxes, iou_threshold=best_iou)
                print(f"NMS过滤: Hatch过滤后数量 {len(filtered_bboxes)} -> NMS后数量 {len(nms_filtered_bboxes)}")
                filtered_bboxes = nms_filtered_bboxes
        
        
        # 转换为draw_rectangle_in_dxf需要的格式
        bbox_list = []
        for bbox in filtered_bboxes:
            x1, y1, x2, y2, conf = bbox[:5]
            ret = yoloxyxy2dxfxyxy([x1, y1, x2, y2, conf])
            bbox_list.append(ret)

        out_post_best_dir = os.path.join(work_dir, "out_post_best")
        draw_rectangle_in_dxf(dxf_file_path, out_post_best_dir, bbox_list, suffix="{}_Holes_{:.1f}_{:.1f}_best.dxf".format("{}", best_conf, best_iou))

        exit()
    
    if args.clear:
        clear_space(work_dir)
        # exit()
    # convert dxf to json file 
    json_path = dxf2json(args.dxfpath, args.dxfname, args.dxfpath)
    json_path = os.path.abspath(json_path)
    output_path = json_file_path.replace("target_json", "target_png").replace(".json", ".png")
    sliding_path = os.path.join(work_dir, 'sliding')
    renderer = DXFRenderer(max_size=args.max_size, min_size=args.min_size, padding_ratio=args.padding_ratio, patch_size=args.patch_size, overlap=args.overlap, auto_size=args.auto_size, factor=args.factor, width=3)
    
    
    # using segment.py
    if args.inference_only:
        print("进入inference_only模式...")
        
        # 1. 尝试从输入DXF文件中提取hatch信息（用于过滤）
        bboxes_hatch = None
        try:
            # 从输入DXF文件中提取hatch信息，这里假设有一个特定的图层包含hatch
            # 如果没有专门的hatch图层，可以设置为None
            converter_main = DXFConverterV2("HATCH")  # 假设hatch在HATCH图层，可根据实际情况调整
            _, bboxes_hatch = converter_main.convert_file(dxf_file_path, json_file_path)
            if bboxes_hatch and len(bboxes_hatch) > 0:
                print(f"提取到 {len(bboxes_hatch)} 个hatch区域用于过滤")
            else:
                bboxes_hatch = None
                print("未找到hatch信息，跳过hatch过滤")
        except Exception as e:
            print(f"提取hatch信息失败: {e}, 跳过hatch过滤")
            bboxes_hatch = None
        
        # 2. 进行推理
        main_bboxes = load_data_and_get_main_bbox(json_file_path)
        print("Main BBoxes = ", main_bboxes)
        time.sleep(2)
        
        # 渲染图像
        for bbox_ in main_bboxes:
            bbox_ = [bbox_['x1'], bbox_['y1'], bbox_['x2'], bbox_['y2']]
            renderer.render(json_path, output_path, bbox=bbox_) # make data in sliding folder
        
        model_path = args.model_path
        model = YOLO(model_path)
        dxf_bboxes = []
        
        # 对所有图像进行推理
        print("开始推理...")
        for folder in glob(os.path.join(sliding_path, "*")):
            for image_path in tqdm(glob(os.path.join(folder, "*.png")), "Inferring:..."):
                json_metadata_path = os.path.join(os.path.dirname(image_path), "meta_data.json")
                with open(json_metadata_path, 'r') as f:
                    metadata = json.load(f)
                try:
                    string = os.path.basename(image_path).split(".")[0].split("_")
                    patch_x, patch_y = int(string[2]), int(string[3])
                    predictions = predict_image(
                        model=model,
                        image_path=image_path,
                        conf_threshold=0.1,  # 先用较低阈值推理，后面再用args.conf过滤
                        imgsz=args.patch_size,
                        verbose=False,
                        save=True,
                    )
                    for i, pred in enumerate(predictions):
                        x1, y1, x2, y2, conf, class_id = pred
                        dxf_x1, dxf_y1 = convert_png2dxf_coord(x1, y1, patch_x, patch_y, metadata)
                        dxf_x2, dxf_y2 = convert_png2dxf_coord(x2, y2, patch_x, patch_y, metadata)
                        dxf_bboxes.append([dxf_x1, dxf_y2, dxf_x2, dxf_y1, conf])
                except:
                    print("Meeting whole.png")
        
        print(f"推理完成，原始检测数量: {len(dxf_bboxes)}")
        
        # 3. 使用args.conf过滤置信度
        filtered_bboxes = []
        for bbox in dxf_bboxes:
            if len(bbox) >= 5 and bbox[4] >= args.conf:
                filtered_bboxes.append(bbox)
        
        print(f"置信度过滤 (conf >= {args.conf}): {len(dxf_bboxes)} -> {len(filtered_bboxes)}")
        
        # 4. 使用hatch进行过滤（如果有的话）
        if bboxes_hatch is not None and len(bboxes_hatch) > 0:
            original_count = len(filtered_bboxes)
            final_filtered_bboxes = []
            
            for pred_box in filtered_bboxes:
                should_filter = False
                
                # 检查当前预测框与所有hatch框的overlap rate
                for hatch_box in bboxes_hatch:
                    overlap_rate = calculate_overlap_rate(pred_box, hatch_box)
                    
                    # 如果overlap rate > 0.2，则过滤掉该预测框
                    if overlap_rate > 0.2:
                        should_filter = True
                        break
                
                if not should_filter:
                    final_filtered_bboxes.append(pred_box)
            
            filtered_bboxes = final_filtered_bboxes
            hatch_filtered_count = len(filtered_bboxes)
            print(f"Hatch过滤: {original_count} -> {hatch_filtered_count} (使用overlap_rate > 0.2过滤)")
        
        # 5. 使用NMS进一步过滤重复检测
        if len(filtered_bboxes) > 0:
            converted_bboxes = convert(filtered_bboxes)
            if converted_bboxes is not None:
                nms_filtered_bboxes = nms(converted_bboxes, iou_threshold=0.3)
                print(f"NMS过滤: {len(filtered_bboxes)} -> {len(nms_filtered_bboxes)}")
                filtered_bboxes = nms_filtered_bboxes
        
        # 保存中间结果（调试用）
        nms_results_inference_path = os.path.join(work_dir, f"nms_results_inference_{base_name}.txt")
        save_debug_file(filtered_bboxes, nms_results_inference_path)
        
        # 6. 后处理：提取实体信息
        print("开始提取所有实体...")
        allbe_extractor = AllbeExtractor(json_file_path)
        allbe_data = allbe_extractor.extract_all()
        allbe_json_path = os.path.join(work_dir, f"allbe_inference_{base_name}.json")
        save_debug_file(allbe_data, allbe_json_path)
        
        # 提取详细实体信息
        allbe_detailed_extractor = AllbeDetailedExtractor(json_file_path)
        allbe_detailed_data = allbe_detailed_extractor.extract_all()
        allbe_detailed_json_path = os.path.join(work_dir, f"allbe_detailed_inference_{base_name}.json")
        save_debug_file(allbe_detailed_data, allbe_detailed_json_path)
        
        # 7. 提取闭合连通分量
        print("开始提取闭合连通分量...")
        close_extractor = CloseExtractor(allbe_json_path, scale_factor=1.5, tolerance_factor=10, debug=False)
        close_data = close_extractor.extract_closed_components(filtered_bboxes, require_degree_2=False)
        close_json_path = os.path.join(work_dir, f"close_inference_{base_name}.json")
        save_debug_file(close_data, close_json_path)
        
        # 如果需要可视化
        # if args.debug:
        #     close_extractor.visualize(close_data)
        
        # 8. 提取尺寸信息
        print("开始提取尺寸信息...")
        extractor = DimensionExtractor(
            allbe_path=allbe_detailed_json_path,
            close_path=close_json_path,
            t1=1.0,
            t2=1.0,
            debug=False,
            slope_tolerance=0.1,
            parallel_distance_threshold=999980.0,
            midpoint_distance_threshold=9999400.0
        )
        
        output_path_dim = os.path.join(work_dir, f"extracted_dimensions_inference_{base_name}.json")
        results = extractor.save_results(output_path_dim)
        extractor.generate_excel_reports(results, output_dir=work_dir)
        
        print(f"Dimension提取完成！")
        print(f"处理了 {len(results)} 个检测目标")
        
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
        
        # 9. 绘制最终结果到指定文件夹
        bbox_list_final = []
        try:
            for item in results:
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
                bbox_list_final.append(ret)
        except:
            # 如果dimension提取失败，使用close结果
            print("使用close结果作为最终输出...")
            for item in close_data['closed_components']:
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
                bbox_list_final.append(ret)
        
        # 确保输出目录存在
        final_output_dir = os.path.join(work_dir, "out_post_final")
        os.makedirs(final_output_dir, exist_ok=True)
        
        # 绘制最终结果
        draw_rectangle_in_dxf(
            dxf_file_path, 
            final_output_dir, 
            bbox_list_final
        )
        
        print(f"Inference完成！最终结果已保存到 {final_output_dir}")
        print(f"最终检测数量: {len(bbox_list_final)}")
        print(f"使用置信度阈值: {args.conf}")
        
        exit()
    else:
        if args.segment_bbox:
            main_bboxes = load_data_and_get_main_bbox(json_file_path)
            print("Main BBoxes = ", main_bboxes)
            time.sleep(2)
            for bbox_ in main_bboxes:
                bbox_ = [bbox_['x1'], bbox_['y1'], bbox_['x2'], bbox_['y2']]
                renderer.render(json_path, output_path, bbox=bbox_) # make data in sliding folder
            model_path = args.model_path
            model = YOLO(model_path)
            dxf_bboxes = []
            # Inferring for all images in folders
            for folder in glob(os.path.join(sliding_path, "*")):
                for image_path in tqdm(glob(os.path.join(folder, "*.png")), "Inferring:..."):
                    json_metadata_path = os.path.join(os.path.dirname(image_path), "meta_data.json")
                    with open(json_metadata_path, 'r') as f:
                        metadata = json.load(f)
                    try:
                        string = os.path.basename(image_path).split(".")[0].split("_")
                        patch_x, patch_y = int(string[2]), int(string[3])
                        predictions = predict_image(
                            model=model,
                            image_path=image_path,
                            conf_threshold=0.1,
                            imgsz=args.patch_size,
                            verbose=False,
                            save=True,
                            work_dir=work_dir, 
                        )
                        print(f"Patch size = {patch_x},{patch_y}")
                        for i, pred in enumerate(predictions):
                            x1, y1, x2, y2, conf, class_id = pred
                            dxf_x1, dxf_y1 = convert_png2dxf_coord(x1, y1, patch_x, patch_y, metadata)
                            dxf_x2, dxf_y2 = convert_png2dxf_coord(x2, y2, patch_x, patch_y, metadata)
                            print(f"目标 {i+1}:")
                            print(f"patch size = {patch_x}, {patch_y}")
                            print(f"- 边界框坐标: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")
                            print(f"- dxf边界框坐标: ({dxf_x1:.2f}, {dxf_y2:.2f}, {dxf_x2:.2f}, {dxf_y1:.2f})")
                            print(f"- 置信度: {conf:.3f}")
                            print(f"- 类别ID: {class_id}")
                            width, height = dxf_x2 - dxf_x1, dxf_y1 - dxf_y2
                            # if height > width:
                            #     width, height = height, width
                            # if width / height > 1.5:
                            #     print("Filter bbox")
                            #     continue
                            dxf_bboxes.append([dxf_x1, dxf_y2, dxf_x2, dxf_y1, conf])
                    except:
                        print("Meeting whole.png")
            nms_dxf_bboxes = nms(dxf_bboxes, 0.3)
            print(dxf_bboxes)
            print(nms_dxf_bboxes)
            print(f"Before nms bboxes number = {len(dxf_bboxes)}")
            print(f"After nms bboxes number = {len(nms_dxf_bboxes)}")
            
            # 保存NMS结果（调试用）
            nms_results_path = os.path.join(work_dir, f"nms_results_{base_name}.txt")
            save_debug_file(nms_dxf_bboxes, nms_results_path)

            bbox_list = []
            for bbox in nms_dxf_bboxes:
                x1, y1, x2, y2, conf = bbox 
                ret = yoloxyxy2dxfxyxy([x1, y1, x2, y2, conf])
                bbox_list.append(ret)  
                    
            dxf_output_dir = os.path.join(work_dir, "out")
            draw_rectangle_in_dxf(dxf_file_path, dxf_output_dir, bbox_list)
            
            '''
                1. 调用extract_allbe.py提取所有实体
                2. 从nms_results.txt中提取bbox（x1,y1,x2,y2,conf）,然后调用extract_close.py提取闭合连通分量
            '''

            # 1. 调用extract_allbe.py提取所有实体
            print("开始提取所有实体...")
            allbe_extractor = AllbeExtractor(json_file_path)
            allbe_data = allbe_extractor.extract_all()
            allbe_json_path = os.path.join(work_dir, f"allbe_{base_name}.json")
            save_debug_file(allbe_data, allbe_json_path)
            
            # 调用extract_allbe_detailed.py提取详细实体信息
            allbe_detailed_extractor = AllbeDetailedExtractor(json_file_path)
            allbe_detailed_data = allbe_detailed_extractor.extract_all()
            allbe_detailed_json_path = os.path.join(work_dir, f"allbe_detailed_{base_name}.json")
            save_debug_file(allbe_detailed_data, allbe_detailed_json_path)
            
            # 2. 提取闭合连通分量（直接使用nms_dxf_bboxes，避免文件读写）
            print("开始提取闭合连通分量...")
            close_extractor = CloseExtractor(allbe_json_path, scale_factor=1.5, tolerance_factor=10, debug=False)
            close_data = close_extractor.extract_closed_components(nms_dxf_bboxes, require_degree_2=False)
            close_json_path = os.path.join(work_dir, f"close_{base_name}.json")
            save_debug_file(close_data, close_json_path)
            
            # 如果需要可视化
            # if args.debug:
            #     close_extractor.visualize(close_data)
            
            # 从close_data直接获取处理后的bbox
            bbox_list = []
            for item in close_data['closed_components']:
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
            out_post_dir = os.path.join(work_dir, "out_post")
            draw_rectangle_in_dxf(dxf_file_path, out_post_dir, bbox_list)
            print(f"闭合连通分量处理完成，结果已保存到 {out_post_dir}")
            
            # 信息提取
            # 设置容差
            t1 = 1.0  # 判断点是否在线上的容差
            t2 = 1.0  # 找到参考线的容差
            slope_tolerance = 0.1  # 判断直线平行的斜率容差
            parallel_distance_threshold = 999980.0  # 平行线之间的距离阈值
            midpoint_distance_threshold = 9999400.0  # 线段中点之间的距离阈值
            
            # 创建提取器并运行
            extractor = DimensionExtractor(
                allbe_path=allbe_detailed_json_path,
                close_path=close_json_path,
                t1=t1,
                t2=t2,
                debug=False,
                slope_tolerance=slope_tolerance,
                parallel_distance_threshold=parallel_distance_threshold,
                midpoint_distance_threshold=midpoint_distance_threshold
            )
            
            # 执行提取并保存结果
            output_path_dim = os.path.join(work_dir, f"extracted_dimensions_{base_name}.json")
            results = extractor.save_results(output_path_dim)
            extractor.generate_excel_reports(results, output_dir=work_dir)
            print(f"Dimension提取完成！")
            print(f"处理了 {len(results)} 个检测目标")
            print(f"结果已保存到: {output_path_dim}")
            
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

            # 直接使用results结果绘制最终维度框
            bbox_list = []
            for item in results:
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
            
            out_post_dim_dir = os.path.join(work_dir, "out_post_dim")
            draw_rectangle_in_dxf(dxf_file_path, out_post_dim_dir, bbox_list)

            return results