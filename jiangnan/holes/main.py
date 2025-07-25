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
from preprocess.load import dxf2json
from convert2png_v2 import DXFRenderer
from yolo_test import predict_image, visualize_predictions, convert_png2dxf_coord ,nms, predict_image_tta
from draw_dxf import draw_rectangle_in_dxf, yoloxyxy2dxfxyxy
from load_v2 import DXFConverterV2
from evaluate import evaluate, convert, analyze_confidence_thresholds, calculate_iou, calculate_overlap_rate
# from statistic_holes import EntityAnalyzer
from filter_bbox import load_data_and_get_main_bbox
from holes.extract_dimen import DimensionExtractor
# 新增导入模块
from extract_allbe import EntityExtractor as AllbeExtractor
from extract_allbe_detailed import EntityExtractor as AllbeDetailedExtractor
from extract_close import CloseExtractor

def clear_space():
    shutil.rmtree('sliding', ignore_errors=True)
    # shutil.rmtree('out', ignore_errors=True)
    shutil.rmtree('runs', ignore_errors=True)
    # for file in glob("*.json"):
        # os.remove(file)
    # for file in glob("*.txt"):
        # os.remove(file)
    
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help="whether in debug mode")
    parser.add_argument('--clear', action='store_true', help="whether to clear")
    # load.py
    parser.add_argument('--dxfpath', type=str, default="./", help="dxf path")
    parser.add_argument('--dxfname', type=str, default="data1114_v2.dxf", help="dxf name")
    parser.add_argument('--json_path', type=str, default="data1114_v2.json", help="converted output json path")
    # convert2png_v2.py
    parser.add_argument('--auto_size', action='store_true', help="whether using automatic size")
    parser.add_argument('--factor', type=float, default=0.16, help="scale factor")
    parser.add_argument('--max_size', type=int, default=1024, help="max size of canvas, 1024 / 59400 * max(width, height)")
    parser.add_argument('--min_size', type=int, default=1024, help="min size of canvas, 1024 / 59400 * min(width, height)")
    parser.add_argument('--padding_ratio', type=float, default=0.05, help="padding ratio")
    parser.add_argument('--patch_size', type=int, default=2560, help="sliding patch size")
    parser.add_argument('--overlap', type=float, default=0.5, help="overlap ratio")
    parser.add_argument('--segment_bbox', action='store_true', help="bbox from segment.py")
    # yolo_test.py
    parser.add_argument('--model_path', type=str, default="best.pt", help="yolo pt path")
    # draw_dxf.py
    parser.add_argument('--dxf_output_path', type=str, default="./out", help='dirname of output dxf')
    # evaluate.py
    parser.add_argument('--evaluate_only', action='store_true', help="whether to evaluate")
    parser.add_argument('--dxf_path_gt', type=str, default="./data1114_Holes_gt.dxf", help="path to gt dxf")
    parser.add_argument('--output_path_gt', type=str, default="./data1114_Holes_gt.json", help="output path to gt json")
    parser.add_argument('--selected_layer_gt', type=str, default="Holes", help="gt labeled layer name")
    parser.add_argument('--abandon_layer_gt', type=str, default="开孔识别结果", help="abandon")
    parser.add_argument('--dxf_path_pred', type=str, default="./data1114_Holes_pred.dxf", help="path to gt json")
    parser.add_argument('--output_path_pred', type=str, default="./data1114_Holes_pred.json", help="output path to pred json")
    parser.add_argument('--selected_layer_pred', type=str, default="Holes", help="pred labeled layer name")
    # final test version
    parser.add_argument('--inference_only', action='store_true', help="final version")
    parser.add_argument('--conf', type=float, default=0.5, help="final conf")

    args = parser.parse_args()
    
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
        analyze_confidence_thresholds("eval_results_all.json")
        # print(f"Evaluation results: {results}")

        # 读取最佳评估结果
        with open("eval_results.txt", "r") as f:
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

        draw_rectangle_in_dxf(os.path.join(args.dxfpath, args.dxfname), "./out_post_best", bbox_list, suffix="{}_Holes_{:.1f}_{:.1f}_best.dxf".format("{}", best_conf, best_iou))


        exit()
    
    if args.clear:
        clear_space()
        # exit()
    # convert dxf to json file 
    json_path = dxf2json(args.dxfpath, args.dxfname, args.dxfpath)
    json_path = os.path.abspath(json_path)
    output_path = args.json_path.replace("target_json", "target_png").replace(".json", ".png")
    sliding_path = os.path.join(os.path.dirname(json_path), 'sliding')
    renderer = DXFRenderer(max_size=args.max_size, min_size=args.min_size, padding_ratio=args.padding_ratio, patch_size=args.patch_size, overlap=args.overlap, auto_size=args.auto_size, factor=args.factor)
    
    
    # using segment.py
    if args.inference_only:
        print("进入inference_only模式...")
        
        # 1. 尝试从输入DXF文件中提取hatch信息（用于过滤）
        bboxes_hatch = None
        try:
            # 从输入DXF文件中提取hatch信息，这里假设有一个特定的图层包含hatch
            # 如果没有专门的hatch图层，可以设置为None
            converter_main = DXFConverterV2("HATCH")  # 假设hatch在HATCH图层，可根据实际情况调整
            _, bboxes_hatch = converter_main.convert_file(os.path.join(args.dxfpath, args.dxfname), args.json_path)
            if bboxes_hatch and len(bboxes_hatch) > 0:
                print(f"提取到 {len(bboxes_hatch)} 个hatch区域用于过滤")
            else:
                bboxes_hatch = None
                print("未找到hatch信息，跳过hatch过滤")
        except Exception as e:
            print(f"提取hatch信息失败: {e}, 跳过hatch过滤")
            bboxes_hatch = None
        
        # 2. 进行推理
        main_bboxes = load_data_and_get_main_bbox(args.json_path)
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
        
        # 保存中间结果
        with open("nms_results_inference.txt", "w") as f:
            for bbox in filtered_bboxes:
                x1, y1, x2, y2, conf = bbox 
                f.write(f"{x1},{y1},{x2},{y2},{conf}\n")
        
        # 6. 后处理：提取实体信息
        print("开始提取所有实体...")
        allbe_extractor = AllbeExtractor(args.json_path)
        allbe_data = allbe_extractor.extract_all()
        allbe_json_path = "allbe_inference.json"
        with open(allbe_json_path, "w", encoding='utf-8') as f:
            json.dump(allbe_data, f, ensure_ascii=False, indent=4)
        
        # 提取详细实体信息
        allbe_detailed_extractor = AllbeDetailedExtractor(args.json_path)
        allbe_detailed_data = allbe_detailed_extractor.extract_all()
        allbe_detailed_json_path = "allbe_detailed_inference.json"
        with open(allbe_detailed_json_path, "w", encoding='utf-8') as f:
            json.dump(allbe_detailed_data, f, ensure_ascii=False, indent=4)
        
        # 7. 提取闭合连通分量
        print("开始提取闭合连通分量...")
        bbox_list_from_nms = []
        with open("nms_results_inference.txt", "r") as f:
            for line in f.readlines():
                line = line.strip().split(",")
                x1, y1, x2, y2, conf = eval(line[0]), eval(line[1]), eval(line[2]), eval(line[3]), eval(line[4])
                bbox_list_from_nms.append([x1, y1, x2, y2, conf])
        
        close_extractor = CloseExtractor(allbe_json_path, scale_factor=1.5, tolerance_factor=10, debug=False)
        close_json_path = "close_inference.json"
        close_extractor.save_to_json(close_json_path, bbox_list_from_nms, require_degree_2=False, visualize=True)
        
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
        
        output_path_dim = f"extracted_dimensions_inference-{args.dxfname.replace('.dxf', '')}.json"
        results = extractor.save_results(output_path_dim)
        
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
            with open(output_path_dim, "r", encoding='utf-8') as f:
                data = json.load(f)
            for item in data:
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
            with open(close_json_path, "r", encoding='utf-8') as f:
                data = json.load(f)
            for item in data['closed_components']:
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
        os.makedirs("./out_post_final", exist_ok=True)
        
        # 绘制最终结果
        draw_rectangle_in_dxf(
            os.path.join(args.dxfpath, args.dxfname), 
            "./out_post_final", 
            bbox_list_final
        )
        
        print(f"Inference完成！最终结果已保存到 ./out_post_final")
        print(f"最终检测数量: {len(bbox_list_final)}")
        print(f"使用置信度阈值: {args.conf}")
        
        exit()
    else:
        if args.segment_bbox:
            main_bboxes = load_data_and_get_main_bbox(args.json_path)
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
                    json_path = os.path.join(os.path.dirname(image_path), "meta_data.json")
                    with open(json_path, 'r') as f:
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
            with open("nms_results.txt", "w") as f:
                for bbox in nms_dxf_bboxes:
                    x1, y1, x2, y2, conf = bbox 
                    f.write(f"{x1},{y1},{x2},{y2},{conf}\n")

            bbox_list = []
            with open("nms_results.txt", "r") as f:
                for line in f.readlines():
                    line = line.strip().split(",")
                    x1, y1, x2, y2, conf = eval(line[0]), eval(line[1]), eval(line[2]), eval(line[3]), eval(line[4]) 
                    ret = yoloxyxy2dxfxyxy([x1, y1, x2, y2, conf])
                    bbox_list.append(ret)  
                    
            draw_rectangle_in_dxf(os.path.join(args.dxfpath, args.dxfname), args.dxf_output_path, bbox_list)
            bbox_path = "nms_results.txt" # same path as above
            
            '''
                1. 调用extract_allbe.py提取所有实体
                2. 从nms_results.txt中提取bbox（x1,y1,x2,y2,conf）,然后调用extract_close.py提取闭合连通分量
            '''

            # 1. 调用extract_allbe.py提取所有实体
            print("开始提取所有实体...")
            allbe_extractor = AllbeExtractor(args.json_path)
            allbe_data = allbe_extractor.extract_all()
            allbe_json_path = "allbe.json"
            with open(allbe_json_path, "w", encoding='utf-8') as f:
                json.dump(allbe_data, f, ensure_ascii=False, indent=4)
            
            # 调用extract_allbe_detailed.py提取详细实体信息
            allbe_detailed_extractor = AllbeDetailedExtractor(args.json_path)
            allbe_detailed_data = allbe_detailed_extractor.extract_all()
            allbe_detailed_json_path = "allbe_detailed.json"
            with open(allbe_detailed_json_path, "w", encoding='utf-8') as f:
                json.dump(allbe_detailed_data, f, ensure_ascii=False, indent=4)
            # 2. 从nms_results.txt中提取bbox，然后调用extract_close.py提取闭合连通分量
            print("开始提取闭合连通分量...")
            # 从nms_results.txt读取bbox
            bbox_list_from_nms = []
            with open("nms_results.txt", "r") as f:
                for line in f.readlines():
                    line = line.strip().split(",")
                    x1, y1, x2, y2, conf = eval(line[0]), eval(line[1]), eval(line[2]), eval(line[3]), eval(line[4])
                    bbox_list_from_nms.append([x1, y1, x2, y2, conf])
            
            # 调用extract_close.py处理闭合连通分量
            close_extractor = CloseExtractor(allbe_json_path, scale_factor=1.5, tolerance_factor=10, debug=False)
            close_json_path = "close.json"
            close_extractor.save_to_json(close_json_path, bbox_list_from_nms, require_degree_2=False, visualize=True)
            
            # 从close.json读取处理后的bbox
            bbox_list = []
            with open(close_json_path, "r", encoding='utf-8') as f:
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
            final_path = args.dxf_output_path.replace(".dxf", "_final.dxf")
            draw_rectangle_in_dxf(os.path.join(args.dxfpath, args.dxfname), "./out_post", bbox_list)
            print(f"闭合连通分量处理完成，结果已保存到 ./out_post")
            
            # 信息提取
            allbe_path = "allbe_detailed.json"  # 包含所有实体信息的文件
            close_path = "close.json"  # 包含检测到的目标的文件
            output_path = f"extracted_dimensions-{args.dxfname.replace('.dxf', '')}.json"  # 输出文件
            
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
            with open(f"extracted_dimensions-{args.dxfname.replace('.dxf', '')}.json", "r", encoding='utf-8') as f:
                data = json.load(f)
            for item in data:
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
            draw_rectangle_in_dxf(os.path.join(args.dxfpath, args.dxfname), "./out_post_dim", bbox_list)

'''
TODO 
1. 优化推理算法（无本地验证）
2. 合并后续两个程序，在最优提取信息上提取信息。同时支持人工修正。

'''