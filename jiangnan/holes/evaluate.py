import json
import numpy as np
import os
import sys
from load_v2 import DXFConverterV2
from yolo_test import nms 

def calculate_iou(box1: list, box2: list) -> float:
    # Extract coordinates (ignore confidence values)
    x1_1, y1_1, x2_1, y2_1 = min(box1[0], box1[2]), min(box1[1], box1[3]), max(box1[0], box1[2]), max(box1[1], box1[3]) # box1[:4]
    x1_2, y1_2, x2_2, y2_2 = min(box2[0], box2[2]), min(box2[1], box2[3]), max(box2[0], box2[2]), max(box2[1], box2[3])
    
    # Calculate intersection coordinates
    x_left = max(x1_1, x1_2)
    x_right = min(x2_1, x2_2)
    y_top = max(y1_1, y1_2)
    y_bottom = min(y2_1, y2_2)
    
    # 正确处理负坐标系的情况
    # 对于y坐标，需要特别处理负值坐标系
    # if x1_1 < 0 and x1_2 < 0 and x2_1 < 0 and x2_2 < 0:
    #     # 负坐标系下，更小的值实际上离原点更远
    #     # 在负坐标系中，y_top应该是较大的值，y_bottom应该是较小的值
    #     x_right = min(x1_1, x1_2)  # 在负坐标系中，较小值是绝对值更大的负数
    #     x_left = max(x2_1, x2_2)  # 在负坐标系中，较大值是绝对值更小的负数
    # else:
    #     # 传统坐标系处理
    #     x_right = max(x1_1, x1_2)
    #     x_left = min(x2_1, x2_2)

    # if y1_1 < 0 and y1_2 < 0 and y2_1 < 0 and y2_2 < 0:
    #     # 负坐标系下，更小的值实际上离原点更远
    #     # 在负坐标系中，y_top应该是较大的值，y_bottom应该是较小的值
    #     y_top = min(y1_1, y1_2)  # 在负坐标系中，较小值是绝对值更大的负数
    #     y_bottom = max(y2_1, y2_2)  # 在负坐标系中，较大值是绝对值更小的负数
    # else:
    #     # 传统坐标系处理
    #     y_top = max(y1_1, y1_2)
    #     y_bottom = min(y2_1, y2_2)
    
    # print(f"x_left = {x_left}, y_top = {y_top}, x_right = {x_right}, y_bottom = {y_bottom}")
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

def calculate_overlap_rate(pred_box: list, hatch_box: list) -> float:
    """
    计算预测框和hatch框之间的overlap面积与预测框面积的比值
    Args:
        pred_box: 预测框 [x1, y1, x2, y2, conf]
        hatch_box: hatch框 [x1, y1, x2, y2, conf]
    Returns:
        float: overlap面积 / 预测框面积
    """
    # 提取坐标（忽略置信度值）
    x1_pred, y1_pred, x2_pred, y2_pred = min(pred_box[0], pred_box[2]), min(pred_box[1], pred_box[3]), max(pred_box[0], pred_box[2]), max(pred_box[1], pred_box[3])
    x1_hatch, y1_hatch, x2_hatch, y2_hatch = min(hatch_box[0], hatch_box[2]), min(hatch_box[1], hatch_box[3]), max(hatch_box[0], hatch_box[2]), max(hatch_box[1], hatch_box[3])
    
    # 计算交集坐标
    x_left = max(x1_pred, x1_hatch)
    x_right = min(x2_pred, x2_hatch)
    y_top = max(y1_pred, y1_hatch)
    y_bottom = min(y2_pred, y2_hatch)
    
    # 检查是否有交集
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # 计算交集面积
    overlap_area = (x_right - x_left) * (y_bottom - y_top)
    
    # 计算预测框面积
    pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    
    # 返回overlap面积与预测框面积的比值
    overlap_rate = overlap_area / pred_area if pred_area > 0 else 0.0
    return overlap_rate

def is_bbox_contained(pred_box: list, hatch_box: list) -> bool:
    """
    检查pred_box是否完全包含在hatch_box内
    Args:
        pred_box: 预测框 [x1, y1, x2, y2, conf]
        hatch_box: hatch框 [x1, y1, x2, y2, conf]
    Returns:
        bool: 如果pred_box完全包含在hatch_box内则返回True
    """
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_box[0], pred_box[1], pred_box[2], pred_box[3]
    hatch_x1, hatch_y1, hatch_x2, hatch_y2 = hatch_box[0], hatch_box[1], hatch_box[2], hatch_box[3]
    
    # 确保坐标顺序正确
    pred_x1, pred_x2 = min(pred_x1, pred_x2), max(pred_x1, pred_x2)
    pred_y1, pred_y2 = min(pred_y1, pred_y2), max(pred_y1, pred_y2)
    hatch_x1, hatch_x2 = min(hatch_x1, hatch_x2), max(hatch_x1, hatch_x2)
    hatch_y1, hatch_y2 = min(hatch_y1, hatch_y2), max(hatch_y1, hatch_y2)
    
    # 检查是否完全包含：hatch框的左下角小于等于pred框的左下角，hatch框的右上角大于等于pred框的右上角
    return (hatch_x1 <= pred_x1 and hatch_y1 <= pred_y1 and 
            hatch_x2 >= pred_x2 and hatch_y2 >= pred_y2)

def calculate_metrics(pred_boxes: list, gt_boxes: list, conf_threshold: float, iou_threshold: float) -> dict:
    """
    Calculate precision and recall given prediction and ground truth boxes.
    """
    # Filter predictions based on confidence threshold
    filtered_preds = [box for box in pred_boxes if box[4] >= conf_threshold]
    
    # Initialize counters
    true_positives = 0
    matched_gt = set()
    
    # Match predictions to ground truth
    for pred_box in filtered_preds:
        best_iou = 0
        best_gt_idx = -1
        
        # Find best matching ground truth box
        for i, gt_box in enumerate(gt_boxes):
            if i not in matched_gt:
                iou = calculate_iou(pred_box, gt_box)
                # print(11111)
                # print(f"IoU = {iou}")

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
        
        # If IoU exceeds threshold, count as true positive
        if best_iou >= iou_threshold and best_gt_idx != -1:
            true_positives += 1
            matched_gt.add(best_gt_idx)
    
    # Calculate metrics
    false_positives = len(filtered_preds) - true_positives
    false_negatives = len(gt_boxes) - true_positives
    
    precision = true_positives / len(filtered_preds) if filtered_preds else 0
    recall = true_positives / len(gt_boxes) if gt_boxes else 0
    
    return {
        "P": precision, 
        "R": recall, 
        "TP": true_positives,
        "FP": false_positives,
        "FN": false_negatives,
        "total_predictions": len(filtered_preds),
        "total_gt": len(gt_boxes)
    }

def evaluate(bboxes1: list[list], bboxes2: list[list], bboxes3: list[list] = None, bboxes_hatch: list[list] = None, conf1=None, conf2=None) -> dict:
    """
    Evaluate object detection results.
    bboxes1: predicted boxes with confidence scores
    bboxes2: ground truth boxes
    bboxes3: abandon boxes
    bboxes_hatch: hatch boxes for filtering predictions
    conf1: confidence threshold for predictions
    conf2: IoU threshold for matching
    """
    
    # 使用新的过滤逻辑：计算overlap面积与预测框面积的比值（hatch_rate），如果hatch_rate > 0.2则过滤掉预测框
    if bboxes_hatch is not None and len(bboxes_hatch) > 0:
        original_count = len(bboxes1) if bboxes1 else 0
        filtered_bboxes = []
        
        for pred_box in bboxes1:
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
                filtered_bboxes.append(pred_box)
        
        bboxes1 = filtered_bboxes
        filtered_count = len(bboxes1)
        print(f"Hatch过滤: 原始预测框数量 {original_count} -> 过滤后数量 {filtered_count} (使用overlap_rate > 0.2过滤)")

    if bboxes3 is not None:
        # 过滤掉与舍弃框匹配的预测框
        bboxes1 = [box for box in bboxes1 if not any(calculate_iou(box, abandon_box) > 0.1 for abandon_box in bboxes3)]

    if conf1 is None and conf2 is None:
        # Automatic threshold search
        best_metrics = {
            "P": 0, 
            "R": 0,
            "TP": 0,
            "FP": 0, 
            "FN": 0,
            "total_predictions": 0,
            "total_gt": len(bboxes2) if bboxes2 else 0
        }
        best_f1 = 0
        best_conf = 0
        best_iou = 0
        
        # 保存所有conf和iou阈值组合的结果
        all_results = {}
        
        # 创建CSV文件保存详细统计结果
        with open("detection_statistics.csv", "w") as csv_file:
            csv_file.write("置信度阈值,IoU阈值,准确率,召回率,F1值,TP数量,FP数量,FN数量,预测总数,真实总数\n")
        
        # Grid search over confidence and IoU thresholds
        for conf_thresh in np.arange(0.1, 1.0, 0.1):
            conf_key = f"{conf_thresh:.1f}"
            all_results[conf_key] = {}
            
            for iou_thresh in np.arange(0.1, 1.0, 0.1):
                iou_key = f"{iou_thresh:.1f}"
                
                metrics = calculate_metrics(bboxes1, bboxes2, conf_thresh, iou_thresh)
                
                # Calculate F1 score
                p, r = metrics["P"], metrics["R"]
                f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
                print(f"conf = {conf_thresh}, iou = {iou_thresh}, P = {p}, R = {r}, f1 = {f1}, TP = {metrics['TP']}, FP = {metrics['FP']}, FN = {metrics['FN']}")
                
                # 保存当前conf和iou阈值的结果
                all_results[conf_key][iou_key] = {
                    "precision": p,
                    "recall": r,
                    "f1_score": f1,
                    "true_positives": metrics["TP"],
                    "false_positives": metrics["FP"],
                    "false_negatives": metrics["FN"],
                    "total_predictions": metrics["total_predictions"],
                    "total_ground_truth": metrics["total_gt"]
                }
                
                # 将统计数据写入CSV文件
                with open("detection_statistics.csv", "a") as csv_file:
                    csv_file.write(f"{conf_thresh:.1f},{iou_thresh:.1f},{p:.4f},{r:.4f},{f1:.4f},{metrics['TP']},{metrics['FP']},{metrics['FN']},{metrics['total_predictions']},{metrics['total_gt']}\n")
                
                # Update best results if F1 score improves
                if f1 > best_f1:
                    best_f1 = f1
                    best_metrics = metrics
                    best_conf = conf_thresh
                    best_iou = iou_thresh
        
        # Save all results
        with open("eval_results_all.json", "w") as f:
            json.dump(all_results, f, indent=4)
        
        # Save best results
        with open("eval_results.txt", "w") as f:
            results = {
                "best_confidence_threshold": float(best_conf),
                "best_iou_threshold": float(best_iou),
                "precision": best_metrics["P"],
                "recall": best_metrics["R"],
                "f1_score": best_f1,
                "true_positives": best_metrics["TP"],
                "false_positives": best_metrics["FP"],
                "false_negatives": best_metrics["FN"],
                "total_predictions": best_metrics["total_predictions"],
                "total_ground_truth": best_metrics["total_gt"]
            }
            json.dump(results, f, indent=4)
        
        return best_metrics
    else:
        # Use provided thresholds
        metrics = calculate_metrics(bboxes1, bboxes2, conf1, conf2)
        
        # Calculate F1 score
        p, r = metrics["P"], metrics["R"]
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        
        # Save results
        with open("eval_results.txt", "w") as f:
            results = {
                "confidence_threshold": conf1,
                "iou_threshold": conf2,
                "precision": metrics["P"],
                "recall": metrics["R"],
                "f1_score": f1,
                "true_positives": metrics["TP"],
                "false_positives": metrics["FP"],
                "false_negatives": metrics["FN"],
                "total_predictions": metrics["total_predictions"],
                "total_ground_truth": metrics["total_gt"]
            }
            json.dump(results, f, indent=4)
        
        # 将统计数据写入CSV文件
        with open("detection_statistics.csv", "w") as csv_file:
            csv_file.write("置信度阈值,IoU阈值,准确率,召回率,F1值,TP数量,FP数量,FN数量,预测总数,真实总数\n")
            csv_file.write(f"{conf1:.1f},{conf2:.1f},{p:.4f},{r:.4f},{f1:.4f},{metrics['TP']},{metrics['FP']},{metrics['FN']},{metrics['total_predictions']},{metrics['total_gt']}\n")
        
        return metrics

def convert(bboxes: list[list]):
    if bboxes is None:
        return None
    ret = []
    for bbox in bboxes:
        x1, x2 = min(bbox[0], bbox[2]), max(bbox[0], bbox[2])
        y1, y2 = min(bbox[1], bbox[3]), max(bbox[1], bbox[3])
        # # 正确处理负坐标系情况
        # if bbox[1] < 0 and bbox[3] < 0:
        #     # 对于负坐标系，较大的绝对值实际上是较小的值
        #     # 此处y1应该是绝对值较小的值，y2应该是绝对值较大的值
        #     y1, y2 = max(bbox[1], bbox[3]), min(bbox[1], bbox[3])
        # else:
        #     y1, y2 = min(bbox[1], bbox[3]), max(bbox[1], bbox[3])
            
        # 确保每个框都有置信度值
        if len(bbox) > 4:
            conf = bbox[4]
        else:
            # 如果是真实框没有置信度，则添加1.0作为置信度
            conf = 1.0
            
        ret.append([x1, y1, x2, y2, conf])
    
    if len(ret) == 0:
        print("Warning, bbox length = 0")
        return None

    return ret

def analyze_confidence_thresholds(results_file):
    """
    分析每个置信度阈值和IoU阈值组合的性能
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("\n各阈值组合的性能统计:")
    print("-" * 120)
    print(f"{'置信度':^10}{'IoU':^10}{'准确率':^10}{'召回率':^10}{'F1值':^10}{'TP':^8}{'FP':^8}{'FN':^8}{'预测数':^10}{'真实数':^10}{'TP+FP+FN':^10}")
    print("-" * 120)
    
    # 创建一个文件来保存详细的分析结果
    with open("threshold_analysis.txt", "w") as analysis_file:
        analysis_file.write("置信度阈值和IoU阈值组合的性能分析\n")
        analysis_file.write("=" * 100 + "\n\n")
        
        # 遍历所有置信度阈值
        for conf_thresh in sorted(results.keys()):
            analysis_file.write(f"置信度阈值: {conf_thresh}\n")
            analysis_file.write("-" * 100 + "\n")
            analysis_file.write(f"{'IoU阈值':^10}{'准确率':^10}{'召回率':^10}{'F1值':^10}{'TP':^8}{'FP':^8}{'FN':^8}{'预测数':^10}{'真实数':^10}{'TP+FP+FN':^10}\n")
            analysis_file.write("-" * 100 + "\n")
            
            # 遍历该置信度阈值下的所有IoU阈值
            for iou_thresh in sorted(results[conf_thresh].keys()):
                metrics = results[conf_thresh][iou_thresh]
                tp = metrics['true_positives']
                fp = metrics['false_positives']
                fn = metrics['false_negatives']
                total = tp + fp + fn
                
                # 打印到控制台
                print(f"{conf_thresh:^10}{iou_thresh:^10}{metrics['precision']:.4f}  {metrics['recall']:.4f}  {metrics['f1_score']:.4f}   {tp:^8}{fp:^8}{fn:^8}{metrics['total_predictions']:^10}{metrics['total_ground_truth']:^10}{total:^10}")
                
                # 写入分析文件
                analysis_file.write(f"{iou_thresh:^10}{metrics['precision']:.4f}  {metrics['recall']:.4f}  {metrics['f1_score']:.4f}   {tp:^8}{fp:^8}{fn:^8}{metrics['total_predictions']:^10}{metrics['total_ground_truth']:^10}{total:^10}\n")
            
            # 在每个置信度阈值后添加分隔线
            print("-" * 120)
            analysis_file.write("\n" + "=" * 100 + "\n\n")
        
        # 寻找最佳F1值对应的阈值组合
        best_f1 = 0
        best_conf = ""
        best_iou = ""
        
        for conf in results:
            for iou in results[conf]:
                f1 = results[conf][iou]["f1_score"]
                if f1 > best_f1:
                    best_f1 = f1
                    best_conf = conf
                    best_iou = iou
        
        # 添加总结信息
        analysis_file.write("\n总结信息:\n")
        analysis_file.write("-" * 50 + "\n")
        analysis_file.write(f"最佳F1值: {best_f1:.4f}\n")
        analysis_file.write(f"对应置信度阈值: {best_conf}\n")
        analysis_file.write(f"对应IoU阈值: {best_iou}\n")
        best_metrics = results[best_conf][best_iou]
        analysis_file.write(f"TP: {best_metrics['true_positives']}, FP: {best_metrics['false_positives']}, FN: {best_metrics['false_negatives']}\n")
        analysis_file.write(f"准确率: {best_metrics['precision']:.4f}, 召回率: {best_metrics['recall']:.4f}\n")
        
        # 新增：寻找准确率最高的情况下召回率最高的组合
        best_precision = 0
        best_recall_under_best_precision = 0
        best_precision_conf = ""
        best_precision_iou = ""
        
        # 首先找到最高的准确率
        for conf in results:
            for iou in results[conf]:
                precision = results[conf][iou]["precision"]
                if precision > best_precision:
                    best_precision = precision
        
        # 在最高准确率的情况下找到最高的召回率
        for conf in results:
            for iou in results[conf]:
                precision = results[conf][iou]["precision"]
                recall = results[conf][iou]["recall"]
                # 只考虑准确率达到最高值的组合
                if abs(precision - best_precision) < 1e-6:  # 使用浮点数比较
                    if recall > best_recall_under_best_precision:
                        best_recall_under_best_precision = recall
                        best_precision_conf = conf
                        best_precision_iou = iou
        
        # 添加新的总结信息
        analysis_file.write("\n" + "=" * 50 + "\n")
        analysis_file.write("准确率优先策略结果:\n")
        analysis_file.write("-" * 50 + "\n")
        analysis_file.write(f"最高准确率: {best_precision:.4f}\n")
        analysis_file.write(f"在最高准确率下的最高召回率: {best_recall_under_best_precision:.4f}\n")
        analysis_file.write(f"对应置信度阈值: {best_precision_conf}\n")
        analysis_file.write(f"对应IoU阈值: {best_precision_iou}\n")
        
        if best_precision_conf and best_precision_iou:
            precision_metrics = results[best_precision_conf][best_precision_iou]
            analysis_file.write(f"TP: {precision_metrics['true_positives']}, FP: {precision_metrics['false_positives']}, FN: {precision_metrics['false_negatives']}\n")
            
            # 计算对应的F1值
            p = precision_metrics['precision']
            r = precision_metrics['recall']
            f1_precision_priority = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
            analysis_file.write(f"对应F1值: {f1_precision_priority:.4f}\n")
    
    # 打印最佳结果
    print(f"\n最佳性能组合:")
    print(f"置信度阈值 = {best_conf}, IoU阈值 = {best_iou}, F1值 = {best_f1:.4f}")
    print(f"TP = {best_metrics['true_positives']}, FP = {best_metrics['false_positives']}, FN = {best_metrics['false_negatives']}")
    print(f"准确率 = {best_metrics['precision']:.4f}, 召回率 = {best_metrics['recall']:.4f}")
    
    # 打印准确率优先策略结果
    print(f"\n准确率优先策略结果:")
    print(f"最高准确率 = {best_precision:.4f}")
    print(f"在最高准确率下的最高召回率 = {best_recall_under_best_precision:.4f}")
    print(f"对应置信度阈值 = {best_precision_conf}, IoU阈值 = {best_precision_iou}")
    if best_precision_conf and best_precision_iou:
        precision_metrics = results[best_precision_conf][best_precision_iou]
        print(f"TP = {precision_metrics['true_positives']}, FP = {precision_metrics['false_positives']}, FN = {precision_metrics['false_negatives']}")
        
        # 计算对应的F1值
        p = precision_metrics['precision']
        r = precision_metrics['recall']
        f1_precision_priority = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        print(f"对应F1值 = {f1_precision_priority:.4f}")

if __name__ == "__main__":

    dxf_path_gt = './阴影和真值.dxf'
    output_path_gt = './阴影和真值.json'
    selected_layer_gt = "开孔识别结果"
    abandon_layer_gt = "开孔识别结果1"

    dxf_path_pred = './阴影和真值.dxf'
    output_path_pred = './阴影和真值.json'
    selected_layer_pred = "Holes"

    
    converter_gt = DXFConverterV2(selected_layer_gt)
    bboxes_gt, bboxes_hatch = converter_gt.convert_file(dxf_path_gt, output_path_gt)
    
    if len(abandon_layer_gt) > 0:
        converter_ab = DXFConverterV2(abandon_layer_gt)
        bboxes_ab, _ = converter_ab.convert_file(dxf_path_gt, output_path_gt)
    else:
        bboxes_ab = None

    converter_pred = DXFConverterV2(selected_layer_pred)
    bboxes_pred, _ = converter_pred.convert_file(dxf_path_pred, output_path_pred)

    print("gt boxes number = ", len(bboxes_gt))
    print("pred boxes number = ", len(bboxes_pred))
    print("ab boxes number = ", len(bboxes_ab))
    print("hatch boxes number = ", len(bboxes_hatch))
    # 在evaluate之前对预测框进行NMS处理
    converted_pred = convert(bboxes_pred)
    if converted_pred is not None:
        # 使用yolo_test中的nms函数，IoU阈值设为0.5
        nms_pred = nms(converted_pred, iou_threshold=0.5)
        print(f"NMS前预测框数量: {len(converted_pred)}")
        print(f"NMS后预测框数量: {len(nms_pred)}")
    else:
        nms_pred = None
        print("预测框为空，跳过NMS处理")
    
    # exit()
    # bboxes_pred = [[265061.9118044584,-287044.62146601186,265766.6937838789,-287778.4531252597,0.9316078424453735]]
    # bboxes_gt = [[265061.9118044584,-287044.62146601186,265766.6937838789,-287778.4531252597,0.9316078424453735]]
    # bboxes_ab = None
    # bboxes_hatch = []
    results = evaluate(nms_pred, convert(bboxes_gt), convert(bboxes_ab), convert(bboxes_hatch))
    
    # 计算F1值
    p, r = results["P"], results["R"]
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
    
    print(f"评估结果: {results}")
    print(f"总结: ")
    print(f"  真实目标数量: {results['total_gt']}")
    print(f"  预测目标数量: {results['total_predictions']}")
    print(f"  正确检测数量 (真阳性): {results['TP']}")
    print(f"  错误检测数量 (假阳性): {results['FP']}")
    print(f"  漏检数量 (假阴性): {results['FN']}")
    print(f"  准确率: {results['P']:.4f} = {results['TP']}/{results['total_predictions'] if results['total_predictions'] > 0 else 1}")
    print(f"  召回率: {results['R']:.4f} = {results['TP']}/{results['total_gt']}")
    print(f"  F1值: {f1:.4f} = 2 * ({p:.4f} * {r:.4f}) / ({p:.4f} + {r:.4f})")
    
    # 分析所有置信度阈值的性能
    analyze_confidence_thresholds("eval_results_all.json")
