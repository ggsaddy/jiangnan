'''
中文回答，对于函数def predict_image，我现在需要多尺度tta推理，基础分辨率还是1024，但是总体推理分辨率为768，896，1024，1152，1280五个分辨率，请给出新的推理接口predict_image_tta,并且要求还是以1024为基础！这点要注意
'''
from ultralytics import YOLO
import cv2
import numpy as np
import json 
from glob import glob 
from tqdm import tqdm 
import os 


def predict_image_tta(
    model,
    image_path: str,
    conf_threshold: float = 0.25,
    base_imgsz: int = 1024,
    scales: list = None,
    verbose: bool = False,
    save: bool = True,
) -> list:
    """
    使用多尺度TTA(Test Time Augmentation)进行目标检测推理
    
    参数:
        model: YOLO模型实例
        image_path: 待检测图片路径
        conf_threshold: 置信度阈值，默认0.25
        base_imgsz: 基础分辨率大小，默认1024
        scales: 相对于base_imgsz的缩放比例列表，如果为None则使用默认值[0.75, 0.875, 1.0, 1.125, 1.25]
        verbose: 是否打印详细信息
        save: 是否保存推理结果
        
    返回:
        所有尺度预测框的集合，每个预测框格式为[x1, y1, x2, y2, conf, class_id]

    Usage:
    predictions = predict_image_tta(
        model=model,
        image_path="test.png",
        conf_threshold=0.25
    )
    """
    # 如果未指定scales，使用默认的多尺度设置
    if scales is None:
        scales = [0.5, 0.75, 1.0, 1.5, 2]  # 对应768, 896, 1024, 1152, 1280
    
    all_predictions = []
    
    # 对每个尺度进行推理
    for scale in scales:
        current_imgsz = int(base_imgsz * scale)
        if verbose:
            print(f"Processing scale {scale:.3f}, image size: {current_imgsz}")
            
        # 使用当前尺度进行推理
        results = model.predict(
            source=image_path,
            conf=conf_threshold,
            imgsz=current_imgsz,
            verbose=verbose,
            save=save,
            augment=False,  # 不在单次推理中使用augment，因为我们已经在做多尺度TTA
        )
        
        # 收集当前尺度的预测结果
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # 获取预测框坐标
                    bbox = box.xyxy[0].cpu().numpy()
                    # 获取置信度
                    conf = float(box.conf[0])
                    # 获取类别ID
                    class_id = int(box.cls[0])
                    
                    # 将预测结果添加到集合中
                    pred = [*bbox, conf, class_id]
                    all_predictions.append(pred)
    
    # 对所有尺度的预测框进行NMS处理
    final_predictions = nms(all_predictions, iou_threshold=0.5)
    
    return final_predictions

def predict_image(
    model,
    image_path: str,
    conf_threshold: float = 0.25,
    imgsz: int = 1024,
    verbose: bool = False,
    save: bool = True,
) -> list:
    """
    使用YOLO模型对单张图片进行目标检测
    
    参数:
        model_path: YOLO模型权重文件路径
        image_path: 待检测图片路径
        conf_threshold: 置信度阈值,默认0.25
        imgsz: 输入图片大小,默认1024
        
    返回:
        预测框列表,每个预测框格式为[x1, y1, x2, y2, conf, class_id]
    """

    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        imgsz=imgsz,
        verbose=verbose,
        save=save,
        augment=True,
    )


    predictions = []
    for result in results:

        if result.boxes is not None:
            for box in result.boxes:

                bbox = box.xyxy[0].cpu().numpy()

                conf = float(box.conf[0])

                class_id = int(box.cls[0])
                
                pred = [*bbox, conf, class_id]
                predictions.append(pred)
    
    return predictions


def visualize_predictions(image_path: str, predictions: list, save_path: str = None):
    image = cv2.imread(image_path)
    
    for pred in predictions:
        x1, y1, x2, y2, conf, class_id = pred
        
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        text = f"conf: {conf:.2f}"
        cv2.putText(image, text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if save_path:
        cv2.imwrite(save_path, image)
    else:
        cv2.imshow("Predictions", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def convert_png2dxf_coord(png_x, png_y, patch_x, patch_y, metadata):
    scale = metadata['scale']
    offset_x = metadata['offset_x']
    offset_y = metadata['offset_y']
    overlap = metadata['overlap']
    padding = metadata['padding']
    patch_size = metadata['patch_size']
    stride = int(patch_size * (1 - overlap))
    
    global_x = patch_x + png_x
    global_y = patch_y + png_y
    

    dxf_x = (global_x  - offset_x) / scale
    dxf_y = (offset_y - global_y ) / scale
    
    return dxf_x, dxf_y

def nms(bboxes: list, iou_threshold: float = 0.9) -> list:
    # 提取坐标和置信度
    x1 = [box[0] for box in bboxes]
    y1 = [box[1] for box in bboxes]
    x2 = [box[2] for box in bboxes]
    y2 = [box[3] for box in bboxes]
    conf = [box[4] for box in bboxes]
    
    # 计算每个边界框的面积
    areas = [(x2[i] - x1[i]) * (y2[i] - y1[i]) for i in range(len(bboxes))]
    
    # 按置信度降序排序，获取索引
    indices = sorted(range(len(bboxes)), key=lambda i: conf[i], reverse=True)
    
    keep = []
    remove = set()
    
    for i in range(len(indices)):
        if indices[i] in remove:
            continue
        keep.append(indices[i])
        # 计算当前框与后面的框的IoU
        for j in range(i+1, len(indices)):
            idx_j = indices[j]
            if idx_j in remove:
                continue
            # 计算交集坐标
            inter_x1 = max(x1[indices[i]], x1[idx_j])
            inter_y1 = max(y1[indices[i]], y1[idx_j])
            inter_x2 = min(x2[indices[i]], x2[idx_j])
            inter_y2 = min(y2[indices[i]], y2[idx_j])
            
            # 计算交集面积
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            else:
                inter_area = 0.0
            
            # 计算并集面积
            union_area = areas[indices[i]] + areas[idx_j] - inter_area
            
            # 计算IoU
            iou = inter_area / union_area if union_area > 0 else 0.0
            
            # 如果IoU超过阈值，标记该框待删除
            if iou > iou_threshold:
                remove.add(idx_j)
    
    # 返回保留的边界框
    return [bboxes[i] for i in keep]

if __name__ == "__main__":
    
    model_path = "/Users/ieellee/Downloads/codebase/ship/best.pt"
    model = YOLO(model_path)
    
    dxf_bboxes = []
    for image_path in tqdm(glob("/Users/ieellee/Downloads/codebase/ship/holes_demo_sliding_61604-48772-158131-104492/*.png"), "Inferring:..."):
        json_path = os.path.join(os.path.dirname(image_path), "meta_data.json")
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        try:
            string = os.path.basename(image_path).split(".")[0].split("_")
            patch_x, patch_y = int(string[1]), int(string[2])
            predictions = predict_image(
                model=model,
                image_path=image_path,
                conf_threshold=0.7,
                imgsz=1024,
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
                print(f"- dxf边界框坐标: ({dxf_x1:.2f}, {dxf_y1:.2f}, {dxf_x2:.2f}, {dxf_y2:.2f})")
                print(f"- 置信度: {conf:.3f}")
                print(f"- 类别ID: {class_id}")
                width, height = dxf_x2 - dxf_x1, dxf_y1 - dxf_y2
                if height > width:
                    width, height = height, width
                if width / height > 1.5:
                    print("Filter bbox")
                    continue
                dxf_bboxes.append([dxf_x1, dxf_y1, dxf_x2, dxf_y2, conf])
        except:
            print("Meeting whole.png")
    nms_dxf_bboxes = nms(dxf_bboxes, 0.1)
    print(f"Before nms bboxes number = {len(dxf_bboxes)}")
    print(f"After nms bboxes number = {len(nms_dxf_bboxes)}")
    with open("nms_results.txt", "w") as f:
        for bbox in nms_dxf_bboxes:
            x1, y1, x2, y2, conf = bbox 
            f.write(f"{x1},{y1},{x2},{y2},{conf}\n")
            
    # bboxes = []
    # with open("nms_results.txt", "r") as f:
    #     for line in f.readlines():
    #         x1, y1, x2, y2, conf = [eval(i) for i in line.strip().split(",")] 
    #         bboxes.append([x1, y1, x2, y2, conf])
    #     nms_bboxes = nms(bboxes, 0.1)
    #     print(len(nms_bboxes))


'''
!tar xfvz archive.tar.gz
!pip install --no-index --find-links=./packages ultralytics
!rm -rf ./packages 
'''