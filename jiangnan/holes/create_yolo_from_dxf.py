'''
我现在需要做的事情是，给定一张dxf图纸，以及一个layername输入：
1. 使用load_v2.py将bbox标注信息提取出来，这些bbox都代表同一个类别。
2. 参考convert2png_v2.py，这个函数的功能是可以将一个dxf按照滑动窗口逻辑转化为分辨率为patch_size的图片。可以选择使用auto_size自动统计max_siz和min_size，也可以自己指定特定的参数。因为第一步已经得到了标注bbox，第二步我需要在滑动窗口制作patch图片的同时制作包含标注的yolo训练数据集，这个过程中有几点需要注意：
    2.1. 如果目标bbox出现在图像边缘，比如bbox不完整这种情况，如果bbox长或宽遮挡比例超过25%则舍弃这个bbox。
    2.2. 在2.1的前提下，只保留包含bbox的patch，并保存为图片。
    2.3. 对于不包含bbox的图片，20%的概率保存为纯背景进行训练。
    2.4. 整理完所有数据后，按照80%训练，20%测试进行划分数据集，整理为yolo的格式：
        ship_datasets
            -images
                -train
                -val
            -labels
                -train
                -val
            ship_conf.yaml
        其中ship_conf.yaml的内容为：
            path: ./ship_datasets
            train: images/train # train images (relative to 'path') 
            val: images/val # val images (relative to 'path') 

            # Classes
            names:
            0: holes
'''


import os
import json
import yaml 
import glob 
import random
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm
from load import dxf2json
from load_v2 import DXFConverterV2
from convert2png_v2 import DXFRenderer, convert_png2dxf_coord

class DXF2YOLO:
    def __init__(self, patch_size=1024, overlap=0.5, bbox_overlap_threshold=0.25):
        """
        初始化转换器
        Args:
            patch_size: 滑动窗口大小
            overlap: 滑动窗口重叠率
            bbox_overlap_threshold: bbox被截断的阈值，超过此比例则舍弃
        """
        self.patch_size = patch_size
        self.overlap = overlap
        self.bbox_overlap_threshold = bbox_overlap_threshold
        self.renderer = DXFRenderer(
            max_size=1024*6,
            min_size=1024,
            padding_ratio=0.05,
            patch_size=patch_size,
            overlap=overlap,
            auto_size=True
        )

    def convert_to_yolo_format(self, bbox, img_width, img_height):
        """
        将bbox转换为YOLO格式 [class_id, x_center, y_center, width, height]
        所有值归一化到[0,1]区间
        """
        x1, y1, x2, y2 = bbox[:4]
        
        # 计算中心点和宽高
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        # 归一化
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        
        return [0, x_center, y_center, width, height]  # 0 表示holes类别


    def convert_dxf2png_coord(self, dxf_x, dxf_y, metadata):
        """
        将DXF坐标转换为PNG图像坐标
        """
        scale = metadata['scale']
        offset_x = metadata['offset_x']
        offset_y = metadata['offset_y']
        padding = metadata['padding']
        
        # 转换为全局PNG坐标
        png_x = dxf_x * scale + offset_x 
        png_y = offset_y - dxf_y * scale
        
        return png_x, png_y

    def check_bbox_valid(self, bbox, patch_coords, metadata):
        """
        检查bbox在当前patch中是否有效，并进行坐标转换
        """
        px, py, px2, py2 = patch_coords
        
        # 将DXF坐标转换为PNG坐标
        x1, y1 = self.convert_dxf2png_coord(bbox[0], bbox[1], metadata)
        x2, y2 = self.convert_dxf2png_coord(bbox[2], bbox[3], metadata)
        
        # 确保坐标顺序正确
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # 检查bbox是否与patch有交集
        if x2 < px or x1 > px2 or y2 < py or y1 > py2:
            return None
            
        # 计算交集区域
        inter_x1 = max(x1, px)
        inter_y1 = max(y1, py)
        inter_x2 = min(x2, px2)
        inter_y2 = min(y2, py2)
        
        # 计算原始bbox的宽高
        orig_width = x2 - x1
        orig_height = y2 - y1
        
        # 计算交集区域的宽高
        inter_width = inter_x2 - inter_x1
        inter_height = inter_y2 - inter_y1
        
        # 计算被截断的比例
        width_ratio = inter_width / orig_width
        height_ratio = inter_height / orig_height
        
        # 如果任一维度被截断超过阈值，则舍弃
        if width_ratio < (1 - self.bbox_overlap_threshold) or height_ratio < (1 - self.bbox_overlap_threshold):
            return None
            
        # 返回相对于patch的坐标
        return [
            inter_x1 - px,
            inter_y1 - py,
            inter_x2 - px,
            inter_y2 - py,
            bbox[4] if len(bbox) > 4 else 1.0
        ]

    def process_dxf(self, dxf_path, selected_layer, output_dir):
        """
        处理单个DXF文件
        """
        # 创建DXF转换器并获取bbox
        converter = DXFConverterV2(selected_layer)
        dxf2json(os.path.dirname(dxf_path), os.path.basename(dxf_path), os.path.dirname(dxf_path))
        bboxes = converter.convert_file(dxf_path, dxf_path.replace('.dxf', '') + "_bbox.json")
        
        if not bboxes:
            print(f"No bboxes found in {dxf_path}")
            return
        
        # 使用renderer处理DXF文件
        base_max_size, base_min_size = self.renderer.render(dxf_path.replace('.dxf', '.json'), output_dir) # 先渲染一个第一版数据
        # int(max_size * 1.5), int(min_size * 1.5)
        # int(max_size * 2.0), int(min_size * 2.0)
        # self.renderer = DXFRenderer(
        #     max_size=max_size,
        #     min_size=min_size,
        #     padding_ratio=0.05,
        #     patch_size=1024,
        #     overlap=0.5,
        #     auto_size=False
        # )
        # scale_factors = [1.5, 2.0]  # 可以根据需要调整缩放比例
        # for scale in scale_factors:
        #     self.renderer = DXFRenderer(
        #         max_size=int(base_max_size * scale),
        #         min_size=int(base_min_size * scale),
        #         padding_ratio=0.05,
        #         patch_size=1024,
        #         overlap=0.5,
        #         auto_size=False
        #     )
        #     self.renderer.render(dxf_path.replace('.dxf', '.json'), output_dir)

        
        # 获取metadata
        meta_path = os.path.join("sliding", 
                               f"{os.path.basename(dxf_path).split('.')[0]}_sliding_*", 
                               "meta_data.json")
        meta_files = glob.glob(meta_path)
        if not meta_files:
            print(f"No metadata found for {dxf_path}")
            return
        for i in range(len(meta_files)):

            with open(meta_files[i], 'r') as f:
                metadata = json.load(f)
        
            # 计算滑动窗口参数
            stride = int(self.patch_size * (1 - self.overlap))
            
            # 处理每个patch
            patches_dir = os.path.dirname(meta_files[i])
            valid_patches = []
            
            for img_path in glob.glob(os.path.join(patches_dir, "*_patch_*.png")):
                if "whole" in img_path:
                    continue
                    
                # 解析patch坐标
                coords = img_path.split('_patch_')[-1].split('.')[0]
                px, py = map(int, coords.split('_'))
                
                # 检查每个bbox在当前patch中的有效性
                valid_bboxes = []
                patch_coords = [px, py, px + self.patch_size, py + self.patch_size]
                
                for bbox in bboxes:
                    adjusted_bbox = self.check_bbox_valid(bbox, patch_coords, metadata)
                    if adjusted_bbox is not None:
                        valid_bboxes.append(adjusted_bbox)
                
                # 如果patch中有有效的bbox或随机保留一些背景patch
                if valid_bboxes or (random.random() < 0.2):
                    valid_patches.append({
                        'image_path': img_path,
                        'bboxes': valid_bboxes
                    })
        
        return valid_patches

    def create_yolo_dataset(self, dxf_dir, selected_layer, output_dir):
        """
        创建YOLO格式的数据集
        """
        # 创建数据集目录结构
        dataset_dir = Path(output_dir) / "ship_datasets"
        for split in ['train', 'val']:
            (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # 处理所有DXF文件
        all_patches = []
        for dxf_file in glob.glob(os.path.join(dxf_dir, "*.dxf")):
            patches = self.process_dxf(dxf_file, selected_layer, output_dir)
            if patches:
                all_patches.extend(patches)
        
        # 随机划分训练集和验证集
        print("all patches length = ", len(all_patches))
        random.shuffle(all_patches)
        split_idx = int(len(all_patches) * 0.8)
        train_patches = all_patches[:split_idx]
        val_patches = all_patches[split_idx:]
        
        # 处理训练集和验证集
        for split, patches in [('train', train_patches), ('val', val_patches)]:
            for idx, patch_data in enumerate(patches):
                # 复制图片
                img_name = f"{split}_{idx}.png"
                shutil.copy(
                    patch_data['image_path'],
                    dataset_dir / 'images' / split / img_name
                )
                
                # 创建标签文件
                label_path = dataset_dir / 'labels' / split / f"{split}_{idx}.txt"
                with open(label_path, 'w') as f:
                    for bbox in patch_data['bboxes']:
                        yolo_bbox = self.convert_to_yolo_format(bbox, self.patch_size, self.patch_size)
                        f.write(' '.join(map(str, yolo_bbox)) + '\n')
        
        # 创建配置文件
        yaml_content = {
            'path': str(dataset_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': {
                0: 'holes'
            }
        }
        
        with open(dataset_dir / 'ship_conf.yaml', 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert DXF files to YOLO dataset')
    parser.add_argument('--dxf_dir', type=str, default="dxfs", help='Directory containing DXF files')
    parser.add_argument('--layer', type=str, default="非标开孔", help='Layer name to process')
    parser.add_argument('--output_dir', type=str, default="", help='Output directory')
    parser.add_argument('--patch_size', type=int, default=1024, help='Patch size')
    parser.add_argument('--overlap', type=float, default=0.5, help='Overlap ratio')
    args = parser.parse_args()
    
    converter = DXF2YOLO(
        patch_size=args.patch_size,
        overlap=args.overlap
    )
    converter.create_yolo_dataset(args.dxf_dir, args.layer, args.output_dir)

if __name__ == '__main__':
    main()



'''
yolo detect train model=yolo11n.pt data=ship_datasets/ship_conf.yaml epochs=10 batch=8 imgsz=1024 device=cpu name=v1 project=YOLO11-n_ship warmup_epochs=1 optimizer=AdamW cos_lr=True flipud=0.5 fliplr=0.5 degrees=45 mixup=0 copy_paste=0 save=True save_period=1 plots=True exist_ok=True verbose=True cache=True multi_scale=True
'''