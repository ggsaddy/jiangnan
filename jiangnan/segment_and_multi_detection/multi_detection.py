import argparse
import os
from segment_and_multi_detection.load import dxf2json
from segment_and_multi_detection.split.main import segment_v4, multi_detect
import ezdxf
import logging
import json
import multiprocessing
from glob import glob
import sys

def draw_rectangle_in_dxf(file_path, folder, file_name, bbox_lists):
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()
    
    if "SECTION DRAWING" not in doc.layers:
        doc.layers.add("SECTION DRAWING", color=5)
    else:
        entity_to_delete = [e for e in msp if e.dxf.layer == "SECTION DRAWING" or e.dxf.layer == "SECTION DRAWING"]
        
        for e in entity_to_delete:
            msp.delete_entity(e)
            
    
    line_width = 100
    extend_threshold = 100
    
    correct = 0
    wrong = 0
    for item in bbox_lists:
        bbox = item[0]
        success = item[1]
        
        x1 = bbox["x1"] - extend_threshold
        y1 = bbox["y1"] - extend_threshold
        x2 = bbox["x2"] + extend_threshold
        y2 = bbox["y2"] + extend_threshold
        
        if success == 1 or success == 3: #正确的剖面符号和子图类型
            if success == 1:
                correct += 1
            msp.add_lwpolyline([
                [x1, y1, line_width, line_width],
                [x1, y2, line_width, line_width],
                [x2, y2, line_width, line_width],
                [x2, y1, line_width, line_width]],
                close=True,
                dxfattribs={
                    "layer": "SECTION DRAWING" #蓝色
                }
            )
        elif success == 0: #错误的剖面符号
            wrong += 1
            x1 = bbox["x1"] - extend_threshold 
            y1 = bbox["y1"] - extend_threshold
            x2 = bbox["x2"] + extend_threshold
            y2 = bbox["y2"] + extend_threshold
            
            msp.add_lwpolyline([
            [x1, y1, line_width, line_width],
            [x1, y2, line_width, line_width],
            [x2, y2, line_width, line_width],
            [x2, y1, line_width, line_width]],
            close=True,
            dxfattribs={
                "layer": "SECTION DRAWING",
                "color": 256,
                "true_color": ezdxf.colors.rgb2int((255, 0, 0)) #红色
            })
        elif success == 2: #未匹配上的剖面符号
            wrong += 1
            x1 = bbox["x1"] - extend_threshold 
            y1 = bbox["y1"] - extend_threshold
            x2 = bbox["x2"] + extend_threshold
            y2 = bbox["y2"] + extend_threshold
            
            msp.add_lwpolyline([
            [x1, y1, line_width, line_width],
            [x1, y2, line_width, line_width],
            [x2, y2, line_width, line_width],
            [x2, y1, line_width, line_width]],
            close=True,
            dxfattribs={
                "layer": "SECTION DRAWING",
                "color": 256,
                "true_color": ezdxf.colors.rgb2int((0, 255, 255)) #青色
            })
        elif success == 4: #从剖面符号指示子图的箭头和剖面
            x1 = bbox["x1"] 
            y1 = bbox["y1"]
            x2 = bbox["x2"]
            y2 = bbox["y2"]
            
            msp.add_lwpolyline([
            [x1, y1, 30, 30],
            [x2, y2, 30, 30]],
            close=True,
            dxfattribs={
                "layer": "SECTION DRAWING",
                "color": 256,
                "true_color": ezdxf.colors.rgb2int((255, 0, 0))
            }) #红色
            
        elif success == 5: #剖面
            x1 = bbox["x1"] 
            y1 = bbox["y1"]
            x2 = bbox["x2"]
            y2 = bbox["y2"]
            
            msp.add_lwpolyline([
            [x1, y1, 30, 30],
            [x2, y2, 30, 30]],
            close=True,
            dxfattribs={
                "layer": "SECTION DRAWING",
                "color": 256,
                "true_color": ezdxf.colors.rgb2int((255, 192, 203)) #粉色
            }) #红色

        elif success == 7:
            x1 = bbox["x1"] - extend_threshold 
            y1 = bbox["y1"] - extend_threshold
            x2 = bbox["x2"] + extend_threshold
            y2 = bbox["y2"] + extend_threshold
            
            msp.add_lwpolyline([
            [x1, y1, line_width, line_width],
            [x1, y2, line_width, line_width],
            [x2, y2, line_width, line_width],
            [x2, y1, line_width, line_width]],
            close=True,
            dxfattribs={
                "layer": "SECTION DRAWING",
                "color": 256,
                "true_color": ezdxf.colors.rgb2int((128, 0, 128))  # 紫色
                #紫色
            })
        elif success == 8:
            x1 = bbox["x1"] - extend_threshold 
            y1 = bbox["y1"] - extend_threshold
            x2 = bbox["x2"] + extend_threshold
            y2 = bbox["y2"] + extend_threshold
            
            msp.add_lwpolyline([
            [x1, y1, line_width, line_width],
            [x1, y2, line_width, line_width],
            [x2, y2, line_width, line_width],
            [x2, y1, line_width, line_width]],
            close=True,
            dxfattribs={
                "layer": "SECTION DRAWING",
                "color": 256,
                "true_color": ezdxf.colors.rgb2int(( 255, 165, 0))  # 橙色

                #橙色
            })
        elif success == 9:
            x1 = bbox["x1"] - extend_threshold 
            y1 = bbox["y1"] - extend_threshold
            x2 = bbox["x2"] + extend_threshold
            y2 = bbox["y2"] + extend_threshold
            
            msp.add_lwpolyline([
            [x1, y1, line_width, line_width],
            [x1, y2, line_width, line_width],
            [x2, y2, line_width, line_width],
            [x2, y1, line_width, line_width]],
            close=True,
            dxfattribs={
                "layer": "SECTION DRAWING",
                "color": 256,
                "true_color": ezdxf.colors.rgb2int((0, 128, 0))  # 绿色
                #绿色
            })
        elif success == 10:
            x1 = bbox["x1"] - extend_threshold 
            y1 = bbox["y1"] - extend_threshold
            x2 = bbox["x2"] + extend_threshold
            y2 = bbox["y2"] + extend_threshold
            
            msp.add_lwpolyline([
            [x1, y1, line_width, line_width],
            [x1, y2, line_width, line_width],
            [x2, y2, line_width, line_width],
            [x2, y1, line_width, line_width]],
            close=True,
            dxfattribs={
                "layer": "SECTION DRAWING",
                "color": 256,
                "true_color": ezdxf.colors.rgb2int((255, 215, 0))  # 金色
                #金色
            })
        elif success == 11:
            x1 = bbox["x1"] - extend_threshold 
            y1 = bbox["y1"] - extend_threshold
            x2 = bbox["x2"] + extend_threshold
            y2 = bbox["y2"] + extend_threshold
            
            msp.add_lwpolyline([
            [x1, y1, line_width, line_width],
            [x1, y2, line_width, line_width],
            [x2, y2, line_width, line_width],
            [x2, y1, line_width, line_width]],
            close=True,
            dxfattribs={
                "layer": "SECTION DRAWING",
                "color": 256,
                "true_color": ezdxf.colors.rgb2int((255,215, 0))  
            })
        elif success == 12: #相似剖面
            x1 = bbox["x1"] 
            y1 = bbox["y1"]
            x2 = bbox["x2"]
            y2 = bbox["y2"]
            
            msp.add_lwpolyline([
            [x1, y1, 30, 30],
            [x2, y2, 30, 30]],
            close=True,
            dxfattribs={
                "layer": "SECTION DRAWING",
                "color": 256,
                # 橙色
                "true_color": ezdxf.colors.rgb2int(( 255, 165, 0))
            })
    doc.saveas(os.path.join(folder, "{}_SECTION_DRAWING.dxf".format(file_name)))
    return correct, wrong, file_name


def multi_detection_main(input_file, input_folder, output_folder):
    correct_ratio={}
    input_file = os.path.abspath(os.path.normpath(input_file))
    element_list = input_file.split(os.sep)
    input_folder = element_list[0] + "/" + os.path.join(*element_list[1:-1])
    file_name = os.path.basename(input_file)[:-4]
    print("Processing file: {}".format(file_name))
    json_file = os.path.join(output_folder, file_name + ".json")
    img_file = os.path.join(output_folder, file_name + ".jpg")
    split_file = os.path.join(output_folder, file_name + "_split.dxf")
    
    dxf2json(input_folder, file_name, output_folder)
    print("Split start.....")
    json_result, bbox_result, _, false_count = multi_detect(json_file,img_file)
    json_name = os.path.join(output_folder, file_name) + "_multi.json"
    with open(json_name, 'w', encoding='utf-8') as f:
        for res in json_result:
            f.write(json.dumps(res, ensure_ascii=False, indent=4))
        print("writing success")
    correct, wrong, file_name = draw_rectangle_in_dxf(input_file, output_folder, file_name, bbox_result)
    correct_ratio[file_name+"正确数"]=correct
    correct_ratio[file_name+"错误数"]=wrong-false_count
    correct_ratio[file_name+"正确率"]=1 if(correct+wrong-false_count)==0 else (correct/(correct+wrong-false_count))
    
    #NEWADD0807
    # json_file=json.dumps(correct_ratio, ensure_ascii=False, indent=4)
    # correct_ratio_name = os.path.join(output_folder, file_name) + "_剖面符号规则统计.json"
    # with open(correct_ratio_name, 'w', encoding='utf-8') as f:
    #     f.write(json_file)
    #NEWADD0807
    json_merged_name = os.path.join(output_folder, file_name) + "_multi_merged.json"
    with open(json_merged_name, 'w', encoding='utf-8') as f:
        for res in json_result:
            for i in range(len(res) - 1, -1, -1):
                item = res[i]
                if isinstance(item, dict):
                    if "相似场景" in item.keys() or "子图调用次数" in item.keys():
                        res.pop(i)  # 使用pop(index)删除指定位置的元素
            f.write(json.dumps(res, ensure_ascii=False, indent=4))
    multi_detection_result_list= []
    json_name = os.path.join(output_folder, file_name) + "_multi.json"
    with open(json_name, 'r', encoding='utf-8') as f:
        multi_detection_result_list = json.load(f)
    return multi_detection_result_list

def main(input_folder, output_folder):
    input_files = glob(os.path.join(input_folder, "*.dxf"))
    output_folder = os.path.abspath(os.path.normpath(output_folder))
    os.makedirs(output_folder, exist_ok=True)
    for input_file in input_files:
        multi_detection_result_list=multi_detection_main(input_file, input_folder, output_folder)
        print(f"Processed {input_file}, result: {multi_detection_result_list}")
    print("All files processed successfully.")
if __name__ == "__main__":
    multiprocessing.freeze_support()
    logging.basicConfig(filename="error.txt", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
    input_folder = "C:/Users/31285/Desktop/曲子睿的文件夹/复旦/CAD分割/DXFStruct/output_check"
    output_folder = "C:/Users/31285/Desktop/曲子睿的文件夹/复旦/CAD分割/DXFStruct/output_check/segment_0804"
    main(input_folder, output_folder)
    