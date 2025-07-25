import argparse
import os

import ezdxf.colors
from load import dxf2json
from split.main import segment_v4, multi_detect_local
import ezdxf
import logging
import json
import multiprocessing
from glob import glob

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
        #NEWADD
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
            #NEWADD
    doc.saveas(os.path.join(folder, "{}_SECTION_DRAWING.dxf".format(file_name)))

def multi_detection_main(input_file, input_folder, output_folder):
    file_name = os.path.basename(input_file)[:-4]
    print("Processing file: {}".format(file_name))
    input_dxf_file = os.path.join(input_folder, file_name + ".dxf")
    dxf2json(input_folder, file_name, output_folder)
    input_json_file = os.path.join(output_folder, file_name + ".json")
    
    json_result, bbox_result, _, _ = multi_detect_local(input_json_file)
    
    draw_rectangle_in_dxf(input_dxf_file, output_folder, file_name, bbox_result)
    
    json_name = os.path.join(output_folder, file_name) + "_multi.json"

    with open(json_name, 'w', encoding='utf-8') as f:
        for res in json_result:
            f.write(json.dumps(res, ensure_ascii=False, indent=4))
        print("writing success")
    
def main():
    #input_json_file = "/disk1/user4/work/造船厂/结构AI/qzr/output/多级剖图20250424.json"
    input_json_file = "/disk1/user4/work/造船厂/结构AI/qzr/output/test_0725v6.json"
    output_folder = "/disk1/user4/work/造船厂/结构AI/qzr/output"
    DEBUG_dxf2json  = True
    input_folder = os.path.dirname(input_json_file)
    file_name = os.path.basename(input_json_file)[:-5]
    input_dxf_file = os.path.join(input_folder, file_name + ".dxf")
    if DEBUG_dxf2json:
        dxf2json(input_folder, file_name, output_folder)
    input_json_file = os.path.join(output_folder, file_name + ".json")
    
    json_result, bbox_result, _, _ = multi_detect_local(input_json_file)
    
    draw_rectangle_in_dxf(input_dxf_file, output_folder, file_name, bbox_result)
    
    json_name = os.path.join(output_folder, file_name) + "_multi.json"

    with open(json_name, 'w', encoding='utf-8') as f:
        for res in json_result:
            f.write(json.dumps(res, ensure_ascii=False, indent=4))
        print("writing success")
        
if __name__ == "__main__":
    #main()
    input_file = "/disk1/user4/work/造船厂/结构AI/qzr/output/test_0725v6.dxf"
    input_folder = "/disk1/user4/work/造船厂/结构AI/qzr/output"
    output_folder = "/disk1/user4/work/造船厂/结构AI/qzr/output"
    multi_detection_main(input_file, input_folder, output_folder)