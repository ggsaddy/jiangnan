import argparse
import os
from segment_and_multi_detection.load import dxf2json
from segment_and_multi_detection.segment.main import segment_v4,segment_v4_bound
from segment_and_multi_detection.split.main import multi_detect,multi_detect_bound
from segment_and_multi_detection.split.multi import union_bbox_list
from segment_and_multi_detection.split.multi import merged_bbox_to_json
from segment_and_multi_detection.split.multi import merged_polygon_to_json
import ezdxf
import logging
import json
import multiprocessing
from glob import glob
import numpy as np
from ezdxf.addons import odafc
#bound.json format:
# {
#     "x1": 0,
#     "y1": 0,
#     "x2": 10000,
#     "y2": 10000
# }
DEBUG_SEGMENT = False
def draw_rectangle_in_dxf(file_path,folder,file_name,bbox_list):
    doc= ezdxf.readfile(file_path)
    print(doc.dxfversion)
    if doc.dxfversion < ezdxf.DXF2000:
        doc.upgrade(to_version="R2000")
    msp = doc.modelspace()
    if "SPLIT" not in doc.layers and "Split" not in doc.layers:
        doc.layers.add(name="SPLIT", color=1)
    
    else:
        entity_to_delete =[e for e in msp if e.dxf.layer == "SPLIT" or e.dxf.layer == "Split"]

        for e in entity_to_delete:
            msp.delete_entity(e)
    
    json_str=json.dumps(bbox_list, indent=4)
    with open(os.path.join(folder, {} + "_bbox.json".format(file_name)), 'w', encoding='utf-8') as f:
        f.write(json_str)
    
    for idx,bbox in enumerate(bbox_list):
        x1=bbox["x1"]
        y1=bbox["y1"]
        x2=bbox["x2"]
        y2=bbox["y2"]

        top_left = (x1, y1)
        top_right = (x2, y1)
        bottom_left = (x1, y2)
        bottom_right = (x2, y2)

        msp.add_line(start=top_left, end=top_right, dxfattribs={'layer': 'SPLIT'})
        msp.add_line(start=top_right, end=bottom_right, dxfattribs={'layer': 'SPLIT'})
        msp.add_line(start=bottom_right, end=bottom_left, dxfattribs={'layer': 'SPLIT'})
        msp.add_line(start=bottom_left, end=top_left, dxfattribs={'layer': 'SPLIT'})

        text=msp.add_text("NO:{}".format(idx),dxfattribs={'layer': 'SPLIT',"height":200})
        text.dxf.insert =((x1 + x2) / 2, y2)

    doc.saveas(os.path.join(folder, "{}_split.dxf".format(file_name)))


def draw_rectangle_in_dxf_multi_detect_segment_version(file_path, folder, file_name, bbox_lists):
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
    doc.saveas(os.path.join(folder, "{}_split_with_multi.dxf".format(file_name)))
    return correct, wrong, file_name


def draw_rectangle_in_dxf_multi_detect_segment_merged_version(file_path, folder, file_name, bbox_lists):
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
        
        if  success == 3: #正确的剖面符号和子图类型
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
        
    doc.saveas(os.path.join(folder, "{}_split_with_multi_merged.dxf".format(file_name)))
    return correct, wrong, file_name

def draw_polygon_in_dxf(file_path, folder, file_name, bbox_list):

    output_file=os.path.join(folder, "{}_split.dxf".format(file_name))
    doc = ezdxf.readfile(file_path)
    print(doc.dxfversion)
    if doc.dxfversion < ezdxf.DXF2000:
        print("Upgrade DXF version to R2000")
    msp = doc.modelspace()
    if "Split" in doc.layers:
        doc.layers.remove("Split")
        doc.layers.add("SPLIT", color=1)
    else:
        doc.layers.add(name="SPLIT", color=1)
    
    entity_to_delete = [e for e in msp if e.dxf.layer == "SPLIT" or e.dxf.layer == "Split"]
    for e in entity_to_delete:
        msp.delete_entity(e)
    json_data=[]
    new_bbox_list=[]
    for bbox in bbox_list:
        if type(bbox)!=type(bbox_list[0]):
            continue
        coords=list(bbox.exterior.coords)
        if coords not in new_bbox_list:
            new_bbox_list.append(coords)
    json_poly_data=[]
    for idx, coords in enumerate(new_bbox_list):
        msp.add_lwpolyline(coords, close=True, dxfattribs={'layer': 'SPLIT'})
        coords = np.array(coords)
        x1= np.min(coords[:, 0])
        x2= np.max(coords[:, 0])
        y1= np.min(coords[:, 1])
        y2= np.max(coords[:, 1])
        b={
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        }
        if b not in json_data:
            json_data.append(b)
        polygon={
            "coordinates": [coords.tolist()],
        }
        if polygon not in json_poly_data:
            json_poly_data.append(polygon)
        
    json_str= json.dumps(json_data, indent=4)
    json_poly_str= json.dumps(json_poly_data, indent=4)
    with open(os.path.join(folder, "{}_bbox.json".format(file_name)), 'w', encoding='utf-8') as f:
        f.write(json_str)
    with open(os.path.join(folder, "{}_polygon.json".format(file_name)), 'w', encoding='utf-8') as f:
        f.write(json_poly_str)
    doc.saveas(os.path.join(folder, "{}_split.dxf".format(file_name)))

def segment_all_main(input_file,input_folder,output_folder,input_bound_json):
    input_file=os.path.abspath(os.path.normpath(input_file))
    element_list=input_file.split(os.sep)
    input_folder=element_list[0]+"/"+os.path.join(*element_list[1:-1])
    file_name=os.path.basename(input_file)[:-4]
    json_file=os.path.join(output_folder, file_name + ".json")
    img_file=os.path.join(output_folder, file_name + ".jpg")
    split_file=os.path.join(output_folder, file_name + "_split.dxf")
    print("split start .....")
    print(input_file)
    dxf2json(input_folder,file_name,output_folder)
    print("segment start .....")
    bbox_list=segment_v4_bound(json_file,img_file,input_bound_json)
    print(bbox_list)
    if len(bbox_list)!=0:
        print("visualize start .....")
        draw_polygon_in_dxf(input_file, output_folder, file_name, bbox_list)
    
    if DEBUG_SEGMENT==False:
        json_result,bbox_result,_,false_count=multi_detect_bound(json_file,img_file,input_bound_json)
        print(bbox_result)
        json_name=os.path.join(output_folder, file_name + "_multi.json")
        json_bbox_file=os.path.join(output_folder, file_name + "_bbox.json")
        with open(json_name, 'w', encoding='utf-8') as f:
            for res in json_result:
                f.write(json.dumps(res, ensure_ascii=False, indent=4))
                print(f"Writing to {json_name} completed.")
        json_merged_name = os.path.join(output_folder, file_name) + "_multi_merged.json"
        with open(json_merged_name, 'w', encoding='utf-8') as f:
            for res in json_result:
                for i in range(len(res) - 1, -1, -1):
                    item = res[i]
                    if isinstance(item, dict):
                        if "相似场景" in item.keys() or "子图调用次数" in item.keys():
                            res.pop(i)  # 使用pop(index)删除指定位置的元素
                f.write(json.dumps(res, ensure_ascii=False, indent=4))
        correct,wrong, file_name=draw_rectangle_in_dxf_multi_detect_segment_version(
            split_file, output_folder, file_name, bbox_result
        )
        bbox_list_merged=union_bbox_list(bbox_result,json_bbox_file)
        output_json_bbox_file=os.path.join(output_folder, file_name + "_merged_bbox.json")
        merged_bbox_to_json(bbox_list_merged,json_bbox_file,output_json_bbox_file)
        json_polygon_file=os.path.join(output_folder, file_name + "_polygon.json")
        output_json_polygon_file=os.path.join(output_folder, file_name + "_merged_polygon.json")
        merged_polygon_to_json(bbox_list_merged,json_polygon_file,output_json_polygon_file)
        correct,wrong,file_name=draw_rectangle_in_dxf_multi_detect_segment_merged_version(split_file, output_folder, file_name, bbox_list_merged)
        result_mergerd_bbox=[]
        with open(output_json_polygon_file, 'r', encoding='utf-8') as f:
            result_mergerd_bbox = json.load(f)
    return result_mergerd_bbox

def main(input_folder,output_folder,input_bound_json):
    input_list=glob(os.path.join(input_folder, "*.dxf"))
    output_folder=os.path.abspath(os.path.normpath(output_folder))
    os.makedirs(output_folder, exist_ok=True)
    for input_file in input_list:
        try:
            if select_file is not None:
                if select_file not in input_file:
                    continue
        except:
            pass
        result_mergerd_bbox_list=segment_all_main(input_file, input_folder, output_folder,input_bound_json)
        print(f"Processed {input_file}, result: {result_mergerd_bbox_list}")
if __name__ == "__main__":
    multiprocessing.freeze_support()
    logging.basicConfig(filename="error.txt",level=logging.DEBUG,format="%(asctime)s - %(levelname)s - %(message)s")
    input_folder = "C:/Users/31285/Desktop/曲子睿的文件夹/复旦/CAD分割/DXFStruct/output_check"
    output_folder = "C:/Users/31285/Desktop/曲子睿的文件夹/复旦/CAD分割/DXFStruct/output_check/segment_0804"
    input_bound_json="C:/Users/31285/Desktop/曲子睿的文件夹/复旦/CAD分割/DXFStruct/output_check/bound.json"
    main(input_folder, output_folder,input_bound_json)
    print("ALL FINISHED!")