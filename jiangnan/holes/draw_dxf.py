import os 
import numpy as np 
import ezdxf
import json


def draw_rectangle_in_dxf(file_path, folder, bbox_list, suffix="{}_Holes.dxf"):

    folder = os.path.normpath(os.path.abspath(folder))
    os.makedirs(folder, exist_ok=True)


    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()
    
    if "Holes" not in doc.layers:
        doc.layers.add("Holes", color=30) 

    for idx, bbox in enumerate(bbox_list):
        x1 = bbox["x1"]
        y1 = bbox["y1"]
        x2 = bbox["x2"]
        y2 = bbox["y2"]
        conf = bbox["conf"]

        top_left = (x1, y1)
        top_right = (x2, y1)
        bottom_right = (x2, y2)
        bottom_left = (x1, y2)
        

        # msp.add_line(top_left, top_right, dxfattribs={"layer": "Holes"})  # 红色线条
        # msp.add_line(top_right, bottom_right, dxfattribs={"layer": "Holes"})
        # msp.add_line(bottom_right, bottom_left, dxfattribs={"layer": "Holes"})
        # msp.add_line(bottom_left, top_left, dxfattribs={"layer": "Holes"})
        # 使用polyline绘制矩形
        points = [top_left, top_right, bottom_right, bottom_left]
        msp.add_lwpolyline(points, close=True, dxfattribs={"layer": "Holes"})

        text = msp.add_text("Holes: {:.2f}".format(conf), dxfattribs={"layer": "Holes", "height":200})
        text.dxf.insert = ((x1+x2)/2, y2)
        # msp.add_text("NO.{}".format(idx), dxfattribs={"layer": "Split", "height": 100}).set_dxf_attrib("insert",(x1, y1-20))

        # 保存修改后的 DXF 文件

    file_name = os.path.basename(file_path)[:-4]

    doc.saveas(os.path.join(folder, suffix.format(file_name)))

    print(f"Holes detect result drawn in {os.path.join(folder, suffix.format(file_name))}")
    print("Done....")



# def draw_lwpolyline_in_dxf(file_path, folder, bbox_list):
#     folder = os.path.normpath(os.path.abspath(folder))
#     os.makedirs(folder, exist_ok=True)


#     doc = ezdxf.readfile(file_path)
#     msp = doc.modelspace()
    
#     if "Braket" not in doc.layers:
#         doc.layers.add("Braket", color=30)     


def yoloxyxy2dxfxyxy(bbox: list) -> dict:
    x1, y1, x2, y2, conf = bbox 
    # x1, y1, x2, y2 = x1, y2, x2, y1
    ret = {
        "x1": x1,
        "y1": y2,
        "x2": x2,
        "y2": y1,
        "conf": conf
    }
    return ret


if __name__ == "__main__":


    out_folder = "./out"
    
    bbox_list = []
    with open("nms_results.txt", "r") as f:
        for line in f.readlines():
            line = line.strip().split(",")
            x1, y1, x2, y2, conf = eval(line[0]), eval(line[1]), eval(line[2]), eval(line[3]), eval(line[4]) 
            ret = yoloxyxy2dxfxyxy([x1, y1, x2, y2, conf])
            bbox_list.append(ret)
            
            
    draw_rectangle_in_dxf("./holes_demo.dxf", "./out", bbox_list)
    # bbox_list = [
    #     {
    #         "x1": -269913.3779689523,
    #         "y1": -52909.36737834463,
    #         "x2": -268713.7157786599,
    #         "y2": -52683.02591981944
    #     },
    #     {
    #         "x1": -294848.5755681503,
    #         "y1": -40863.99982124572,
    #         "x2": -289561.7285149262,
    #         "y2": -35275.72210980379
    #     }
    # ]



    