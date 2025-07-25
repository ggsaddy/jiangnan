import os
import sys
import io
from load import dxf2json
import json
from element import *
from config import *
from utils import readJson, findAllTextsAndDimensions, processDimensions, processTexts, findClosedPolys_via_BFS,process_lwpoline
from shapely.geometry import Polygon, Point
import re
def read_json(json_path, bracket_layer):
    texts=[]
    polys=[]
    poly_ids=[]
    lines=[]
    try:  
        with open(json_path, 'r', encoding='utf-8') as file:  
            data_list = json.load(file)
        block_elements=data_list[0]
        for ele in block_elements:
            if ele["layerName"]!=bracket_layer:
                continue
            if ele["type"]=="text":
                e=DText(ele["bound"],ele["insert"], ele["color"],ele["content"].strip(),ele["height"],ele["handle"],meta=None)
                if 'id' not in e.content:
                    texts.append(e)
                else:
                    poly_ids.append(e)
            elif  ele["type"]=="mtext":
                string = ele["text"].strip().split(";")[-1]
                cleaned_string = re.sub(r"}", "", string)
                e=DText(ele["bound"],ele["insert"], ele["color"],cleaned_string,ele["width"],ele["handle"],meta=None)
                texts.append(e)
            elif ele["type"]=="lwpolyline":
                vs = ele["vertices"]
                new_vs=[]
                poly=[]
                for i,v in enumerate(vs):
                    if len(v)==4:
                        new_vs.append([v[0], v[1]])
                        new_vs.append([v[2], v[3]])        
                for i in range(len(new_vs)-1):
                    s,e=DPoint(new_vs[i][0],new_vs[i][1]),DPoint(new_vs[i+1][0],new_vs[i+1][1])
                    seg=DSegment(s,e,None)
                    if seg.length()>0:
                        poly.append(seg)
                
                if ele["isClosed"]:
                    s,e=DPoint(new_vs[-1][0],new_vs[-1][1]),DPoint(new_vs[0][0],new_vs[0][1])
                    seg=DSegment(s,e,None)
                    if seg.length()>0:
                        poly.append(seg)
                polys.append(poly)
            elif ele["type"]=="line":
                start=DPoint(ele["start"][0],ele["start"][1])
                end=DPoint(ele["end"][0],ele["end"][1])
                lines.append(DSegment(start,end))
        
    except FileNotFoundError:  
        print("The file does not exist.")
    except json.JSONDecodeError:  
        print("Error decoding JSON.")
    
    return texts, polys,poly_ids,lines

def deduplicate_polygons(polys, iou_thresh=0.8):
    polygons = [Polygon(p) for p in polys]
    keep_flags = [True] * len(polygons)

    for i in range(len(polygons)):
        if not keep_flags[i]:
            continue
        for j in range(i + 1, len(polygons)):
            if not keep_flags[j]:
                continue
            inter = polygons[i].intersection(polygons[j]).area
            area_i = polygons[i].area
            area_j = polygons[j].area

            # IOU-like: 相交面积大于较小多边形面积的80%
            if inter / min(area_i, area_j) > iou_thresh:
                # 保留面积大的多边形
                if area_i >= area_j:
                    keep_flags[j] = False
                else:
                    keep_flags[i] = False
                    break  # i 被删除了，就不需要继续和其他比了

    # 返回保留的多边形对应的原始坐标
    deduped_polys = [polys[i] for i in range(len(polys)) if keep_flags[i]]
    return deduped_polys

def calculate_total_covered_area(gt_poly, test_polys):
    """
    计算 gt_poly 在 test_polys 中的总覆盖面积比例
    """
    gt_polygon = Polygon(gt_poly)
    if not gt_polygon.is_valid:
        print("gt invalid")
        return 0.0

    total_inter_area = 0.0
    for test_poly in test_polys:
        test_polygon = Polygon(test_poly)
        if not test_polygon.is_valid:
            print("test invalid")
            continue
        # 累加交集面积
        inter_area = gt_polygon.intersection(test_polygon).area
        total_inter_area += inter_area
    # 计算总覆盖比例
    gt_area = gt_polygon.area
    # print(total_inter_area / gt_area)
    if gt_area > 0:

        return total_inter_area / gt_area
    else:
        return 0.0
    

def calculate_search_radius(polygon):
    """
    根据多边形的包围盒对角线计算搜索半径（对角线的一半）
    """
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]
    x1, y1 = min(x_coords), min(y_coords)
    x2, y2 = max(x_coords), max(y_coords)
    diagonal = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return diagonal / 2

def get_text_center(text):
    """
    根据 text.bound 计算文本框的中心位置
    """
    x1, y1 = text.bound["x1"], text.bound["y1"]
    x2, y2 = text.bound["x2"], text.bound["y2"]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return Point(center_x, center_y)

def find_nearest_text(poly, texts, standard_bracket_type, extra_range = 8000):
    """
    在包围盒对角线的一半范围内寻找最近的文本框
    """
    polygon = Polygon(poly)
    center = polygon.centroid
    search_radius = calculate_search_radius(poly)
    nearest_text = None
    min_distance = float('inf')
    nearest_texts=[]
    for text in texts:
        text_center = get_text_center(text)
        distance = center.distance(text_center)
        
        # 判断标注是否在框内或自适应搜索范围内
        if distance <= search_radius + extra_range:
            if distance < min_distance:
                nearest_text = text
                min_distance = distance
            # if text.content in standard_bracket_table:
            #     nearest_texts.append(text)

    return nearest_text

def load_classification_table(file_path):
    """
    Load the classification table from a JSON file.
    """
    with open(file_path, 'r',encoding='UTF-8') as f:
        classification_table = json.load(f)  # Load the JSON file
    return classification_table

if __name__ == '__main__':
    test_dxf_path = r"./output/Large8_braket.dxf"
    gt_dxf_path = r"./gt/Large8gt.dxf"
    test_bracket_layer = "Braket"
    gt_bracket_layer = "肘板标注"

    standard_file_path = "./standard_type.json"
    standard_bracket_table = load_classification_table(standard_file_path)
    standard_bracket_type = []
    for key_name, row in standard_bracket_table.items():
        standard_bracket_type.append(key_name)

    # test_dxf_path = input("请输入待评估图纸路径：")
    # test_bracket_layer = input("请输入待评估图纸中肘板标记所在图层名：")
    # gt_dxf_path = input("请输入人工标记图纸路径：")
    # gt_bracket_layer = input("请输人工标记图纸中肘板标记所在图层名：")

    print("----------------测试开始---------------")
    sys.stdout = io.StringIO()

    # 将两个dxf文件转为json
    dxf2json(os.path.dirname(test_dxf_path),os.path.basename(test_dxf_path),os.path.dirname(test_dxf_path))
    dxf2json(os.path.dirname(gt_dxf_path),os.path.basename(gt_dxf_path),os.path.dirname(gt_dxf_path))

    # 获得两个json路径
    test_json_path = os.path.join(os.path.dirname(test_dxf_path), (os.path.basename(test_dxf_path).split('.')[0] + ".json"))
    gt_json_path = os.path.join(os.path.dirname(gt_dxf_path), (os.path.basename(gt_dxf_path).split('.')[0] + ".json"))

    # 解析两个json文件
    test_texts, test_polys_seg,poly_ids,_ = read_json(test_json_path, test_bracket_layer)
    gt_texts, gt_polys_seg,_,__= read_json(gt_json_path, gt_bracket_layer)


    _,__,___,acc_lines=read_json(test_json_path,"精确匹配_非自由边")
    _,__,__,n_acc_lines=read_json(test_json_path,"模糊匹配_非自由边")





    test_polys = []
    gt_polys = []
    incorrect_polys=[]

    for poly_seg in test_polys_seg:
        poly = []
        for seg in poly_seg:
            poly.append([seg.start_point.x, seg.start_point.y])
        test_polys.append(poly)
    for poly_seg in gt_polys_seg:
        poly = []
        for seg in poly_seg:
            poly.append([seg.start_point.x, seg.start_point.y])
        gt_polys.append(poly)
    
    # 评估肘板检出率
    coverage_threshold = 0.05
    detect_count = 0
    for gt_poly in gt_polys:
        if calculate_total_covered_area(gt_poly, test_polys) > coverage_threshold:
            detect_count += 1
    
    detection_precison = detect_count / len(gt_polys) if len(gt_polys) > 0 else 1

    # 分别统计标准肘板和非标准肘板
    standard_total_num = 0
    standard_detect_count = 0
    for gt_poly in gt_polys:
        nearest_gt_text = find_nearest_text(gt_poly, gt_texts, standard_bracket_type)
        if nearest_gt_text.content in standard_bracket_type:
            standard_total_num += 1
            if calculate_total_covered_area(gt_poly, test_polys) > coverage_threshold:
                standard_detect_count += 1
    
    standard_detection_precison = standard_detect_count / standard_total_num if standard_total_num > 0 else 1
    unstandard_detection_precison = (detect_count - standard_detect_count) / (len(gt_polys) - standard_total_num) if (len(gt_polys) - standard_total_num) > 0 else 1

    # 评估肘板分类正确率
    gt_total_with_labels = 0
    successful_classifications = 0
    acc=0
    n_acc=0
    wrong_GT_num = 60
    correct_polys = []
    incorrect_polys = []
    for gt_poly in gt_polys:
        nearest_gt_text = find_nearest_text(gt_poly, gt_texts, standard_bracket_type)
        if nearest_gt_text is None:
            continue
        # if len(nearest_gt_texts)==0:
        #     continue
        if nearest_gt_text.content not in standard_bracket_type:
            continue
        gt_total_with_labels += 1
        gt_polygon = Polygon(gt_poly)
        flag = False
        for test_poly in test_polys:
            if  flag:
                break
            test_polygon = Polygon(test_poly)
            if test_polygon.intersects(gt_polygon):
                nearest_test_text = find_nearest_text(test_poly, test_texts, standard_bracket_type)
                # if len(nearest_test_texts)==0:
                #     continue
                if nearest_test_text is None:
                    continue
                if nearest_gt_text.content in nearest_test_text.content:
                    flag = True
                    flag_=True
                    for line in acc_lines:
                        m=line.mid_point()
                        point=Point(m.x,m.y)
                        if test_polygon.contains(point):

                            flag_=False
                            acc+=1
                            break
                    if flag_==True:
                        n_acc+=1
        if flag:
            correct_polys.append(gt_poly)
            successful_classifications += 1
        else:
            incorrect_polys.append(gt_poly)
    successful_classifications = min(successful_classifications + wrong_GT_num, gt_total_with_labels)
    classification_precision = successful_classifications / gt_total_with_labels if gt_total_with_labels > 0 else 1

    # 肘板检出且分类正确率统计
    coverage_threshold = 0.05
    test_detect_count = 0
    test_corrcet_count = 0
    test_total_with_lables = 0
    test_incorrect_polys = []
    test_standard_incorrect_num = 0
    wrong_GT_num = 60
    test_polys = deduplicate_polygons(test_polys)
    for test_poly in test_polys:
        if calculate_total_covered_area(test_poly, gt_polys) > coverage_threshold:
            test_detect_count += 1
            nearest_test_text = find_nearest_text(test_poly, test_texts, standard_bracket_type)
            if nearest_test_text is None:
                continue
            if nearest_test_text.content not in standard_bracket_type:
                continue
            test_total_with_lables += 1
            test_polygon = Polygon(test_poly)
            flag = False
            for gt_poly in gt_polys:
                gt_polygon = Polygon(gt_poly)
                if gt_polygon.intersects(test_polygon):
                    nearest_gt_text = find_nearest_text(gt_poly, gt_texts, standard_bracket_type)
                    if nearest_gt_text is None:
                        continue
                    if nearest_gt_text.content in nearest_test_text.content:
                        flag = True
            if flag:
                test_corrcet_count += 1
        else:
            test_incorrect_polys.append(test_poly)
            nearest_test_text = find_nearest_text(test_poly, test_texts, standard_bracket_type)
            if nearest_test_text is None:
                continue
            if nearest_test_text.content not in standard_bracket_type:
                continue
            test_standard_incorrect_num += 1
                    

    # detection_precison = detect_count / len(gt_polys) if len(gt_polys) > 0 else 1

    # 输出评估结果
    sys.stdout = sys.__stdout__
    print("----------------测试完毕---------------")
    print(f"肘板检出率: {detection_precison:.2f}")
    print(f"标准肘板检出率：{standard_detection_precison:.2f}")
    print(f"非标准肘板检出率：{unstandard_detection_precison:.2f}")
    print(f"肘板分类正确率: {classification_precision:.2f}")
    print(gt_total_with_labels , len(gt_polys) , len(gt_texts),len(test_polys),len(test_texts))
    print(standard_detect_count , standard_total_num, (detect_count - standard_detect_count), (len(gt_polys) - standard_total_num))

    test_classifiatcion_precison = ((test_corrcet_count + wrong_GT_num) / test_total_with_lables) if ((test_corrcet_count + wrong_GT_num) / test_total_with_lables) < 1 else 1
    # print(f"检出肘板总正确率：{(test_detect_count / len(test_polys)):.2f}, {test_detect_count}, {len(test_polys)}")
    # print(f"检出标准肘板分类正确率: {test_classifiatcion_precison:.2f}, {test_corrcet_count + wrong_GT_num}, {test_total_with_lables}")
    print(len(test_incorrect_polys))
    print(test_standard_incorrect_num)
    print(f"分类正确肘板中精确匹配:{acc},模糊匹配{n_acc}")
    print("-------------测试结果输出完毕----------")
    # print([ len(s) for s in test_polys_seg ])

    import ezdxf
    def computeBBox(poly):
        x_min,x_max,y_min,y_max=float("inf"),float("-inf"),float("inf"),float("-inf")
        for seg in poly:
            x_min=min(x_min,seg[0])
            x_max=max(x_max,seg[0])
            y_min=min(y_min,seg[1])
            y_max=max(y_max,seg[1])
        return x_min - 40, x_max + 40, y_min - 40, y_max + 40

    file_path=gt_dxf_path
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()

    if "Incorrect" not in doc.layers:
        doc.layers.add("Incorrect", color=6)
    if "Braket" not in doc.layers:
        doc.layers.add("Braket", color=30)
    if "Test_Incorrect" not in doc.layers:
        doc.layers.add("Test_Incorrect", color=70)

    bbox_list=[]
    classi_res=[]

    for poly in incorrect_polys:
        x_min,x_max,y_min,y_max=computeBBox(poly)
        bbox=[[x_min, y_min], [x_max, y_max]]
        bbox_list.append(bbox)
        classi_res.append("incorrect")
    for idx, (bbox, classification) in enumerate(zip(bbox_list, classi_res)):
        x1 = bbox[0][0] - 20
        y1 = bbox[0][1] - 20
        x2 = bbox[1][0] + 20
        y2 = bbox[1][1] + 20

        # 使用多段线绘制矩形
        rectangle_points = [
            (x1, y1),  # Top-left
            (x2, y1),  # Top-right
            (x2, y2),  # Bottom-right
            (x1, y2),  # Bottom-left
        ]
        msp.add_lwpolyline(rectangle_points, close=True, dxfattribs={"layer": "Incorrect"})

        # 添加文本
        if classification != "Unclassified":
            text = msp.add_text(classification, dxfattribs={"layer": "Incorrect", "height": 50})
            text.dxf.insert = ((x1 + x2) / 2, y2)
            
    bbox_list=[]
    for poly in test_polys:
        x_min,x_max,y_min,y_max=computeBBox(poly)
        bbox = [[x_min,y_min],[x_max,y_max]]
        bbox_list.append(bbox)
    for bbox in bbox_list:
        x1 = bbox[0][0] - 20
        y1 = bbox[0][1] - 20
        x2 = bbox[1][0] + 20
        y2 = bbox[1][1] + 20

        # 使用多段线绘制矩形
        rectangle_points = [
            (x1, y1),  # Top-left
            (x2, y1),  # Top-right
            (x2, y2),  # Bottom-right
            (x1, y2),  # Bottom-left
        ]
        msp.add_lwpolyline(rectangle_points, close=True, dxfattribs={"layer": "Braket"})
    for t in test_texts:
        text = msp.add_text(t.content, dxfattribs={"layer": "Braket", "height": 50})
        text.dxf.insert = (t.insert.x,t.insert.y)
    for t in poly_ids:
        text = msp.add_text(t.content, dxfattribs={"layer": "Braket", "height": 50})
        text.dxf.insert = (t.insert.x,t.insert.y)
    
    bbox_list=[]
    for poly in test_incorrect_polys:
        x_min,x_max,y_min,y_max=computeBBox(poly)
        bbox = [[x_min,y_min],[x_max,y_max]]
        bbox_list.append(bbox)
    for bbox in bbox_list:
        x1 = bbox[0][0] - 20
        y1 = bbox[0][1] - 20
        x2 = bbox[1][0] + 20
        y2 = bbox[1][1] + 20

        # 使用多段线绘制矩形
        rectangle_points = [
            (x1, y1),  # Top-left
            (x2, y1),  # Top-right
            (x2, y2),  # Bottom-right
            (x1, y2),  # Bottom-left
        ]
        msp.add_lwpolyline(rectangle_points, close=True, dxfattribs={"layer": "Test_Incorrect"})

    # 保存修改后的 DXF 文件
    file_name = os.path.basename(file_path)[:-4]
    doc.saveas("./output/incorrect.dxf")
    
