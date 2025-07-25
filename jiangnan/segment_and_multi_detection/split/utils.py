import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from split.src import component, UnionFind, all_components, sub_components
from tqdm import tqdm
from collections import deque
import copy
from multiprocessing import Pool
import math
# 高宽比
height_list = {"0": 841,
                "1": 594,
                "2": 420,
                "3": 297}
jump_polyline = ["polyline", "lwpolyline", "spline", "hatch", "arc", "line", "leader", "dimension"]
# 判断 box1 是否完全包含 box2
def is_fully_contained_with_outlier(box1, box2):
    # 放宽条件，在 box2 的上下左右四个边界分别加上或减去 1 的缓冲
    return (box1['x1'] <= box2['x1'] + 1 and box1['x2'] >= box2['x2'] - 1 and
            box1['y1'] <= box2['y1'] + 1 and box1['y2'] >= box2['y2'] - 1)

#判断坐标轴
def deal_axis(box1, box2):
    return (box1['x1'] > box2['x1']  and box1['x2'] < box2['x2'] and
            box1['y1'] < box2['y1']  and  box1['y2'] > box2['y2'])

# 判断 box1 是否严格包含 box2
def is_fully_contained_strictly(box1, box2):
    return (box1['x1'] < box2['x1']  and box1['x2'] > box2['x2'] and
            box1['y1'] < box2['y1']  and box1['y2'] > box2['y2'] )

def judge_double_contain(box1, box2):
    return is_fully_contained_with_outlier(box1, box2) or is_fully_contained_with_outlier(box2, box1)

#判断两个bbox是否相交
def is_intersecting(box1, box2):
    intersecting =  not (box1['x2'] < box2['x1'] or  # box1 在 box2 的左边
                        box1['x1'] > box2['x2'] or  # box1 在 box2 的右边
                        box1['y2'] < box2['y1'] or  # box1 在 box2 的下面
                        box1['y1'] > box2['y2'])    # box1 在 box2 的上面
    return intersecting

 
# 判断两个包围盒是否相交或一个包含另一个
def is_intersecting_or_containing(box1, box2):
    
    if type(box1) == list:
        for i in range(len(box1)):
            result = is_intersecting_or_containing(box1[i], box2)
            if result == True:
                return result
        
        return result

    
    intersecting = not (box1['x2'] < box2['x1'] or  # box1 在 box2 的左边
                        box1['x1'] > box2['x2'] or  # box1 在 box2 的右边
                        box1['y2'] < box2['y1'] or  # box1 在 box2 的下面
                        box1['y1'] > box2['y2'])    # box1 在 box2 的上面

    containing = is_fully_contained_with_outlier(box1, box2)

    contained_by = is_fully_contained_with_outlier(box2, box1)

    return intersecting or containing or contained_by

#判断两个bbox的距离是否在阈值内
def is_nearby(box1, box2, threshold):
    if type(box2) == list:
        result = True
        for i in range(len(box2)):
            tmp = is_nearby(box1, box2[i], threshold)
            result = result and tmp
        return result
   
    if is_fully_contained_with_outlier(box1, box2) or is_fully_contained_with_outlier(box2, box1):
        return False
    
    
    if is_intersecting_or_containing(box1, box2) or is_intersecting_or_containing(box2, box1):
        return True
    
    # 获取 box1 和 box2 的边界坐标
    x1_min, y1_min, x1_max, y1_max = float(box1["x1"]), float(box1["y1"]), float(box1["x2"]), float(box1["y2"])
    x2_min, y2_min, x2_max, y2_max = float(box2["x1"]), float(box2["y1"]), float(box2["x2"]), float(box2["y2"])


    # 计算水平和垂直方向的最近距离
    dx = max(0, x2_min - x1_max, x1_min - x2_max)
    dy = max(0, y2_min - y1_max, y1_min - y2_max)

    # 计算两个 box 之间的最近距离
    distance = (dx ** 2 + dy ** 2) ** 0.5
    
    # 判断距离是否在阈值范围内
    return distance <= threshold

#计算两个bbox距离
def calc_distance_on_bbox(box1, box2):
    
    if is_fully_contained_with_outlier(box2, box1):
        return 0
    
    x1_min, y1_min, x1_max, y1_max = float(box1["x1"]), float(box1["y1"]), float(box1["x2"]), float(box1["y2"])
    x2_min, y2_min, x2_max, y2_max = float(box2["x1"]), float(box2["y1"]), float(box2["x2"]), float(box2["y2"])

     # 计算水平和垂直方向的最近距离
    dx = max(0, x2_min - x1_max, x1_min - x2_max)
    dy = max(0, y2_min - y1_max, y1_min - y2_max)

    # 计算两个 box 之间的最近距离
    distance = (dx ** 2 + dy ** 2) ** 0.5

    # 判断距离是否在阈值范围内
    return distance


def calc_distance_from_bbox_to_point(box, point):
    # 获取 box 的边界坐标
    x_min, y_min, x_max, y_max = float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
    
    # 计算水平和垂直方向的最近距离
    dx = max(0, point[0] - x_max, x_min - point[0])
    dy = max(0, point[1] - y_max, y_min - point[1])

    # 计算 box 到点的距离
    distance = (dx ** 2 + dy ** 2) ** 0.5
    
    return distance
def calc_distance_from_center_point(box1, box2):
    center1 = [(box1["x1"] + box1["x2"]) / 2, (box1["y1"] + box1["y2"]) / 2] 
    center2 = [(box2["x1"] + box2["x2"]) / 2, (box2["y1"] + box2["y2"]) / 2] 

    return cal_eucilean_distance(center1, center2)


def cal_eucilean_distance(v1, v2):
    x = abs(round(v1[0] - v2[0]))
    y = abs(round(v1[1] - v2[1]))
    
    return math.sqrt(x ** 2 + y ** 2)

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# 包围盒的映射函数（坐标轴->图纸，100倍）
def calculate_std_length(bbox, threshold=0.01):
    length = bbox['x2'] - bbox['x1']
    return length * threshold

# 计算包围盒的面积
def calculate_area(bbox):
    width = bbox['x2'] - bbox['x1']
    height = bbox['y2'] - bbox['y1']
    return width * height

# 加载 JSON 数据
def load_data(filepath):
    with open(filepath, 'r', encoding='utf8') as fp:
        result = all_components()
        json_data = json.load(fp)
        if len(json_data) == 2:
            json_data = json_data[0]
        for data in json_data:
            tmp = component(data["type"], data["bound"], data)
            result.push(tmp)
    
    return result

#去除重复图框
def remove_duplicate_tukuang(main_bbox_dict):
    duplicate_tukuang = []
    new_bbox_dict = {}
    count = 1
    new_bbox_dict[0] = main_bbox_dict[0]

    for i in range(1, len(main_bbox_dict)):
        contain = False
        area_i = calculate_area(main_bbox_dict[i])
        for j in range(len(new_bbox_dict)):
            area_j = calculate_area(new_bbox_dict[j])
            print("i: ", i, main_bbox_dict[i])
            print("j: ",j, new_bbox_dict[j])
            if judge_double_contain(new_bbox_dict[j], main_bbox_dict[i]):
                contain = True
                if area_i > area_j:
                    duplicate_tukuang.append(new_bbox_dict[j])
                    new_bbox_dict[j] = main_bbox_dict[i]
                else:
                    duplicate_tukuang.append(main_bbox_dict[i])
        
        if not contain:
            new_bbox_dict[count] = main_bbox_dict[i]
            count += 1
    
    return new_bbox_dict, duplicate_tukuang

def check_duplicate(tukuang, dupli_tukuangs):
    for i in range(len(dupli_tukuangs)):
        if tukuang["x1"] == dupli_tukuangs[i]["x1"] and \
            tukuang["y1"] == dupli_tukuangs[i]["y1"] and \
            tukuang["x2"] == dupli_tukuangs[i]["x2"] and \
            tukuang["y2"] == dupli_tukuangs[i]["y2"]:
            return False

    return True

#去除左上、右上、左下、右下四个区域的位置
def get_forbidden_area(bbox, x_longer):
    forbidden_area_thres1_x = 10 / 107
    forbidden_area_thres1_y = 0.030
    forbidden_area_thres2_x = 0.20

    if x_longer: #横向图框
        forbidden_area1 = {
            "x1": bbox["x1"],
            "x2": bbox["x1"] + forbidden_area_thres1_x * (bbox["x2"] - bbox["x1"]),
            "y1": bbox["y2"] - forbidden_area_thres1_y * (bbox["y2"] - bbox["y1"]),
            "y2": bbox["y2"]
        }
        forbidden_area2 = {
            "x1": bbox["x2"] - forbidden_area_thres2_x * (bbox["x2"] - bbox["x1"]),
            "x2": bbox["x2"],
            "y1": bbox["y2"] - forbidden_area_thres1_y * (bbox["y2"] - bbox["y1"]),
            "y2": bbox["y2"]
        }
        return forbidden_area1, forbidden_area2, None, None
    else: #纵向图框
        forbidden_area1 = {
            "x1": bbox["x2"] - forbidden_area_thres1_y * (bbox["x2"] - bbox["x1"]),
            "x2": bbox["x2"],
            "y1": bbox["y2"] - forbidden_area_thres1_x * (bbox["y2"] - bbox["y1"]),
            "y2": bbox["y2"]
        }
        forbidden_area2 = {
            "x1": bbox["x2"] - forbidden_area_thres1_y * (bbox["x2"] - bbox["x1"]),
            "x2": bbox["x2"],
            "y1": bbox["y1"],
            "y2": bbox["y1"] + forbidden_area_thres2_x * (bbox["y2"] - bbox["y1"])
        }
        forbidden_area3 = {
            "x1": bbox["x1"],
            "x2": bbox["x1"] + forbidden_area_thres1_y * (bbox["x2"] - bbox["x1"]),
            "y1": bbox["y1"],
            "y2": bbox["y1"] + forbidden_area_thres1_x * (bbox["y2"] - bbox["y1"])
        }
        forbidden_area4 = {
            "x1": bbox["x1"],
            "x2": bbox["x1"] + forbidden_area_thres1_y * (bbox["x2"] - bbox["x1"]),
            "y1": bbox["y2"] - forbidden_area_thres2_x * (bbox["y2"] - bbox["y1"]),
            "y2": bbox["y2"]
        }

        return forbidden_area1, forbidden_area2, forbidden_area3, forbidden_area4
    
def calc_chang_kuan_ratio(bbox):
    length = bbox["x2"] - bbox["x1"]
    width = bbox["y2"] - bbox["y1"]

    if length > width:
        return 1

    return 0

#获取大图框中的小图框和左上、右上、左下、右下四个区域
def get_new_tukuang_and_forbidden_area(bbox):
    tukuang_threshold_x1 = 1 / 60
    tukuang_threshold_x2 = 1 / 60
    tukuang_threshold_y1 = 200 #绝对值判断
    tukuang_threshold_y2 = 1 / 42

    if calc_chang_kuan_ratio(bbox):
        x1 = bbox["x1"] + tukuang_threshold_x1 * (bbox["x2"] - bbox["x1"])
        x2 = bbox["x2"] - tukuang_threshold_x2 * (bbox["x2"] - bbox["x1"])
        y1 = bbox["y1"] + 200
        y2 = bbox["y2"] - tukuang_threshold_y2 * (bbox["y2"] - bbox["y1"])

        new_bbox = {
            "x1":x1,
            "x2":x2,
            "y1":y1,
            "y2":y2
        }
        f1, f2, f3, f4 = get_forbidden_area(new_bbox, 1)
        return new_bbox, f1, f2, f3, f4
    else:
        x1 = bbox["x1"] + tukuang_threshold_y1 * (bbox["x2"] - bbox["x1"])
        x2 = bbox["x2"] - tukuang_threshold_y2 * (bbox["x2"] - bbox["x1"])
        y1 = bbox["y1"] + 200
        y2 = bbox["y2"] - tukuang_threshold_x2 * (bbox["y2"] - bbox["y1"])
        new_bbox = {
            "x1":x1,
            "x2":x2,
            "y1":y1,
            "y2":y2
        }
        f1, f2, f3, f4 = get_forbidden_area(new_bbox, 0)
        return new_bbox, f1, f2, f3, f4
    
# 加载 JSON 数据并获得主要bbox
def load_data_and_get_main_bbox(filepath, threshold=0.1):
    main_bbox_dict = {}
    result = {}
    count = 0
    area_threshold = 1102342000
    human_add_list = []
    with open(filepath, 'r', encoding='utf8') as fp:
        json_data = json.load(fp)
        if len(json_data) == 2:
            json_data_main = json_data[0]
            json_data_remain = json_data[1]
        for data in json_data_main:
            if data["layerName"] == "SPLIT" and data["color"] !=  1: #人工处理，和原本正确的红色框进行区分（改变颜色）
                human_add_list.append(data["bound"])
            else:
                area = calculate_area(data["bound"])
                if area >= area_threshold * 0.99 and area < area_threshold * 20:
                    main_bbox_dict[count] = data["bound"]
                    count += 1
                   
        if len(main_bbox_dict) == 0:
            raise ValueError("未检测到图框, 请检查图框类型和尺寸.")
        
        #图框去重
        main_bbox_dict, duplicate_tukuang = remove_duplicate_tukuang(main_bbox_dict)
        
        for data in json_data_main:
            # NEWADD
            if data["layerName"] != "SPLIT" and "修改" not in data["layerName"]: #排除程序画的红线
            # </NEWADD>
                #获取人工处理的元素
                already_add = False
                for human_index, human_bbox in enumerate(human_add_list):
                    if is_fully_contained_strictly(human_bbox, data["bound"]):
                        tmp = component(data["type"], data["bound"], data)
                        tmp.manual_index = human_index
                        for i, main_bbox in main_bbox_dict.items():
                            if is_fully_contained_strictly(main_bbox, data["bound"]):
                                if i in result and human_index in result[i]["human"]:
                                    result[i]["human"][human_index].push(tmp)
                                elif i in result and human_index not in result[i]["human"]:
                                    result[i]["human"][human_index] = all_components()
                                    result[i]["human"][human_index].push(tmp)
                                else:
                                    result[i] = {"main":all_components(), "human":{}}
                                    result[i]["human"][human_index] = all_components()
                                    result[i]["human"][human_index].push(tmp)
                        already_add = True
                if already_add == False:
                    #获取剩余在图框中且不在禁区内的元素
                    for i, bbox in main_bbox_dict.items():
                        new_bbox, f1, f2, f3, f4 = get_new_tukuang_and_forbidden_area(bbox)
                        if is_fully_contained_strictly(new_bbox, data["bound"]) and check_duplicate(data["bound"], duplicate_tukuang) and \
                            not is_fully_contained_with_outlier(f1, data["bound"]) and not is_fully_contained_with_outlier(f2, data["bound"]):
                                if f3 is not None and f4 is not None:
                                    if not is_fully_contained_with_outlier(f3, data["bound"]) and not is_fully_contained_with_outlier(f4, data["bound"]):
                                        tmp = component(data["type"], data["bound"], data)
                                        if i in result:
                                            result[i]["main"].push(tmp)
                                        else:
                                            result[i] = {"main":all_components(), "human":{}}
                                            result[i]["main"].push(tmp)
                                else:
                                    tmp = component(data["type"], data["bound"], data)
                                    if i in result:
                                        result[i]["main"].push(tmp)
                                    else:
                                        result[i] = {"main":all_components(), "human":{}}
                                        result[i]["main"].push(tmp)
                                            

    return result, json_data_remain


# 本地load方法
def load_data_and_get_main_bbox_local(filepath, threshold=0.1):
    main_bbox_dict = {}
    count = 0
    
    human_add_list = []
    
    result = {}
    
    with open(filepath, 'r', encoding='utf8') as fp:
        json_data = json.load(fp)
        
        if len(json_data) == 2:
            json_data_main = json_data[0]
            json_data_remain = json_data[1]
        for data in json_data_main:
            if data["layerName"] == "Human":
                human_add_list.append(data["bound"])
            else:
                length = calculate_std_length(data["bound"])
                for std_length in height_list.values(): 
                    if abs(length - std_length) < threshold:
                        main_bbox_dict[count] = data["bound"]
                        count += 1
        
        if len(main_bbox_dict) == 0:
            raise ValueError("未检测到图框, 请检查图框类型和尺寸.")
        
        
        for data in json_data_main:
            if data["layerName"] != "Split" or data["layerName"] != "split" or data["layerName"] != "SPLIT":
                already_add = False
                for human_index, human_bbox in enumerate(human_add_list):
                    if is_fully_contained_strictly(human_bbox, data["bound"]):
                        tmp = component(data["type"], data["bound"], data)
                        tmp.manual_index = human_index
                        for i, main_bbox in main_bbox_dict.items():
                            if is_fully_contained_strictly(main_bbox, data["bound"]):
                                if i in result and human_index in result[i]["human"]:
                                    result[i]["human"][human_index].push(tmp)
                                elif i in result and human_index not in result[i]["human"]:
                                    result[i]["human"][human_index] = all_components()
                                    result[i]["human"][human_index].push(tmp)
                                else:
                                    result[i] = {"main":all_components(), "human":{}}
                                    result[i]["human"][human_index] = all_components()
                                    result[i]["human"][human_index].push(tmp)
                        already_add = True
                if already_add == False:
                    for i, data_bbox in main_bbox_dict.items():
                        if is_fully_contained_strictly(data_bbox, data["bound"]) or deal_axis(data_bbox, data["bound"]):
                        
                            if deal_axis(data_bbox, data["bound"]):
                                data["bound"]['x1'] = data_bbox['x1']
                                data["bound"]['x2'] = data_bbox['x2']
                            
                            tmp = component(data["type"], data["bound"], data)
                            if i in result:
                                result[i]["main"].push(tmp)
                            else:
                                result[i] = {"main":all_components(), "human":{}}
                                result[i]["main"].push(tmp)
        
    return result, json_data_remain

# 过滤包围盒
def filter_bounding_boxes(components: all_components, threshold=0.1):
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = float('-inf'), float('-inf')
    
    for component in components.component_list:
        x_min = min(x_min, component.bbox['x1'])
        y_min = min(y_min, component.bbox['y1'])
        x_max = max(x_max, component.bbox['x2'])
        y_max = max(y_max, component.bbox['y2'])

    max_area = (x_max - x_min) * (y_max - y_min)
    
    filterd_component_list = all_components()
    for component in components.component_list:
        if calculate_area(component.bbox) < max_area * threshold:
            filterd_component_list.push(component)
    # print(x_max, x_min, y_max, y_min)
    # 过滤面积小于最大面积阈值的包围盒
    return filterd_component_list, x_min, y_min, x_max, y_max

#获取可视化的位置
def get_visualize_bbox(components):
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = float('-inf'), float('-inf')
    
    if type(components) == list:
        for component in components:
            for comp in component:
                x_min = min(x_min, comp.bbox['x1'])
                y_min = min(y_min, comp.bbox['y1'])
                x_max = max(x_max, comp.bbox['x2'])
                y_max = max(y_max, comp.bbox['y2'])
    else:
        for comp in components:
            x_min = min(x_min, comp.bbox['x1'])
            y_min = min(y_min, comp.bbox['y1'])
            x_max = max(x_max, comp.bbox['x2'])
            y_max = max(y_max, comp.bbox['y2'])
    
    return x_min, y_min, x_max, y_max

#计算垂直距离
def calc_vertical_distance(comp1, comp2):

    vertical_distance = float("inf")
    # 判断 x 轴范围是否有重叠
    if comp1.bbox["x2"] >= comp2.bbox["x1"] and comp1.bbox["x1"] <= comp2.bbox["x2"]:
        # 如果在竖直方向上 bbox 在当前 bbox 之下
        if comp2.bbox["y1"] > comp1.bbox["y2"]:
            vertical_distance = comp2.bbox["y1"] - comp1.bbox["y2"]
        # 如果在竖直方向上 bbox 在当前 bbox 之上
        elif comp2.bbox["y1"] < comp1.bbox["y2"]:
            vertical_distance = comp1.bbox["y2"] - comp2.bbox["y1"]
        else:
            vertical_distance = 0  # 它们在竖直方向上已经接触


    return vertical_distance

#计算垂直向下的距离
def calc_only_down_distance(comp1, comp2):
    vertical_distance = float("inf")
    # 判断 x 轴范围是否有重叠
    if comp1.bbox["x2"] > comp2.bbox["x1"] and comp1.bbox["x1"] < comp2.bbox["x2"]:
        # 如果在竖直方向上 bbox 在当前 bbox 之下
        if comp2.bbox["y1"] <= comp1.bbox["y2"]:
            vertical_distance = comp1.bbox["y2"] - comp2.bbox["y1"]
        
    return vertical_distance

#计算垂直向下的距离
def calc_only_down_distance_strict(comp1, comp2):
    vertical_distance = float("inf")
    # 判断 x 轴范围是否有重叠
    if comp1.bbox["x2"] > comp2.bbox["x1"] and comp1.bbox["x1"] < comp2.bbox["x2"]:
        # 如果在竖直方向上 bbox 在当前 bbox 之下
        if comp2.bbox["y2"] <= comp1.bbox["y2"]:
            vertical_distance = comp1.bbox["y2"] - comp2.bbox["y2"]
        
    return vertical_distance

#计算垂直向上的距离
def calc_only_up_distance_on_bbox(box1, box2):
    
    vertical_distance = float("inf")
    if is_fully_contained_strictly(box2, box1):
        return 0
    
    # 判断 x 轴范围是否有重叠
    if box1["x2"] > box2["x1"] and box1["x1"] < box2["x2"]:
       
        # 如果在竖直方向上 bbox 在当前 bbox 之下
        if box2["y1"] > box1["y2"]:
            vertical_distance = box2["y1"] - box1["y2"]
        elif box2["y2"] < box1["y2"]:
            pass
        else:
            vertical_distance = 0
        
    return vertical_distance

#找到下方最近的非text元素
def find_nearest_down_none_text(comp: component, comp_list: all_components, jump_handle=[], threshold=None):
     #根据y坐标找最近的包围盒
    min_distance = float("inf")
    nearest_comp = None
    nearest_index = 0
    for i in range(len(comp_list)):
        
        if comp_list[i].data["type"] == "text" or comp_list[i].data["type"] == "mtext" or comp_list[i].data["type"] == "solid" or \
            comp_list[i].data["handle"] in jump_handle or \
            comp_list[i].data["type"] == "insert" or \
            (comp_list[i].type in jump_polyline and calculate_area(comp_list[i].bbox) > 4e6):
            continue
        
        distance = calc_only_down_distance(comp, comp_list[i])
        if  distance > 1 and distance < min_distance and comp.data["handle"] != comp_list[i].data["handle"]:
            min_distance = distance
            nearest_comp = comp_list[i]
            nearest_index = i
    
    if threshold is not None:
        if min_distance > threshold:
            return None, None
    
    return nearest_comp, nearest_index
        


# 可视化包围盒
def visualize_bounding_boxes(final_filtered_comp_list: all_components, x_min, y_min, x_max, y_max, output_path):
    plt.figure()
    ax = plt.gca()

    # 绘制每个包围盒
    for component in final_filtered_comp_list.component_list:
        rect = Rectangle((component.bbox['x1'], component.bbox['y1']), component.bbox['x2'] - component.bbox['x1'], component.bbox['y2'] - component.bbox['y1'],
                         linewidth=0.25, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)

    # 绘制每个组的最大包围盒
    for group, bbox in final_filtered_comp_list.groups.items():
        rect = Rectangle((bbox['x1'], bbox['y1']), bbox['x2'] - bbox['x1'], bbox['y2'] - bbox['y1'],
                         linewidth=1, edgecolor='red', facecolor='none')
        # print(rect)
        ax.add_patch(rect)
    
    # 设置x轴和y轴的显示范围
    try:
        ax.set_xlim(x_min - 10000, x_max + 10000)
        ax.set_ylim(y_min - 10000, y_max + 10000)
    except:
        raise ValueError("未检测到图框, 请检查图框类型和尺寸.")

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Segmentation Result')
    plt.savefig(output_path)

#可视化多级剖图
def visualize_multi(bbox_list, output_path, x_min, y_min, x_max, y_max):
    plt.figure()
    ax = plt.gca()
    
    for bbox, success in bbox_list:
        
        rect = Rectangle((bbox['x1'], bbox['y1']), bbox['x2'] - bbox['x1'], bbox['y2'] - bbox['y1'],
                         linewidth=0.25, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)
    
     # 设置x轴和y轴的显示范围
    try:
        ax.set_xlim(x_min - 10000, x_max + 10000)
        ax.set_ylim(y_min - 10000, y_max + 10000)
    except:
        raise ValueError("未检测到图框, 请检查图框类型和尺寸.")
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Segmentation Result')
    plt.savefig(output_path)  

def visualize_many_bbox(components_list, x_min, y_min, x_max, y_max, output_path, color='red'):
    plt.figure()
    ax = plt.gca()
    components_list = [item for sublist in components_list for item in (sublist if isinstance(sublist, list) else [sublist])]
    for components in tqdm(components_list):
         # 绘制每个包围盒
        for component in components:
            rect = Rectangle((component.bbox['x1'], component.bbox['y1']), component.bbox['x2'] - component.bbox['x1'], component.bbox['y2'] - component.bbox['y1'],
                            linewidth=0.5, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)

        # 绘制每个组的最大包围盒
        for group, bbox in components.groups.items():
            rect = Rectangle((bbox['x1'], bbox['y1']), bbox['x2'] - bbox['x1'], bbox['y2'] - bbox['y1'],
                            linewidth=1, edgecolor=color, facecolor='none')
            # print(rect)
            ax.add_patch(rect)

    # 设置x轴和y轴的显示范围
    try:
        ax.set_xlim(x_min - 10000, x_max + 10000)
        ax.set_ylim(y_min - 10000, y_max + 10000)
    except:
        raise ValueError("未检测到图框, 请检查图框类型和尺寸.")

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Segmentation Result')
    plt.savefig(output_path)

#获取矩形包围盒
def get_bbox(components_list: list):
    bbox_list = []
    for components in components_list:
        bbox = components.get_final_bbox()
        
        bbox_list.extend(bbox)
        
    return bbox_list

#获取多边形包围盒
def get_polygon_bbox(components_list: list):
    bbox_list = []
    for components in components_list:
        poly = components.get_polygon_bbox()
        
        bbox_list.append(poly)
        
    return bbox_list

#保存为json（不需要）
def save_to_json(components_list: list, file_path: str):
    add_data = None
    with open(file_path, 'r', encoding='utf8') as fp:
        json_data = json.load(fp)
        if len(json_data) == 2:
            add_data = json_data[1]
            
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    folder_name = os.path.join(os.path.dirname(file_path), file_name)
    create_path(folder_name)
    
    count = 0
    for components in components_list:
        segment_list = components.return_result()
        for segment in segment_list:
            
            save_name = file_name + '_' + str(count) +'.json'
            
            if add_data != None:
                segment = [segment, add_data]
            # 将数据保存为 JSON 文件
            with open(os.path.join(folder_name, save_name), "w", encoding='utf-8') as json_file:
                json.dump(segment, json_file, ensure_ascii=False, indent=4)  # ensure_ascii=False 用于保存中文，indent=4 格式化输出
            
            count += 1

#判断是否为标题的文本
def judge_text(comp: component):
    return comp != None and \
        ((comp.data["type"] == "text" or (comp.data["type"] == "mtext") and \
        (comp.data["content"][0] != '+' and comp.data["content"][0] != '-')) \
        and comp.data["color"] == 3)

#判断是否为标题下方的绿线
def judge_line_under_title(comp: component, comp_list: all_components):

    if comp != None and comp.data["color"] == 3 and comp.data["type"] == "line":

        intersect_time = 0
        intersect_line = []
        for i in range(len(comp_list)):
            if (comp_list[i].type == "line" or comp_list[i].type == "lwpolyline") and comp_list[i].data["handle"] != comp.data["handle"]:
                intersect = is_intersecting(comp.bbox, comp_list[i].bbox)
                if intersect == True and calculate_area(comp_list[i].bbox) < 300000:
                    intersect_time += 1
                    intersect_line.append(comp_list[i])

        if intersect_time == 1 and abs(intersect_line[0].bbox["y2"] - comp.bbox["y2"] < 200):
            return True, intersect_line
        elif intersect_time == 0 or intersect_time == 2:
            return True, intersect_line
        else:
            return False, []

    return comp != None and (((comp.data["type"] == "line" or comp.data["type"] == "lwpolyline") and comp.data["color"] == 3) or comp.type == "leader"), []

#判断标题，并根据标题进行bfs搜索子图
def judge_title(index: int, comp: component, comp_list: all_components):
    
    if judge_text(comp):
        
        nearest_down_line, _ = find_nearest_down_none_text(comp, comp_list, threshold=1000)

        is_line, intersect_list = judge_line_under_title(nearest_down_line, comp_list)

        if is_line or ("content" in comp.data.keys() and len(comp.data["content"]) >= 3 and ("%%U" in comp.data["content"] or "%%u" in comp.data["content"])) or comp.data["type"] == "mtext" :

            
            comp.title = comp.data["content"].strip()
            
            if "%%U" in comp.data["content"] or "%%u" in comp.data["content"]: #特殊情况判断
                if "%%U" in comp.data["content"]:
                    comp.title = comp.title.split("%%U")[1].strip()
                else:
                    comp.title = comp.title.split("%%u")[1].strip()
                    
            print(comp.title)
            print(comp.data["handle"])

            if ("content" in comp.data.keys() and len(comp.data["content"]) >= 3 and ("%%U" in comp.data["content"] or "%%u" in comp.data["content"])) : #特殊情况判断
                nearest_down_line = comp

            
            
            start_comp, start_index = find_nearest_down_none_text(nearest_down_line, comp_list, [i.data["handle"] for i in intersect_list])
            if start_comp == None:
                return None, -1
            
            #子图搜索
            sub_comps, index = bfs_by_comp_main(comp.title, index, start_index, comp_list)
            if nearest_down_line is not None:
                get_sub_title_content(nearest_down_line, sub_comps)
            if (comp.data["handle"]=="AF564"):
                print("debug")
                print(sub_comps.sub_title)
            return sub_comps, index
    return None, -1
    


def search_titles(comp_list: all_components):
    
    n = len(comp_list)
    uf = UnionFind(n)
    comp_list.bind_uf(uf)
    sub_comps_list = []
    research_indexs = []
    
    #添加标题中的+/-
    deal_plus_and_minus(comp_list)

    for i in range(n):
        sub_comps, index = judge_title(i, comp_list[i], comp_list)
        
        if sub_comps is not None:
            sub_comps_list.append(sub_comps)
        
        if index != -1:
            research_indexs.append(index)
    
    return sub_comps_list, research_indexs

#添加标题中的+/-
def deal_plus_and_minus(comp_list):
    for comp in comp_list:
        if comp != None and \
        ((comp.data["type"] == "text" or comp.data["type"] == "mtext")) and \
        (comp.data["content"][0] == '+' or comp.data["content"][0] == '-') \
        and comp.data["color"] == 3:
            threshold = 500
            min_distance = float("inf")
            nearest_comp = None
            nearest_index = 0

            for i in range(len(comp_list)):
                if comp_list[i].data["type"] == "text" and comp_list[i].data["color"] == 3 and comp_list[i].data["handle"] != comp.data["handle"]:
                    distance = calc_distance_on_bbox(comp.bbox, comp_list[i].bbox)

                    if distance < min_distance:
                        min_distance = distance
                        nearest_comp = comp_list[i]
                        nearest_index = i
            
            if min_distance < threshold:
                nearest_comp.data["content"] = nearest_comp.data["content"] + comp.data["content"]

def deal_plus_and_minus_in_sub_title(comp_list):
    for comp in comp_list:
        if comp != None and \
        ((comp.data["type"] == "text" or comp.data["type"] == "mtext")) and \
        (comp.data["content"][0] == '+' or comp.data["content"][0] == '-'):
            threshold = 500
            min_distance = float("inf")
            nearest_comp = None
            nearest_index = 0

            for i in range(len(comp_list)):
                if comp_list[i].data["type"] == "text" and comp_list[i].data["color"] == comp.data["color"] and comp_list[i].data["handle"] != comp.data["handle"]:
                    distance = calc_distance_on_bbox(comp.bbox, comp_list[i].bbox)

                    if distance < min_distance:
                        min_distance = distance
                        nearest_comp = comp_list[i]
                        nearest_index = i
            
            if min_distance < threshold:
                nearest_comp.data["content"] = nearest_comp.data["content"] + comp.data["content"]

#根据标题bfs搜索子图
def bfs_by_comp_main(title: str, title_index: int, start_index: int, comp_list: all_components):
    
    n = len(comp_list)
    
    visited = [False] * n
    result = []
    queue = deque([start_index, title_index])
    
    result.append(start_index)
    result.append(title_index)
    comp_list[start_index].is_merged = True
    comp_list[title_index].is_merged = True
    visited[title_index] = True
    
    while queue:
        current = queue.popleft()
        visited[current] = True
        is_polyline = False
        if comp_list[current].type in jump_polyline and calculate_area(comp_list[current].bbox) > 4e6: #如果是较长多段线则跳过（较长多段线bbox存在问题，不能直接判断）
            is_polyline = True
        for i in range(n): 
            if is_polyline or (comp_list[i].type in jump_polyline and calculate_area(comp_list[i].bbox) > 4e6):
                if not visited[i] and judge_intersect(comp_list[current], comp_list[i]):
                    comp_list[i].is_merged = True
                    visited[i] = True
                    queue.append(i)
                    result.append(i)
            else:
                if not visited[i] and (is_nearby(comp_list[current].bbox, comp_list[i].bbox, threshold=100) or \
                                       judge_intersect(comp_list[current], comp_list[i])):
                    comp_list[i].is_merged = True
                    visited[i] = True
                    queue.append(i)
                    result.append(i)
                    
    sub_comps = all_components(value=[comp_list[i] for i in result], title=title)
    # visualize_bounding_boxes(sub_comps, *get_visualize_bbox(sub_comps), "../result/test.png")
    # exit()
    if len(result) < 10:
        return sub_comps, start_index
    return sub_comps, -1
    

    
def sort_comp_list_by_max_area(comps: all_components):
    max_area = 0
    bboxs = comps.get_final_bbox()
    for bbox in bboxs:
        max_area = max(max_area, calculate_area(bbox))

    return max_area

#comps_list去重
def remove_duplicate_comps(comps_list: list):
    filtered_comps_list = []
    max_area = 0
    comps_list.sort(key = lambda x:sort_comp_list_by_max_area(x), reverse=True)
    for i in range(len(comps_list)):
        if not check_bbox(filtered_comps_list, comps_list[i]):
            filtered_comps_list.append(comps_list[i])
    
    return filtered_comps_list

def check_bbox(comps_list: list, comps: all_components):
    
    bbox1 = comps.get_final_bbox()
    for i in range(len(comps_list)):
        bbox2 = comps_list[i].get_final_bbox()
        for coord1 in bbox1:
            for coord2 in bbox2:
                # 比较每个键的值是否相同
                if (coord1['x1'] == coord2['x1'] and coord1['y1'] == coord2['y1']) or \
                    (coord1['x2'] == coord2['x2'] and coord1['y2'] == coord2['y2']):
                    return True
                
    return False

#从一点出发找到最近的comp
def search_nearest_comp_from_point(point: list, ori_components: all_components, arrow: component):
    min_distance = float("inf")
    index = -1

    for i in range(len(ori_components)):
        center_x = (ori_components[i].data["bound"]["x1"] + ori_components[i].data["bound"]["x2"]) / 2
        center_y = (ori_components[i].data["bound"]["y1"] + ori_components[i].data["bound"]["y2"]) / 2
        if ori_components[i].data["handle"] == arrow.data["handle"]:
            continue
        distance = cal_eucilean_distance(point, [center_x, center_y])
        
        if distance < min_distance:
            min_distance = distance
            index = i
    
    if index != -1 and min_distance < 2000:
        return ori_components[index], index

    return None, -1

#根据箭头寻找子图
def search_zitu_by_arrow(arrow: component, ori_components: all_components, human_components: dict):
    end_point = [arrow.data["vertices"][0][0], arrow.data["vertices"][0][1]]
    comp, comp_index = search_nearest_comp_from_point(end_point, ori_components, arrow)
    if comp is not None:
        if comp.manual_index == 0:
            zitu = bfs_by_arrow(comp_index, ori_components, arrow)
        else:
            zitu = human_components[comp.manual_index]
            # visualize_bounding_boxes(zitu, *get_visualize_bbox(zitu),"test.png")
        return zitu
    
#根据箭头末端进行bfs
def bfs_by_arrow(start_index: int, comp_list: all_components, arrow: component):
    n = len(comp_list)
    
    visited = [False] * n
    result = []
    queue = deque([start_index])
    result.append(start_index)
    
    while queue:
        current = queue.popleft()
        visited[current] = True
        is_polyline = False
        if comp_list[current].type in jump_polyline and calculate_area(comp_list[current].bbox) > 4e6:
            is_polyline = True
        for i in range(n): 

            if comp_list[i].data["handle"] == arrow.data["handle"]:
                continue

            if is_polyline or (comp_list[i].type in jump_polyline and calculate_area(comp_list[i].bbox) > 4e6):
                if not visited[i] and judge_intersect(comp_list[current], comp_list[i]):
                    visited[i] = True
                    queue.append(i)
                    result.append(i)
            else:
                if not visited[i] and (is_nearby(comp_list[current].bbox, comp_list[i].bbox, threshold=200) or \
                                       judge_intersect(comp_list[current], comp_list[i])):
                    visited[i] = True
                    queue.append(i)
                    result.append(i)
                    
    sub_comps = all_components(value=[comp_list[i] for i in result])
    # visualize_bounding_boxes(sub_comps, *get_visualize_bbox(sub_comps), "../result/test.png")
    # exit()
    return sub_comps


#############################0703新需求
##############0711
def distance_from_point_to_bbox(point, bbox):
    """计算点到边界框的最小距离"""
    px, py = point[0], point[1]
    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
    
    # 计算点到矩形的最小距离
    dx = max(abs(x1 - px), 0, abs(px - x2))
    dy = max(abs(y1 - py), 0, abs(py - y2))
    return (dx * dx + dy * dy) ** 0.5
##############0711

def search_zitu_by_arrow_special(arrow: component, ori_components: all_components, human_components: dict):
    end_point = [arrow.data["vertices"][0][0], arrow.data["vertices"][0][1]]
    comp, comp_index = search_nearest_comp_from_point(end_point, ori_components, arrow)
    if comp is not None:
        if comp.manual_index == 0:
            zitu = bfs_by_arrow_special(comp_index, ori_components, arrow)
        else:
            zitu = human_components[comp.manual_index]
            # visualize_bounding_boxes(zitu, *get_visualize_bbox(zitu),"test.png")
        return zitu
    


##############0711
#根据箭头末端进行bfs
def bfs_by_arrow_special(start_index: int, comp_list: all_components, arrow: component):
    n = len(comp_list)
    arrow_end = [arrow.data["vertices"][0][0], arrow.data["vertices"][0][1]]
    MAX_DISTANCE = 1000  # 限制距离为1000
    visited = [False] * n
    result = []
    queue = deque([start_index])
    result.append(start_index)
    
    while queue:
        current = queue.popleft()
        visited[current] = True
        is_polyline = False
        if comp_list[current].type in jump_polyline and calculate_area(comp_list[current].bbox) > 4e6:
            is_polyline = True
        for i in range(n): 

            if comp_list[i].data["handle"] == arrow.data["handle"]:
                continue

            # 添加距离检查


            if is_polyline or (comp_list[i].type in jump_polyline and calculate_area(comp_list[i].bbox) > 4e6):
                if not visited[i] and judge_intersect(comp_list[current], comp_list[i]):
                    visited[i] = True
                    queue.append(i)
                    result.append(i)
            else:
                if not visited[i] and (is_nearby(comp_list[current].bbox, comp_list[i].bbox, threshold=100) or \
                                       judge_intersect(comp_list[current], comp_list[i])):
                    
                    if distance_from_point_to_bbox(arrow_end, comp_list[i].bbox) > MAX_DISTANCE:
                        continue
                    else:
                        print("debug_arrow:{}".format(comp_list[i].data["handle"]))
                        print("distance:{}".format(distance_from_point_to_bbox(arrow_end, comp_list[i].bbox)))
                        print("arrow_end:{}".format(arrow_end))
                    visited[i] = True
                    queue.append(i)
                    result.append(i)
                    if arrow.data["handle"]=="AF7DE":
                        print("debug_arrow:{}".format(comp_list[i].data["handle"]))
                    
    sub_comps = all_components(value=[comp_list[i] for i in result])
    
    # visualize_bounding_boxes(sub_comps, *get_visualize_bbox(sub_comps), "../result/test.png")
    # exit()
    return sub_comps
##################################################
##############0711



#bfs其余未归并的子图
def search_unmerged_zitu(start_index: int, comp_list: all_components):
    n = len(comp_list)
    
    visited = [False] * n
    result = []
    queue = deque([start_index])
    result.append(start_index)
    comp_list[start_index].is_merged = True
 
    
    while queue:
        current = queue.popleft()
        visited[current] = True
        is_polyline = False
        if comp_list[current].type in jump_polyline and calculate_area(comp_list[current].bbox) > 4e6:
            is_polyline = True
        for i in range(n): 
            if is_polyline or (comp_list[i].type in jump_polyline and calculate_area(comp_list[i].bbox) > 4e6):
                if not visited[i] and judge_intersect(comp_list[current], comp_list[i]):
                    comp_list[i].is_merged = True
                    visited[i] = True
                    queue.append(i)
                    result.append(i)
            else:
                if not visited[i] and (is_nearby(comp_list[current].bbox, comp_list[i].bbox, threshold=200) or \
                                       judge_intersect(comp_list[current], comp_list[i])):
                    comp_list[i].is_merged = True
                    visited[i] = True
                    queue.append(i)
                    result.append(i)
                    
    sub_comps = all_components(value=[comp_list[i] for i in result])
    # visualize_bounding_boxes(sub_comps, *get_visualize_bbox(sub_comps), "../result/test.png")
    # exit()
    return sub_comps

#归并到最近的bbox
def merge_to_nearest_bbox(comps: all_components, unmerged_components_list: list):

    min_distance = float("inf")
    index = -1
    comps_bbox = comps.get_final_bbox()[0]

    for i in range(len(unmerged_components_list)):
        other_bbox = unmerged_components_list[i].get_final_bbox()[0]
        # distance = calc_only_up_distance_on_bbox(comps_bbox, other_bbox)
        distance = calc_distance_on_bbox(comps_bbox, other_bbox)

        if distance < min_distance:
            min_distance = distance
            index = i

    if index != -1 and min_distance < 1e5:
        # print(min_distance)
        # visualize_bounding_boxes(unmerged_components_list[index], *get_visualize_bbox(unmerged_components_list[index]),"final.png")

        unmerged_components_list[index].merge(comps)



#无主子图归并主函数
def merge_nearest_comp(ori_components_list: list, filtered_all_comps: all_components, research_index):
    
    #两种方法都使用
    #1.归并到最近的bbox
    # for index in research_index:
    #     unmerged_new_comps_without_title = search_unmerged_zitu(index, filtered_all_comps)
    #     merge_to_nearest_bbox(unmerged_new_comps_without_title, ori_components_list)

    #2.归并到最近的归并好的子图
    for i in range(len(filtered_all_comps)):
        if filtered_all_comps[i].is_merged == False:
            
            unmerged_new_comps_without_title = search_unmerged_zitu(i, filtered_all_comps)
            # visualize_bounding_boxes(unmerged_new_comps_without_title, *get_visualize_bbox(unmerged_new_comps_without_title),"test{}.png".format(i))
            
            if len(unmerged_new_comps_without_title) > 1 or unmerged_new_comps_without_title[0].type == "insert" or \
            unmerged_new_comps_without_title[0].type == "text" or \
            unmerged_new_comps_without_title[0].type == "mtext":
                merge_to_nearest_merged_comp(unmerged_new_comps_without_title, ori_components_list)
    #
    for i in range(len(ori_components_list)):
        if ori_components_list[i].title != "" and ori_components_list[i].title is not None:
        # 根据title找comp
            title_comp= None
            for j in range(len(ori_components_list[i].component_list)):
                if ori_components_list[i].component_list[j].data["type"] == "text" and\
                ori_components_list[i].component_list[j].data["content"] == ori_components_list[i].title:
                    title_comp = ori_components_list[i].component_list[j]
                    break
            if title_comp is not None:
                nearest_down_line, _ = find_nearest_down_none_text(title_comp, ori_components_list[i], threshold=1000)
                if nearest_down_line is not None:
                    get_sub_title_content(nearest_down_line, ori_components_list[i])
    #

    return ori_components_list

#归并到最近的归并好的子图
def merge_to_nearest_merged_comp(comps_list: all_components, ori_components_list: list):
    min_distance = float("inf")
    nearest_components = None

    bbox = comps_list.get_final_bbox()[0]

    for components in ori_components_list:
        for i in range(len(components)):
            if components[i].type in jump_polyline and calculate_area(components[i].bbox) > 4e6:
                continue
            if components[i].is_merged == True and \
            not is_fully_contained_strictly(bbox, components[i].bbox) and \
            not is_fully_contained_strictly(components[i].bbox, bbox):
                
                distance = calc_distance_on_bbox(bbox, components[i].bbox)

                if distance < min_distance:
                    min_distance = distance
                    nearest_components = components
    if min_distance < 1e5:
        nearest_components.merge(comps_list)

def solve_big_contain_small(components_list: list):
    """
    从 components_list 中两两比较 all_components 进行去重。
    """
    n = len(components_list)
    for i in range(n):
        for j in range(i + 1, n):
            A, B = components_list[i], components_list[j]
            
            # 创建 bbox 映射到 component 的字典
            bbox_to_component = {}
            
            for comp in A.component_list:
                bbox_to_component[comp.data["handle"]] = ('A', comp)
            # print(bbox_to_component)
            # assert 0
            for comp in B.component_list:
                bbox_tuple = comp.data["handle"]
                if bbox_tuple in bbox_to_component:
                    if len(A) <= len(B):
                        continue  # 保留在 A 中
                    else:
                        bbox_to_component[bbox_tuple] = ('B', comp)  # 替换为 B 中的 component
                else:
                    bbox_to_component[bbox_tuple] = ('B', comp)
            
            # 重新整理 component_list
            A.component_list = [comp for owner, comp in bbox_to_component.values() if owner == 'A']
            B.component_list = [comp for owner, comp in bbox_to_component.values() if owner == 'B']
            
            # 重新初始化 UnionFind 结构
            A.uf = UnionFind(len(A.component_list))
            B.uf = UnionFind(len(B.component_list))
            A.uf.init_by_one()
            B.uf.init_by_one()
            
            A.update_groups()
            B.update_groups()
    
    return components_list


#获取comp的起点和终点
def get_start_and_end_point(comp: component):

    if "start" in comp.data.keys() and "end" in comp.data.keys():
        start_point = comp.data["start"]
        end_point = comp.data["end"]

        return start_point, end_point
    elif "vertices" in comp.data.keys():
        start_point_list = []
        end_point_list = []
        for i in range(len(comp.data["vertices"])):
            if len(comp.data["vertices"][i]) == 5:
                v = comp.data["vertices"][i]
                x, y, s_a, e_a, r = v[0], v[1], v[2], v[3], v[4]
                s_a = s_a / math.pi * 180
                e_a = e_a / math.pi * 180
                if s_a > e_a:
                    e_a += 360
                x_new = x + r * math.cos(s_a * math.pi / 180)
                y_new = y + r * math.cos(s_a * math.pi / 180)
                start_point = [x_new, y_new]
                x_new = x + r * math.cos(e_a * math.pi / 180)
                y_new = y + r * math.cos(e_a * math.pi / 180)
                end_point = [x_new, y_new]
                
                start_point_list.append(start_point)
                end_point_list.append(end_point)
                
            elif len(comp.data["vertices"]) == 4:
                start_point = [comp.data["vertices"][i][0], comp.data["vertices"][i][1]]
                end_point = [comp.data["vertices"][i][2], comp.data["vertices"][i][3]]
                
                start_point_list.append(start_point)
                end_point_list.append(end_point)
                
        return start_point_list, end_point_list
    else:
        start_point = (comp.bbox["x1"], comp.bbox["y1"])
        end_point = (comp.bbox["x2"], comp.bbox["y2"])

        return start_point, end_point

#叉乘
def cross_product(v1, v2):
        return v1[0] * v2[1] - v1[1] * v2[0]

#将直线朝两端延长一小部分
def extend_line_segment(p1, p2, extend_length=10):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    origin_length = math.hypot(dx, dy)
    
    if origin_length == 0:
        return p1, p2
    
    unit_vector = (dx / origin_length, dy / origin_length)
    
    new_p1 = (p1[0] - unit_vector[0] * extend_length, p1[1] - unit_vector[1] * extend_length)
    new_p2 = (p2[0] + unit_vector[0] * extend_length, p2[1] + unit_vector[1] * extend_length)
    
    return new_p1, new_p2

#判断两直线相交
def judge_intersect_on_line(p1, p2, p3, p4, epsilon=1e-9):
    
    p1, p2 = extend_line_segment(p1 ,p2)
    p3, p4 = extend_line_segment(p3, p4)
    
    if max(p1[0], p2[0]) < min(p3[0], p4[0]) or \
       max(p1[1], p2[1]) < min(p3[1], p4[1]) or \
       max(p3[0], p4[0]) < min(p1[0], p2[0]) or \
       max(p3[1], p4[1]) < min(p1[1], p2[1]):
        return False

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = cross(p1, p2, p3)
    d2 = cross(p1, p2, p4)
    d3 = cross(p3, p4, p1)
    d4 = cross(p3, p4, p2)

    if (d1 * d2 < 0) and (d3 * d4 < 0):
        return True

    def on_segment(a, b, c):
        return (min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and 
                min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))

    if d1 == 0 and on_segment(p1, p2, p3):
        return True
    if d2 == 0 and on_segment(p1, p2, p4):
        return True
    if d3 == 0 and on_segment(p3, p4, p1):
        return True
    if d4 == 0 and on_segment(p3, p4, p2):
        return True

    return False

#判断两个comp相交，可能是直线，也可能是多段线（list），是多段线则进行分别判断，只要有一处相交即为相交
def judge_intersect(comp1: component, comp2: component):
    start_1, end_1 = get_start_and_end_point(comp1)
    start_2, end_2 = get_start_and_end_point(comp2)
    if len(start_1) == 0 or len(start_2) == 0:
        return False
    if isinstance(start_1[0], list) and isinstance(start_2[0], list):
        for i in range(len(start_1)):
            line1_start = start_1[i]
            line1_end = end_1[i]
            for j in range(len(start_2)):
                line2_start = start_2[j]
                line2_end = end_2[j]
                if judge_intersect_on_line(line1_start, line1_end, line2_start, line2_end):
                    return True
        return False
    elif isinstance(start_1[0], list):
        line2_start, line2_end = start_2, end_2
        for i in range(len(start_1)):
            line1_start = start_1[i]
            line1_end = end_1[i]
            if judge_intersect_on_line(line1_start, line1_end, line2_start, line2_end):
                return True
        
        return False
    
    elif isinstance(start_2[0], list):
        line1_start, line1_end = start_1, end_1
        for i in range(len(start_2)):
            line2_start = start_2[i]
            line2_end = end_2[i]
            if judge_intersect_on_line(line1_start, line1_end, line2_start, line2_end):
                return True
        
        return False
    else:
        line1_start, line1_end = start_1, end_1
        line2_start, line2_end = start_2, end_2
        
        return judge_intersect_on_line(line1_start, line1_end, line2_start, line2_end)
    
    
#将人工重新框选结果合并到程序中
def merge_human_result(ori_components_list, human_components, ori_components):
    
    
    for components in human_components.values():
        title = None
        nearest_down_line = None
        for comp in components:
            if judge_text(comp):
        
                nearest_down_line, _ = find_nearest_down_none_text(comp, components, threshold=1000)

                is_line, intersect_list = judge_line_under_title(nearest_down_line, components)

                if is_line or ("content" in comp.data.keys() and len(comp.data["content"]) >= 3 and ("%%U" in comp.data["content"] or "%%u" in comp.data["content"])) or comp.data["type"] == "mtext" :

                    
                    comp.title = comp.data["content"].strip()
                    if "%%U" in comp.data["content"] or "%%u" in comp.data["content"]:
                        if "%%U" in comp.data["content"]:
                            comp.title = comp.title.split("%%U")[1].strip()
                        else:
                            comp.title = comp.title.split("%%u")[1].strip()
                    title = comp.title
                    print("人工加入标题: ", title)
                    break
        new_comps = all_components(value=components.component_list, title=title)
        if nearest_down_line is not None:
            get_sub_title_content(nearest_down_line, new_comps)
        ori_components_list.append(new_comps)
    
    
    #加入到原始components，以实现箭头指向搜寻
    for components in human_components.values():
        components.update_after_merge()
        for comp in components:
            ori_components.push(comp)

    return ori_components_list, ori_components

#找下方最近的comp
def find_nearest_down(comp: component, comp_list: all_components, threshold=None):
    min_distance = float("inf")
    nearest_comp = None
    for i in range(len(comp_list)):
        
        distance = calc_only_down_distance_strict(comp, comp_list[i])
        
        if  distance > 1 and distance < min_distance and comp.data["handle"] != comp_list[i].data["handle"]:
            min_distance = distance
            nearest_comp = comp_list[i]
    if threshold is not None:
        if min_distance > threshold:
            return None, min_distance

    return nearest_comp, min_distance

def get_sub_title_content(title_line: component, comp_list: all_components, threshold=400):
    
    #副标题的角标需要该副标题关联
    deal_plus_and_minus_in_sub_title(comp_list)
    
    queue = deque([title_line])
    visited = []
    sub_title = []
    while queue:
        cur_comp = queue.popleft()
        nearest_down_under_cur_comp, min_distance = find_nearest_down(cur_comp, comp_list, threshold=600)
        #
        if nearest_down_under_cur_comp is not None and nearest_down_under_cur_comp.data["type"] == "text" \
            and nearest_down_under_cur_comp.data["handle"] not in visited:
        #
            #print("找到副标题: ", nearest_down_under_cur_comp.data["content"])
            queue.append(nearest_down_under_cur_comp)
            sub_title.append(nearest_down_under_cur_comp.data["content"])
            if nearest_down_under_cur_comp is not None and \
            nearest_down_under_cur_comp.data["type"] == "leader": #下方有引线的文字需排除
                sub_title.pop()
    comp_list.sub_title = sub_title