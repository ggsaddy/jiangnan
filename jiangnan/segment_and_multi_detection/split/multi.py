from split.src import component, UnionFind, all_components
from split.utils import *
# from split.main import *
import numpy as np
import math
from collections import defaultdict

#剖面存在的图层
poumian_tuceng = ["Plate_Visible", "Plate_Invisible",
                  "Stiffener_Visible", "Stiffener_Invisible",
                  "Girder_Visible", "Girder_Invisible",
                  "Gerder_Invisible", "Gerder_Visible",
                  "Plan_Invisible", "Plan_Visible"]

def visualize_filtered_bbox(filepath, output_path):
    # 加载数据
    components = load_data(filepath)

    # 过滤包圆盒
    filtered_comps, x_min, y_min, x_max, y_max = filter_bounding_boxes(components, threshold=10)

    # 可视化结果
    visualize_bounding_boxes(filtered_comps, x_min, y_min, x_max, y_max, output_path)


#找到最近的绿色文本
def find_nearest_green_text(comp: component, comp_list: all_components, threshold=None):
    min_distance = float("inf")
    nearest_comp = None
    nearest_index = 0

    for i in range(len(comp_list)):
        if comp_list[i].data["type"] == "text" and comp_list[i].data["color"] == 3:
            
            distance = calc_distance_on_bbox(comp.bbox, comp_list[i].bbox)

            if distance < min_distance:
                min_distance = distance
                nearest_comp = comp_list[i]
                nearest_index = i

    if threshold is not None:
        if min_distance > threshold:
            return None, None
    return nearest_comp, nearest_index


def find_nearest_none_text(comp: component, comp_list: all_components, threshold=None):
    min_distance = float("inf")
    nearest_comp = None
    nearest_index = 0
    # 场景三0618
    for i in range(len(comp_list)):
        if comp_list[i].data["type"] != "text" and comp.data["handle"] != comp_list[i].data["handle"] :
            point=[comp.data["vertices"][-1][0], comp.data["vertices"][-1][1]]
            distance = calc_distance_from_bbox_to_point( comp_list[i].bbox,point)

            if distance < min_distance:
                min_distance = distance
                nearest_comp = comp_list[i]
                nearest_index = i

    if threshold is not None:
        if min_distance > threshold:
            return None     #0710None,None修改为None
    return nearest_comp

def find_nearest_xuxiankuang_text(comp: component, comp_list: all_components, threshold=None):
    min_distance = float("inf")
    nearest_comp = None
    nearest_index = 0

    for i in range(len(comp_list)):
        if comp_list[i].data["type"] == "text" :
            
            distance = calc_distance_on_bbox(comp.bbox, comp_list[i].bbox)

            if distance < min_distance:
                min_distance = distance
                nearest_comp = comp_list[i]
                nearest_index = i

    if threshold is not None:
        if min_distance > threshold:
            return None
    return nearest_comp

def find_nearest_xuxiankuang_text_special(comp: component, comp_list: all_components, threshold=500):
    TYPICAL_SIMILAR_TEXT=[
        "SEE","HB","SIM","DET","see"
    ]
    min_distance = float("inf")
    nearest_index = 0
    nearest_comp =[]
    for i in range(len(comp_list)):
        if comp_list[i].data["type"] == "text" :
            
            distance = calc_distance_on_bbox(comp.bbox, comp_list[i].bbox)
            content = comp_list[i].data["content"]
            if distance < threshold:
                #判定content是否出现了关键字
                if any(keyword in content for keyword in TYPICAL_SIMILAR_TEXT):
                    # if comp_list[i].data["color"] == 3 or comp_list[i].data["color"] == 7 : #只考虑绿色和白色文本
                        nearest_comp.append(comp_list[i].data["content"])
    return nearest_comp

#########################################程序改进0702
def find_xuxiankuang_texts_with_parallel_check(comp: component, comp_list: all_components, threshold=500):
    """
    找到与引线近似平行的所有文本
    """
    
    result_texts = None

    # 获取引线的方向向量
    line_direction = get_line_direction(comp)
    if line_direction is None:
        return result_texts
    min_distance = float("inf")
    for i in range(len(comp_list)):
        if comp_list[i].data["type"] == "text":
            # # 检查文本颜色（只考虑绿色和白色文本）
            # if comp_list[i].data["color"] not in [3, 7]:
            #     continue
                
            distance = calc_distance_on_bbox(comp.bbox, comp_list[i].bbox)
            if distance > threshold or  distance > min_distance:
                continue
            
            # 检查引线与文本是否近似平行
            if is_line_parallel_to_text(comp, comp_list[i]):
                result_texts= comp_list[i].data["content"]
                min_distance = distance
    
    return result_texts

def get_line_direction(comp: component):
    """获取引线的方向向量"""
    if comp.data["type"] == "lwpolyline":
        if "vertices" in comp.data and len(comp.data["vertices"]) >= 2:
            start = comp.data["vertices"][0][:2]
            end = comp.data["vertices"][-1][:2]
        else:
            return None
    elif comp.data["type"] == "line":
        if "start" in comp.data and "end" in comp.data:
            start = comp.data["start"]
            end = comp.data["end"]
        else:
            return None
    elif comp.data["type"]=="leader":
        if "vertices" in comp.data and len(comp.data["vertices"]) >= 2:
            start = comp.data["vertices"][0][:2]
            end = comp.data["vertices"][-1][:2]
        else:
            return None
    else:
        return None
    
    # 计算方向向量并归一化
    direction = [start[0] - end[0], start[1] - end[1]]
    length = (direction[0]**2 + direction[1]**2)**0.5
    if length == 0:
        return None

    return [direction[0]/length, direction[1]/length]

def get_text_direction(text_comp: component):
    """根据文本包围盒估算文本方向"""
    bbox = text_comp.bbox
    width = bbox["x2"] - bbox["x1"]
    height = bbox["y2"] - bbox["y1"]
    length= (width**2 + height**2)**0.5
    if length == 0:
        return None
    # 计算方向向量并归一化
    direction = [width / length, height / length]
    # print ("Direction vector:", direction, "Length:", length)
    # print("Normalized direction:", direction[0], direction[1])
    return direction

def is_line_parallel_to_text(line_comp: component, text_comp: component, angle_threshold=0.9):
    """
    判断引线与文本是否近似平行
    """
    line_direction = get_line_direction(line_comp)
    text_direction = get_text_direction(text_comp)
    
    if line_direction is None or text_direction is None:
        return False
    
    # 计算两个方向向量的余弦值
    dot_product = abs(line_direction[0] * text_direction[0] + line_direction[1] * text_direction[1])
    print("Dot product:", dot_product)
    # 如果余弦值大于阈值，则认为近似平行
    return dot_product > angle_threshold

def find_nearest_xuxiankuang_text_special_improved(comp: component, comp_list: all_components, threshold=500):
    """
    改进版本：找到引线附近所有符合条件的文本，如果有多个则选择最近的
    """
    parallel_texts = find_xuxiankuang_texts_with_parallel_check(comp, comp_list, threshold)
    
    if not parallel_texts:
        return None
    return parallel_texts



##########################################

#判断两点距离是否在阈值内
def judge_point(v1, v2, target, threshold=0.1):
    if abs(cal_eucilean_distance(v1, v2) - target) < threshold:
        return True
    else:
        return False

#识别剖面符号
def judge_sign(comp: component):
  
    if len(comp.data["verticesWidth"]) == 4 and \
       comp.data["verticesWidth"][0][0] == comp.data["verticesWidth"][0][1] and \
       comp.data["verticesWidth"][2][0] != 0 and comp.data["verticesWidth"][2][1] == 0:
        return True
    return False

#根据剖面符号的标题查找子图
def search_zitu_by_title_name(title_name: str, ori_components: list):
    for i in range(len(ori_components)):
        if ori_components[i].title == title_name:
            return ori_components[i]

#判断坐标是否在bbox内0616
def judge_point_in_bound(point, bbox, bbox_extend=0):

    new_bbox = { #进行一些扩大
        "x1": bbox["x1"] - bbox_extend,
        "x2": bbox["x2"] + bbox_extend,
        "y1": bbox["y1"] - bbox_extend,
        "y2": bbox["y2"] + bbox_extend
    }

    if new_bbox["x1"] <= point[0] <= new_bbox["x2"] and \
       new_bbox["y1"] <= point[1] <= new_bbox["y2"]:
        return True
    return False

#判断坐标在bbox附近
def judge_point_near_bound(point, bbox, threshold=60):
    x_min, y_min, x_max, y_max = float(bbox["x1"]), float(bbox["y1"]), float(bbox["x2"]), float(bbox["y2"])
    
    dx= min(abs(point[0] - x_max), abs(point[0] - x_min))
    dy= min(abs(point[1] - y_max), abs(point[1] - y_min))
    if dx <= threshold and point[1] >= y_min and point[1] <= y_max:
        return True
    else :
        if dy <= threshold and point[0] >= x_min and point[0] <= x_max:
            return True
    return False
def judge_point_near_bound_circle(point, xuxiankuang, threshold=60):
    #判断点是否在圆的边界附近
    center_x =xuxiankuang.data["center"][0]
    center_y = xuxiankuang.data["center"][1]
    radius = xuxiankuang.data["radius"]
    distance = ((point[0] - center_x) ** 2 + (point[1] - center_y) ** 2) ** 0.5
    if radius - threshold <= distance <= radius + threshold:
        return True
    else:
        return False
#判断点和bbox距离是否在阈值内
def judge_point_next_to_bound(point, bbox, threshold=100):
    x_min, y_min, x_max, y_max = float(bbox["x1"]), float(bbox["y1"]), float(bbox["x2"]), float(bbox["y2"])

    # 计算水平和垂直方向的最近距离
    dx = max(0, point[0] - x_max, x_min - point[0])
    dy = max(0, point[1] - y_max, y_min - point[1])

    # 计算两个 box 之间的最近距离
    distance = (dx ** 2 + dy ** 2) ** 0.5

    # 判断距离是否在阈值范围内
    if distance <= threshold:
        return True
    return False

#获取点到bbox的距离
def get_point_distance_to_bound(point, bbox):
    x_min, y_min, x_max, y_max = float(bbox["x1"]), float(bbox["y1"]), float(bbox["x2"]), float(bbox["y2"])

    # 计算水平和垂直方向的最近距离
    dx = max(0, point[0] - x_max, x_min - point[0])
    dy = max(0, point[1] - y_max, y_min - point[1])

    # 计算两个 box 之间的最近距离
    distance = (dx ** 2 + dy ** 2) ** 0.5

    return distance

#判断直线起点是否在bbox内
def start_in_bound(vertices, sign):
    start_1 = [vertices[0][0], vertices[0][1]]
    start_2 = [vertices[1][2], vertices[1][3]]

    if judge_point_in_bound(start_1, sign.bbox, 50) or judge_point_in_bound(start_2, sign.bbox, 50):
        return True
    return False

#判断两个剖面符号是否成对，并根据距离远近加入适当容差（距离远的容差大一点）
def add_rongcha(p1, p2, p3, p4, threshold=100, adaptive=False):
    dx1 = p2[0] - p1[0]
    dy1 = p2[1] - p1[1]

    dx2 = p4[0] - p3[0]
    dy2 = p4[1] - p3[1]

    vec_x = p3[0] - p1[0]
    vec_y = p3[1] - p1[1]

    cross_product = vec_x * dy1 - vec_y * dx1
    norm_v = (dx1 ** 2 + dy1 ** 2) ** 0.5

    distance = abs(cross_product) / norm_v
    if adaptive:
        distance_from_two_signs = cal_eucilean_distance(p1, p3)
        threshold *= min(3, max(1, distance_from_two_signs / 2000))
    return distance <= threshold

#判断是否四点共线（剖面符号平行）
def si_dian_gong_xian(p1, p2, p3, p4, threshold=0.99):
    dir1 = p2 - p1
    dir2 = p3 - p4

    cos_theta1 = (dir1[0] * dir2[0] + dir1[1] * dir2[1]) / (math.sqrt(dir1[0] ** 2 + dir1[1] ** 2) * math.sqrt(dir2[0] ** 2 + dir2[1] ** 2))

    dir3 = p1 - p3
    cos_theta2 = (dir1[0] * dir3[0] + dir1[1] * dir3[1]) / \
                (math.sqrt(dir1[0] ** 2 + dir1[1] ** 2) * math.sqrt(dir3[0] ** 2 + dir3[1] ** 2))

    if cos_theta1 >= threshold:  # and cos_theta2 >= 0.99:
        return add_rongcha(p1, p2, p3, p4, adaptive=True)
    return False

#剖面符号匹配成对判断
def pairing_signs(signs_list):
    result = []
    visited = [False] * len(signs_list)
    for i in range(len(signs_list)):
        if visited[i] == False:
            p1x = signs_list[i]["sign"].data["vertices"][0][0]
            p1y = signs_list[i]["sign"].data["vertices"][0][1]
            p2x = signs_list[i]["sign"].data["vertices"][0][2]
            p2y = signs_list[i]["sign"].data["vertices"][0][3]

            for j in range(i + 1, len(signs_list)):
                if visited[j] == False:
                    p3x = signs_list[j]["sign"].data["vertices"][0][0]
                    p3y = signs_list[j]["sign"].data["vertices"][0][1]
                    p4x = signs_list[j]["sign"].data["vertices"][0][2]
                    p4y = signs_list[j]["sign"].data["vertices"][0][3]
                    if si_dian_gong_xian(np.array([p1x, p1y]).astype(np.float32),
                                        np.array([p2x, p2y]).astype(np.float32),
                                        np.array([p3x, p3y]).astype(np.float32),
                                        np.array([p4x, p4y]).astype(np.float32)):
                        #找到平行的，且距离最近的配对
                        min_distance = float("inf")
                        index = j
                        for k in range(j, len(signs_list)):
                            if visited[k] == False:
                                compare_p3x = signs_list[k]["sign"].data["vertices"][0][0]
                                compare_p3y = signs_list[k]["sign"].data["vertices"][0][1]
                                compare_p4x = signs_list[k]["sign"].data["vertices"][0][2]
                                compare_p4y = signs_list[k]["sign"].data["vertices"][0][3]
                                if si_dian_gong_xian(np.array([p1x, p1y]).astype(np.float32),
                                                    np.array([p2x, p2y]).astype(np.float32),
                                                    np.array([compare_p3x, compare_p3y]).astype(np.float32),
                                                    np.array([compare_p4x, compare_p4y]).astype(np.float32)):
                                    distance = calc_distance_on_bbox(signs_list[k]["sign"].bbox,
                                                                    signs_list[i]["sign"].bbox)
                                    if distance < min_distance:
                                        min_distance = distance
                                        index = k
                        # 标记已访问
                        visited[i] = True
                        visited[index] = True
                        result.append((signs_list[i], signs_list[index]))
                        break
    return result

#找到与剖面相连直线上的文本（剖面标题）
def find_text_above_polyline(comp: component, comp_list: all_components, threshold=None):
    min_distance = float("inf")
    nearest_comp = None
    nearest_index = -1


    for i in range(len(comp_list)):
        if comp_list[i].data["type"] == "text" and comp_list[i].data["color"] == 3:
            #获得文本中心坐标
            center_text = [(comp_list[i].bbox["x1"] + comp_list[i].bbox["x2"]) / 2, (comp_list[i].bbox["y1"] + comp_list[i].bbox["y2"]) / 2]
            
            #如果是lwpolyline，用第二个点计算距离
            if comp.data["type"] == "lwpolyline":
                distance = cal_eucilean_distance(center_text, comp.data["vertices"][0][2:4]) 
                
            #如果是line，用start和min的较小值计算距离（不清楚哪个是start，哪个是end）
            elif comp.data["type"] == "line":
                distance = min(cal_eucilean_distance(center_text, comp.data["start"]), cal_eucilean_distance(center_text, comp.data["end"])) 
                
            if distance < min_distance:
                min_distance = distance
                nearest_comp = comp_list[i]
                nearest_index = i

    if threshold is not None:
        if min_distance > threshold:
            return None, None

    return nearest_comp, nearest_index

#找到与剖面符号相连的直线
def get_sign_title_line(sign, components: all_components):
    for i in sign:
        for comp in components:
            if (comp.data["type"] == "lwpolyline" and len(comp.data["vertices"]) == 2 and comp.data["color"] == 7 and start_in_bound(comp.data["vertices"], i["sign"])) or \
               (comp.data["type"] == "line" and comp.data["color"] == 7 and (judge_point_in_bound(comp.data["start"], i["sign"].bbox, 50) or judge_point_in_bound(comp.data["end"], i["sign"].bbox, 49))):
                nearest_green_text, _ = find_text_above_polyline(comp, components, threshold=1000)

                if nearest_green_text is not None:
                    title = nearest_green_text.data["content"]
                    if "SEE" in title: #特殊处理
                        if "\"" in title:
                            title = title.split("\"")[1]
                        else:
                            title = title.replace(" ", "").split("SEE")[1]

                    return title.strip()
            
#找到两个剖面符号附近的相同文字（几乎不用）
def get_sign_title_text(sign, ori_components: all_components):
    result = None
    for i in sign:
        
        for comp in ori_components:
            nearest_green_text, _ = find_nearest_green_text(i["sign"], ori_components, threshold=500)

            if nearest_green_text is not None and len(nearest_green_text.data["content"]) == 1:
                if result is None:
                    result = nearest_green_text.data["content"]
                else:
                    result = result + '-' + nearest_green_text.data["content"]

    return result
  
#判断剖面是否与剖面符号平行
def judge_poumian_parallel_sign(sign: component, poumian: component, sign_bbox=None, threshold=0.99):
    sign_x1 = sign["sign"].data["vertices"][0][0]
    sign_y1 = sign["sign"].data["vertices"][0][1]
    sign_x2 = sign["sign"].data["vertices"][0][2]
    sign_y2 = sign["sign"].data["vertices"][0][3]
    dir1 = np.array([sign_x1, sign_y1]) - np.array([sign_x2, sign_y2])

    poumian_x1 = poumian.data["start"][0]
    poumian_y1 = poumian.data["start"][1]
    poumian_x2 = poumian.data["end"][0]
    poumian_y2 = poumian.data["end"][1]

    dir2 = np.array([poumian_x1, poumian_y1]) - np.array([poumian_x2, poumian_y2])
    if (math.sqrt(dir1[0] ** 2 + dir1[1] ** 2) * math.sqrt(dir2[0] ** 2 + dir2[1] ** 2)) == 0:
        return False
    cos_theta = (dir1[0] * dir2[0] + dir1[1] * dir2[1]) / (math.sqrt(dir1[0] ** 2 + dir1[1] ** 2) * math.sqrt(dir2[0] ** 2 + dir2[1] ** 2))
    if sign_bbox is None:
        return abs(cos_theta) > threshold
    else:
        if abs(cos_theta) > threshold and \
        ((sign_bbox["x1"] <= poumian_x1 <= sign_bbox["x2"] and sign_bbox["y2"] >= poumian_y1 and sign_bbox["y1"] <= poumian_y2) or \
            (sign_bbox["y1"] <= poumian_y1 <= sign_bbox["y2"] and sign_bbox["x2"] >= poumian_x1 and sign_bbox["x1"] <= poumian_x2)):
            return True
        return False
    
#根据剖面符号寻找剖面
def search_poumian_by_signs(sign_pair, orl_components):
    signs_bbox = []
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = float('-inf'), float('-inf')
    for i in sign_pair: #获取两个剖面符号所在的最大bbox
        x_min = min(x_min, i["sign"].bbox["x1"])
        y_min = min(y_min, i["sign"].bbox["y1"])
        x_max = max(x_max, i["sign"].bbox["x2"])
        y_max = max(y_max, i["sign"].bbox["y2"])

    max_sign_bbox = { #进行一些扩大
        "x1": x_min - 80,
        "y1": y_min - 80,
        "x2": x_max + 80,
        "y2": y_max + 80
    }

    poumian_comps = []
    finished_poumian_handle = []
    #找到最近且平行于剖面符号的剖面
    for sign in sign_pair:
        tmp_comp = None
        min_distance = float("inf")
        for comp in orl_components:
            if comp.type == "line" and (comp.data["layerName"] in poumian_tuceng):
                start = comp.data["start"]
                end = comp.data["end"]
                distance = get_point_distance_to_bound(start, sign["sign"].bbox) + \
                          get_point_distance_to_bound(end, sign["sign"].bbox)
                if distance < min_distance and (judge_poumian_parallel_sign(sign, comp, max_sign_bbox)):
                    min_distance = distance
                    tmp_comp = comp

        if tmp_comp is not None and tmp_comp.data["handle"] not in finished_poumian_handle:
            poumian_comps.append(tmp_comp)
            finished_poumian_handle.append(tmp_comp.data["handle"])

    #获取其它平行剖面
    if poumian_comps is not None:
        all_poumian_comps = poumian_comps.copy()
        poumian_comps_handles = [i.data["handle"] for i in poumian_comps]
        for poumian_comp in poumian_comps:
            for comp in orl_components:
                if comp.type == "line" and (comp.data["layerName"] in poumian_tuceng) and \
                   comp.data["handle"] not in poumian_comps_handles:
                    if judge_coline_poumians(comp, poumian_comp, max_sign_bbox):
                        all_poumian_comps.append(comp)
        return all_poumian_comps
    return None


#判断其它剖面是否与该剖面平行
def judge_coline_poumians(comp, poumian_comp, sign_bbox):
    p1 = np.array(comp.data["start"])
    p2 = np.array(comp.data["end"])
    dir1 = p2 - p1

    p3 = np.array(poumian_comp.data["start"])
    p4 = np.array(poumian_comp.data["end"])
    dir2 = p4 - p3

    if (math.sqrt(dir1[0] ** 2 + dir1[1] ** 2) * math.sqrt(dir2[0] ** 2 + dir2[1] ** 2)) == 0:
        return False

    cos_theta = (dir1[0] * dir2[0] + dir1[1] * dir2[1]) / \
               (math.sqrt(dir1[0] ** 2 + dir1[1] ** 2) * math.sqrt(dir2[0] ** 2 + dir2[1] ** 2))
    if abs(cos_theta) > 0.98 and add_rongcha(p1, p2, p3, p4, 20) and \
       ((sign_bbox["x1"] <= comp.bbox["x1"] <= sign_bbox["x2"] and sign_bbox["y2"] >= comp.bbox["y1"] and sign_bbox["y1"] <= comp.bbox["y2"]) or \
        (sign_bbox["y1"] <= comp.bbox["y1"] <= sign_bbox["y2"] and sign_bbox["x2"] >= comp.bbox["x1"] and sign_bbox["x1"] <= comp.bbox["x2"])):
        return True
    return False

#获取从剖面符号到剖面的箭头
def get_arrow_from_sign_to_poumian(sign, poumian):
    arrow_start = [sign.bbox["x1"], sign.bbox["x2"]]
    poumian_start1 = [poumian.bbox["x1"], poumian.bbox["y1"]]
    poumian_start2 = [poumian.bbox["x2"], poumian.bbox["y2"]]
    distance1 = cal_eucilean_distance(arrow_start, poumian_start1)
    distance2 = cal_eucilean_distance(arrow_start, poumian_start2)
    arrow_end = poumian_start1 if distance1 < distance2 else poumian_start2
    arrow_bbox = {
        "x1": arrow_start[0],
        "y1": arrow_start[1],
        "x2": arrow_end[0],
        "y2": arrow_end[1]
    }
    return arrow_bbox

# def get_arrows_from_poumian_list(poumian_list):
#     if len(poumian_list) == 0 or len(poumian_list) == 1:
#         return None
#     else:
#         for i in range(len(poumian_list)):

#返回结果     
def return_result(detection_results_list: list):
    json_results_list = []
    bbox_list = []
    for result in detection_results_list:
        if result["success"] == 7:
            if(result["相似场景"] == "局部重绘"):
                result_dict = dict()
                result_dict["相似场景"] = "局部重绘"
                result_dict["主图标题"] = result["主图标题"]
                result_dict["主图点划线矩形框bbox"] = result["主图点划线矩形框bbox"]
                result_dict["主图点划线矩形框句柄"] = result["主图点划线矩形框句柄"]
                result_dict["子图bbox"] = result["子图bbox"]
                result_dict["引线文字内容"] = result["引线文字内容"]
                json_results_list.append(result_dict)
                #bbox_list.append((result["主图点划线矩形框bbox"], 3))
                if result["主图点划线矩形框bbox"] is not None:
                    bbox_list.append((result["主图点划线矩形框bbox"], 7))
                if result["子图bbox"] is not None:
                    bbox_list.append((result["子图bbox"], 8))
                continue
            if(result["相似场景"] == "文本指向相似"):
                result_dict = dict()
                result_dict["相似场景"] = "文本指向相似"
                result_dict["主图标题"] = result["主图标题"]
                result_dict["主图点划线框bbox"] = result["主图点划线框bbox"]
                result_dict["主图点划线框句柄"] = result["主图点划线框句柄"]
                result_dict["引线文字内容"] = result["引线文字内容"]
                json_results_list.append(result_dict)
                if result["主图点划线框bbox"] is not None:
                    bbox_list.append((result["主图点划线框bbox"], 9))
                continue
            if(result["相似场景"] == "局部放大"):
                result_dict = dict()
                result_dict["相似场景"] = "局部放大"
                result_dict["主图标题"] = result["主图标题"]
                result_dict["主图点划线圆形框或具体结构对象句柄"] = result["主图点划线圆形框或具体结构对象句柄"]
                result_dict["主图点划线圆形框或具体结构对象bbox"] = result["主图点划线圆形框或具体结构对象bbox"]
                result_dict["子图bbox"] = result["子图bbox"]
                result_dict["引线附近文本内容"] = result["引线附近文本内容"]
                json_results_list.append(result_dict)
                if result["主图点划线圆形框或具体结构对象bbox"] is not None:
                    bbox_list.append((result["主图点划线圆形框或具体结构对象bbox"], 10))
                if result["子图bbox"] is not None:
                    bbox_list.append((result["子图bbox"], 11))
                continue
        ##
            if result["相似场景"] == "FR格式子图":
                result_dict = dict()
                result_dict["相似场景"] = "FR格式子图"
                result_dict["主图标题"] = result["主图标题"]
                result_dict["副标题"] = result["副标题"]
                if result["剖面"] is not None:
                    poumian_list = []
                    result_dict["剖面"] = []
                    for poumian in result["剖面"]:
                        poumian_dict = {}
                        poumian_dict["剖面向量句柄"] = poumian.data["handle"]
                        poumian_dict["剖面向量起点"] = poumian.data["start"]
                        poumian_dict["剖面向量终点"] = poumian.data["end"]
                        poumian_bbox = {
                            "x1": poumian.data["start"][0],
                            "y1": poumian.data["start"][1],
                            "x2": poumian.data["end"][0],
                            "y2": poumian.data["end"][1]
                        }
                        poumian_list.append(poumian_bbox)
                        bbox_list.append((poumian_bbox, 12))
                        result_dict["剖面"].append(poumian_dict)
                        result_dict["子图标题"] = result["子图标题"]
                        result_dict["子图副标题"] = result["子图副标题"]
                        result_dict["剖面文本"] = result["剖面文本"]
                if result["zitu"] is not None:
                    zitu_bbox = result["zitu"].get_final_bbox()[0]
                    result_dict["子图bbox"] = zitu_bbox
                json_results_list.append(result_dict)
                continue
            if result["相似场景"] == "WL/DL格式子图":
                result_dict = dict()
                result_dict["相似场景"] = "WL/DL格式子图"
                result_dict["主图标题"] = result["主图标题"]
                result_dict["副标题"] = result["副标题"]
                if result["剖面"] is not None:
                    poumian_list = []
                    result_dict["剖面"] = []
                    for poumian in result["剖面"]:
                        poumian_dict = {}
                        poumian_dict["剖面向量句柄"] = poumian.data["handle"]
                        poumian_dict["剖面向量起点"] = poumian.data["start"]
                        poumian_dict["剖面向量终点"] = poumian.data["end"]
                        poumian_bbox = {
                            "x1": poumian.data["start"][0],
                            "y1": poumian.data["start"][1],
                            "x2": poumian.data["end"][0],
                            "y2": poumian.data["end"][1]
                        }
                        poumian_list.append(poumian_bbox)
                        bbox_list.append((poumian_bbox, 12))
                        result_dict["剖面"].append(poumian_dict)
                        result_dict["子图标题"] = result["子图标题"]
                        result_dict["子图副标题"] = result["子图副标题"]
                        result_dict["剖面文本"] = result["剖面文本"]
                if result["zitu"] is not None:
                    zitu_bbox = result["zitu"].get_final_bbox()[0]
                    result_dict["子图bbox"] = zitu_bbox
                json_results_list.append(result_dict)
                continue
            if result["相似场景"] == "FRDLWL格式子图-待处理":
                result_dict = dict()
                result_dict["相似场景"] = "FRDLWL格式子图-待处理"
                result_dict["wl_dl_title"] = result["wl_dl_title"]
                result_dict["wl_dl_info"] = result["wl_dl_info"]
                result_dict["主图标题"] = result["主图标题"]
                result_dict["副标题"] = result["副标题"]
                result_dict["子图标题"] = result["子图标题"]
                result_dict["子图副标题"] = result["子图副标题"]
                result_dict["signs_info"] = result["signs_info"]
                result_dict["剖面"]= result["剖面"]
                result_dict["剖面文本"]= result["剖面文本"]
                result_dict["all_insert_info"]= result["all_insert_info"]
                if result["zitu"] is not None:
                    zitu_bbox = result["zitu"].get_final_bbox()[0]
                    result_dict["子图bbox"] = zitu_bbox
                json_results_list.append(result_dict)
                continue
            ##

        result_dict = dict()
        result_dict["主图标题"] = result["主图标题"]
        result_dict["副标题"] = result["副标题"]
        
        result_dict["剖面标题"] = result["子图标题"]
        result_dict["剖面副标题"]= result["子图副标题"]
        result_dict["剖面文本"] = result["剖面文本"]
        result_dict["剖面符号bbox"] = [result["sign"][i]["sign"].bbox for i in range(len(result["sign"]))]
        if result["剖面"] is not None:
            poumian_list = []
            result_dict["剖面"] = []
            for poumian in result["剖面"]:
                poumian_dict = {}
                poumian_dict["剖面向量句柄"] = poumian.data["handle"]
                poumian_dict["剖面向量起点"] = poumian.data["start"]
                poumian_dict["剖面向量终点"] = poumian.data["end"]
                poumian_bbox = {
                    "x1": poumian.data["start"][0],
                    "y1": poumian.data["start"][1],
                    "x2": poumian.data["end"][0],
                    "y2": poumian.data["end"][1]
                }
                poumian_list.append(poumian_bbox)
                bbox_list.append((poumian_bbox, 5))
                result_dict["剖面"].append(poumian_dict)
        if result['剖面'] is None:
            result_dict["剖面"] = []
        if result['主图标题'] is None:
            result_dict["主图标题"] = ""
        if result['副标题'] is None:
            result_dict["副标题"] = []
        if result['子图标题'] is None:
            result_dict["剖面标题"] = ""
        if result['子图副标题'] is None:
            result_dict["剖面副标题"] = []
        if result['剖面文本'] is None:
            result_dict["剖面文本"] = ""
            # arrow_from_sign_to_poumian = get_arrow_from_sign_to_poumian(result["sign"[0]["sign"]],poumian_list[0])
            # arrows_from_poumian_to_poumian = get_arrows_from_poumian_list(poumian_list)
            # arrow_from_poumian_to_sign = get_arrow_from_poumian_to_sign(poumian_list[-1], result["sign"][1]["sign"])
        for sign in result["sign"]:
            sign_bbox = sign["sign"].bbox
            bbox_list.append((sign_bbox, result["success"]))
        
        if result["zitu"] is not None:
            zitu_bbox = result["zitu"].get_final_bbox()[0]
            bbox_list.append((zitu_bbox, 3))
            arrou_bbox = {
                "x1": result["sign"][0]["sign"].bbox["x1"],
                "y1": result["sign"][0]["sign"].bbox["y1"],
                "x2": zitu_bbox["x2"],
                "y2": zitu_bbox["y2"]
            }
            bbox_list.append((arrou_bbox, 4))
            result_dict["子图bbox"] = zitu_bbox
        
        json_results_list.append(result_dict)
    
    print(bbox_list)

    return json_results_list, bbox_list

#多级剖图入口函数
def detect(components_list: list, orl_components: all_components, insert_info: dict, human_components: dict):
    print("开始检测多级剖图：call detect_on_every_tukuang")
    result = detect_on_every_tukuang(orl_components, components_list, insert_info, human_components)
    print("结束检测多级剖图：call detect_on_every_tukuang")
    json_result, bbox_list = return_result(result)
    return json_result, bbox_list

#获取从剖面符号开始的箭头
def get_arrow_start_from_sign(sign_tuple, components: all_components):
    for comp in components:
        for sign in sign_tuple:
            if comp.type == "leader":
                start_point = [comp.data["vertices"][-1][0], comp.data["vertices"][-1][1]]
                if judge_point_in_bound(start_point, sign["sign"].bbox, 100):
                    return comp

#获取从虚线框开始的箭头
def get_arrow_start_from_xuxiankuang(xuxiankuang, components: all_components):
    for comp in components:
        if comp.type == "leader":
            start_point = [comp.data["vertices"][-1][0], comp.data["vertices"][-1][1]]

            if judge_point_in_bound(start_point, xuxiankuang.bbox, 100):
                return comp

#获取从虚线框开始的直线和多段线
def get_line_start_from_xuxiankuang(xuxiankuang, components: all_components):
    arrows = []
    for comp in components:
        if comp.type == "lwpolyline" or comp.type == "line":
            if comp.data["color"] == 7 or comp.data["color"] == 3:
                if comp.data["type"] == "lwpolyline":
                    start_point = [comp.data["vertices"][0][0], comp.data["vertices"][0][1]]
                else :
                    start_point = comp.data["start"]
                # 确保comp不是虚线框
                if comp.data["handle"] != xuxiankuang.data["handle"] and \
                judge_point_near_bound(start_point, xuxiankuang.bbox, 60):
                    arrows.append(comp)
    return arrows
                
def get_line_start_from_xuxiankuang_circle(xuxiankuang, components: all_components):
    for comp in components:
        if comp.type == "lwpolyline" or comp.type == "line":
            if comp.data["color"] == 7 or comp.data["color"] == 3:
                if comp.data["type"] == "lwpolyline":
                    start_point = [comp.data["vertices"][0][0], comp.data["vertices"][0][1]]
                else :
                    start_point = comp.data["start"]
                # 确保comp不是虚线框
                if comp.data["handle"] != xuxiankuang.data["handle"] and \
                judge_point_near_bound_circle(start_point, xuxiankuang, 40):
                    return comp
#找到剖面符号所在的主图
def find_sign_zhutu(sign: all_components, done_components_list: list):
    min_area = float("inf")
    zhutu_com_list = None
    for comp_list in done_components_list:
        comp_list_bbox = comp_list.get_final_bbox()[0]
        if is_fully_contained_strictly(comp_list_bbox, sign.bbox):
            area = calculate_area(comp_list_bbox)
            if area < min_area:
                min_area = area
                zhutu_com_list = comp_list

    if zhutu_com_list is not None:
        return zhutu_com_list

#获取未匹配的剖面符号
def get_remain_signs(ori_sign_list, new_signs_list):
    result = []
    for ori_sign in ori_sign_list:
        exist = False
        for new_sign_tuple in new_signs_list:
            for new_sign in new_sign_tuple:
                if ori_sign["sign"].data["handle"] == new_sign["sign"].data["handle"]:
                    exist = True
                    break
            if exist == True:
                break
        if exist == False:
            result.append(ori_sign)
    return result

#获取块中的剖面符号
def judge_sign_in_insert(insert: component, insert_info:dict):
    actual_insert = insert.data["blockName"]
    
    for insert_name in insert_info.keys():
        if actual_insert == insert_name:
            if len(insert_info[insert_name]) == 2:
                count = 0
                for e in insert_info[insert_name]:
                    if e["type"] == "lwpolyline":
                        vertices_width = e["verticesWidth"]
                        if len(vertices_width) == 4 and \
                            vertices_width[0][0] == vertices_width[0][1] and \
                            vertices_width[2][0] != 0 and vertices_width[2][1] == 0:
                                count += 1
            
                if count == 2:
                    return insert_info[insert_name]
                else:
                    return None
        
    return None

#实现从块中坐标转移到世界坐标
def coordinatesmap(p,insert,scales,rotation):
    rr=rotation/180*math.pi
    cosine=math.cos(rr)
    sine=math.sin(rr)

    # x,y=(p[0]*scales[0]+100)/200,(p[1]*scales[1]+100)/200
    x,y=((cosine*p[0]*scales[0]-sine*p[1]*scales[1]))+insert[0],((sine*p[0]*scales[0]+cosine*p[1]*scales[1]))+insert[1]
    return [x, y]

#获取世界坐标下的块里的剖面符号
def get_sign_in_insert(insert: component, insert_info: list):
    sign_list = []
    for e in insert_info:
        global_start_point = coordinatesmap((e["bound"]["x1"], e["bound"]["y1"]), insert.data["insert"], insert.data["scales"], insert.data["rotation"])
        global_end_point = coordinatesmap((e["bound"]["x2"], e["bound"]["y2"]), insert.data["insert"], insert.data["scales"], insert.data["rotation"])

        bound = {
            "x1": global_start_point[0],
            "y1": global_start_point[1],
            "x2": global_end_point[0],
            "y2": global_end_point[1]
        }
        e["bound"] = bound
        
        vertice_1 = coordinatesmap((e["vertices"][0][0], e["vertices"][0][1]), insert.data["insert"], insert.data["scales"], insert.data["rotation"])
        vertice_2 = coordinatesmap((e["vertices"][0][2], e["vertices"][0][3]), insert.data["insert"], insert.data["scales"], insert.data["rotation"])
        vertice_3 = coordinatesmap((e["vertices"][1][2], e["vertices"][1][3]), insert.data["insert"], insert.data["scales"], insert.data["rotation"])
        vertice_4 = coordinatesmap((e["vertices"][2][2], e["vertices"][2][3]), insert.data["insert"], insert.data["scales"], insert.data["rotation"])
        
        e["vertices"] = [
            [vertice_1[0],
             vertice_1[1],
             vertice_2[0],
             vertice_2[1]],
            
            [vertice_2[0],
             vertice_2[1],
             vertice_3[0],
             vertice_3[1]],
            
            [vertice_3[0],
             vertice_3[1],
             vertice_4[0],
             vertice_4[1]],
        ]
        
        sign = component(e["type"], e["bound"], e)
        sign_list.append(sign)
    
    return sign_list

#判断两条直线相交
def judge_two_line_connect(l1_p1, l1_p2, l2_p1, l2_p2):
    if cal_eucilean_distance(l1_p1, l2_p1) < 5 or \
       cal_eucilean_distance(l1_p1, l2_p2) < 5  or \
       cal_eucilean_distance(l1_p2, l2_p1) < 5  or \
       cal_eucilean_distance(l1_p2, l2_p2) < 5:
           return True
    return False

#从块中坐标转化到世界坐标
def convert_from_local_to_world(insert, e):
    new_e = copy.deepcopy(e)
    if "bound" in e.keys():
        global_start_point = coordinatesmap((e["bound"]["x1"], e["bound"]["y1"]), insert.data["insert"], insert.data["scales"], insert.data["rotation"])
        global_end_point = coordinatesmap((e["bound"]["x2"], e["bound"]["y2"]), insert.data["insert"], insert.data["scales"], insert.data["rotation"])

        bound = {
            "x1": global_start_point[0],
            "y1": global_start_point[1],
            "x2": global_end_point[0],
            "y2": global_end_point[1]
        }
        new_e["bound"] = bound
    if "start" in e.keys():
        new_e["start"] = coordinatesmap((e["start"][0], e["start"][1]), insert.data["insert"], insert.data["scales"], insert.data["rotation"])
    if "end" in e.keys():
        new_e["end"] = coordinatesmap((e["end"][0], e["end"][1]), insert.data["insert"], insert.data["scales"], insert.data["rotation"])

    return new_e

#获取块中的直线
def get_line_in_insert(insert, poumian, all_insert_info):
    poumian_point1 = poumian.data["start"]
    poumian_point2 = poumian.data["end"]
    
    actual_insert_name = insert.data["blockName"]
    actual_insert = []
    for insert_name in all_insert_info.keys():
        if actual_insert_name == insert_name:
            actual_insert = all_insert_info[insert_name]
            break
    
    connect_lines = []
    
    for e in actual_insert:
        if e["type"] == "line":
            global_start_point = coordinatesmap((e["bound"]["x1"], e["bound"]["y1"]), insert.data["insert"], insert.data["scales"], insert.data["rotation"])
            global_end_point = coordinatesmap((e["bound"]["x2"], e["bound"]["y2"]), insert.data["insert"], insert.data["scales"], insert.data["rotation"])

            if judge_two_line_connect(poumian_point1, poumian_point2, global_start_point, global_end_point):
                new_e = convert_from_local_to_world(insert, e)
                connect_lines.append(new_e)
    
    return connect_lines

#判断与剖面相连的直线
def judge_line_connect_with_poumian(line: component, poumian: component):
    line_start = line.data["start"]
    line_end = line.data["end"]
    poumian_start = poumian.data["start"]
    poumian_end = poumian.data["end"]
    
    if cal_eucilean_distance(line_start, poumian_start) < 1 or cal_eucilean_distance(line_start, poumian_end) < 1 or \
       cal_eucilean_distance(line_end, poumian_start) < 1 or cal_eucilean_distance(line_end, poumian_end) < 1:
           return True
    
    return False

#新需求：块中成对剖面、与剖面相连直线、与剖面相连的交点
def get_other_info(signs, poumian_list, ori_components, all_insert_info):
    angle_threshold = math.cos(40 * math.pi / 180)  # 30度的余弦值
    tmp_poumian_list = copy.deepcopy(poumian_list)
    
    result = []
    class line_and_count():
        def __init__(self):
            self.dict = {}
            self.count = defaultdict(int)
            
        
        def add(self, line, insert_handle=None):
            # self.dict[line["handle"]] = component(line["type"], line["bound"], line)
            # self.count[line["handle"]] += 1
            if insert_handle is not None:
                unique_key=f"{insert_handle}_{line['handle']}"
            else:
                unique_key=line["handle"]
            self.dict[unique_key] = component(line["type"], line["bound"], line)
            self.count[unique_key] += 1
    line_list = line_and_count()
    
    for poumian in poumian_list:
        point1 = poumian.data["start"]
        point2 = poumian.data["end"]
        for comp in ori_components:
            if comp.data["type"] == "insert":
                if judge_point_next_to_bound(point1, comp.data["bound"], 10) or \
                judge_point_next_to_bound(point2, comp.data["bound"], 10): #insert与剖面相连
                    lines = get_line_in_insert(comp, poumian, all_insert_info) #获得与剖面相连的直线
                    for line in lines:
                        # line_list.add(line)
                        line_list.add(line, insert_handle=comp.data["handle"])


    #统计每条直线出现的次数，1次需判断是否和剖面符号平行，2次则直接加入
    for line_handle, count in line_list.count.items():
        if count == 1:
            if judge_poumian_parallel_sign(signs[0], line_list.dict[line_handle], threshold=0.6) or judge_poumian_parallel_sign(signs[1], line_list.dict[line_handle], threshold=0.6):
                result.append(line_list.dict[line_handle])
                
        elif count == 2:
            result.append(line_list.dict[line_handle]) 
    
    
    #找到与剖面相连的直线
    finish_handle = [i.data["handle"] for i in poumian_list]
    for poumian in tmp_poumian_list:
        for comp in ori_components:
            if comp.data["type"] == "line":
                if judge_line_connect_with_poumian(comp, poumian) and comp.data["handle"] != poumian.data["handle"] and \
                    comp.data["handle"] not in finish_handle:
                        # result.append(comp)
                        # finish_handle.append(comp.data["handle"])
                        # tmp_poumian_list.append(comp)
                        #print("找到与剖面相连的直线:", comp.data["handle"])
                        # 添加角度检查，确保相连的直线角度合理
                        if judge_line_angle_reasonable(comp, poumian, threshold=angle_threshold):  # 使用参数化阈值
                            result.append(comp)
                            finish_handle.append(comp.data["handle"])
                            tmp_poumian_list.append(comp)
                            print("找到与剖面相连且角度合理的直线:", comp.data["handle"])
                        else:
                            print(f"过滤掉角度不合理的相连直线: {comp.data['handle']}")
                        
    #找与剖面相交的交点
    
    return result
########################0710
########################
import re
def compute_subtitle_call_count(sub_title_list):
    if not sub_title_list:
        return 0
    
    total_count = 0
    
    for subtitle in sub_title_list:
        # 只处理以SIM或SIM.结尾的字符串
        subtitle_stripped = subtitle.strip()
        if subtitle_stripped.endswith('SIM.'):
            # 去掉末尾的SIM.，获取主要部分
            main_part = subtitle_stripped[:-4].strip()
        elif subtitle_stripped.endswith('SIM'):
            # 去掉末尾的SIM，获取主要部分
            main_part = subtitle_stripped[:-3].strip()
        else:
            continue
        items = []
        for part in main_part.split(','):
            items.extend([item.strip() for item in part.split('/')])
        items = [item for item in items if item]  # 过滤空项
        
        for item in items:
            if not item:
                continue
            if '-' in item or '~' in item:
                # 处理范围格式
                count = parse_range_item(item)
                total_count += count
            else:
                # 单项计数
                total_count += 1
    
    return total_count


def parse_range_item(item):
    """
    解析范围项，计算范围内的项数
    
    支持的格式:
        - BL4-6 -> 3 (4,5,6)
        - BL4-BL6 -> 3 (4,5,6)
        - BL4~BL6 -> 3 (4,5,6)
        - BL4~6 -> 3 (4,5,6)
    """
    import re
    if '~' in item:
        # 格式: BL4~BL6
        parts = item.split('~')
        if len(parts) == 2:
            start_part = parts[0].strip()
            end_part = parts[1].split('~')[0].strip()
        else:
            return 1
    elif '-' in item:
        # 格式: BL4-6 或 BL4-BL6
        parts = item.split('-')
        if len(parts) == 2:
            start_part = parts[0].strip()
            end_part = parts[1].strip()
        else:
            return 1
    else:
        return 1
    
    # 提取数字
    start_num = extract_number(start_part)
    end_num = extract_number(end_part)
    
    if start_num is not None and end_num is not None:
        return max(1, end_num - start_num + 1)
    else:
        return 1


def extract_number(text):
    # 查找文本中的数字
    numbers = re.findall(r'\d+', text)
    if numbers:
        return int(numbers[-1])  # 取最后一个数字
    return None
#########################0710
#########################0711
#########################
def union_bbox_list(bbox1, jsonfile, tolerance=200):
    import json
    
    # 读取 JSON 文件
    try:
        with open(jsonfile, 'r', encoding='utf-8') as f:
            json_bboxes = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading JSON file: {e}")
        return bbox1
    
    # 获取有效的 JSON bbox 列表
    valid_json_bboxes = []
    for bbox in json_bboxes:
        if isinstance(bbox, dict) and all(key in bbox for key in ['x1', 'y1', 'x2', 'y2']):
            valid_json_bboxes.append(bbox)
    
    def is_bbox_similar(bbox1_dict, bbox2_dict, tolerance):
        """
        检查两个 bbox 是否在容差范围内相似
        """
        return (abs(bbox1_dict['x1'] - bbox2_dict['x1']) <= tolerance and
                abs(bbox1_dict['y1'] - bbox2_dict['y1']) <= tolerance and
                abs(bbox1_dict['x2'] - bbox2_dict['x2']) <= tolerance and
                abs(bbox1_dict['y2'] - bbox2_dict['y2']) <= tolerance)
    
    # 过滤 bbox1，删除与 JSON 中匹配的 bbox
    filtered_bbox1 = []
    for item in bbox1:
        if isinstance(item, tuple) and len(item) == 2:
            bbox_dict, num = item
            if isinstance(bbox_dict, dict) and all(key in bbox_dict for key in ['x1', 'y1', 'x2', 'y2']):
                # 检查是否与 JSON 中任何一个 bbox 相似
                is_similar = False
                for json_bbox in valid_json_bboxes:
                    if is_bbox_similar(bbox_dict, json_bbox, tolerance):
                        is_similar = True
                        break
                
                # 如果不相似，则保留
                if not is_similar:
                    filtered_bbox1.append(item)
            else:
                # 如果格式不正确，保留原项
                filtered_bbox1.append(item)
        else:
            # 如果格式不正确，保留原项
            filtered_bbox1.append(item)
    
    return filtered_bbox1
#########################0711
#在每一个图框中进行解析
def detect_on_every_tukuang(ori_components: all_components, done_components_list: list, all_insert_info: dict, human_components: dict):
    result = []
    signs_list = []
    new_signs_list = []
    #寻找剖面符号
    for comp in ori_components:
        if comp.data["type"] == "lwpolyline" and len(comp.data["vertices"]) == 3:
            if judge_sign(comp):
                print("找到剖面符号", comp.data["handle"])
                zhutu = find_sign_zhutu(comp, done_components_list)
                if zhutu is not None:
                    signs_list.append({"sign": comp, "zhutu_title": zhutu.title})
                else:
                    signs_list.append({"sign": comp, "zhutu_title": None})
        if comp.data["type"] == "insert":
            insert_info =  judge_sign_in_insert(comp, all_insert_info)
            if insert_info is not None:
                paired_signs_list_in_insert = get_sign_in_insert(comp, insert_info)
                print("找到块里的剖面符号：", [i.data["handle"] for i in paired_signs_list_in_insert])
                tmp_list = []
                for insert_sign in paired_signs_list_in_insert:
                    zhutu = find_sign_zhutu(insert_sign, done_components_list)
                    if zhutu is not None:
                        tmp_list.append({"sign": insert_sign, "zhutu_title": zhutu.title})
                    else:
                        tmp_list.append({"sign": insert_sign, "zhutu_title": None})
                tmp_list = tuple(tmp_list)
                new_signs_list.append(tmp_list)
    #对每个成对剖面符号进行处理

    if len(signs_list) != 0 or len(new_signs_list) != 0:
        if len(signs_list) != 0:
            new_signs_list.extend(pairing_signs(signs_list))
        print(len(new_signs_list))
        for signs in new_signs_list:
            
            zitu = None
            poumian_wenben = None
            title = get_sign_title_line(signs, ori_components) #先判断是否有直线指示的标题
            if title is not None:
                print("根据直线找到title: ", title)
                zitu = search_zitu_by_title_name(title, done_components_list)
                poumian_wenben = title
            else: #然后判断是否有箭头指示子图
                arrow = get_arrow_start_from_sign(signs, ori_components)
                if arrow is not None:
                    print("箭头存在", arrow.data["handle"])
                    zitu = search_zitu_by_arrow(arrow, ori_components, human_components)
                else: #最后判断剖面符号附近文本
                    title = get_sign_title_text(signs, ori_components)
                    if title is not None:
                        print("找到剖面符号附近文本: ", title)
                        zitu = search_zitu_by_title_name(title, done_components_list)

            poumian = search_poumian_by_signs(signs, ori_components)

            try:
                #新需求实现
                other_info = get_other_info(signs, poumian, ori_components, all_insert_info)
                if len(other_info) > 0:
                    poumian.extend(other_info)
            except:
                pass
            
            if poumian is not None:
                print("找到剖面", [i.data["handle"] for i in poumian])
            #根据comp找到主图comp
            zhutu = find_sign_zhutu(signs[0]["sign"], done_components_list)
                

            ##<
            if zitu is not None:
                print("DEBUGxxx")
                print("找到子图：", zitu.title)
                print("副标题：", zitu.sub_title)
                if zitu.sub_title is not None:
                    for index,every_sub_title in enumerate(zitu.sub_title):
                        print("子图副标题：", every_sub_title, "index:", index)
                        if is_fr_sim_format_with_former_DLWL(every_sub_title,index,zitu.sub_title):
                            print("检测到FRDLWL场景四，创建待处理项")
                            fr_info_list = extract_fr_numbers(every_sub_title)
                            print("提取的FR信息列表：", fr_info_list)
                            wl_dl_title = zitu.sub_title[index-1]
                            
                            # 创建待处理项，只存储必要的标题信息用于后续跨图框查找
                            for wl_dl_info in fr_info_list:
                                temp_wl_dl_info="FR"+str(wl_dl_info["fr_num"])
                                result.append({"相似场景": "FRDLWL格式子图-待处理",
                                            "wl_dl_title": wl_dl_title,
                                            "wl_dl_info": temp_wl_dl_info,
                                            "子图标题": zitu.title if zitu is not None else None,
                                            "子图副标题": zitu.sub_title if zitu is not None else None,
                                            "主图标题": signs[0]["zhutu_title"],
                                            "剖面": poumian,
                                            "副标题": zhutu.sub_title if zhutu is not None else None,
                                            "剖面文本": poumian_wenben if poumian_wenben is not None else None,
                                            "zitu": zitu,
                                            "all_insert_info": all_insert_info,
                                            "signs_info": {"sign_handles": [s["sign"].data["handle"] for s in signs]},
                                            "success": 7})
                        elif is_fr_sim_format(every_sub_title):
                            print("检测到FR格式副标题")
                            
                            fr_info_list = extract_fr_numbers(every_sub_title)
                            print("提取的FR信息列表：", fr_info_list)
                            
                            # 处理坐标轴
                            result_axis=None
                            for comp in zhutu.component_list:
                                if comp.data["type"]=="insert":
                                    result_axis=detect_coordinate_axis(comp,all_insert_info)
                                    if result_axis["is_axis"]:
                                        print("找到坐标轴：", comp.data["handle"])
                                        break
                            if result_axis is not None and result_axis["is_axis"]:
                                #针对每一个fr_info进行位置计算
                                x_positions =[]
                                x_positions= calculate_fr_x_positions(fr_info_list, result_axis)
                                if x_positions and len(x_positions) > 0 and poumian is not None and len(poumian) > 0:
                                    print("#$%^^",poumian)
                                    result_by_position = find_elements_at_fr_positions(zhutu, x_positions, poumian, threshold=50)
                                    for x_pos, elements in result_by_position.items():
                                        print(f"在位置 {x_pos} 处找到的元素: {elements}")
                                        result.append({"相似场景": "FR格式子图",
                                                    "主图标题": signs[0]["zhutu_title"],
                                                    "副标题": zhutu.sub_title if zhutu is not None else None,
                                                    "sign": signs,
                                                    "剖面": elements,
                                                    "zitu": zitu,
                                                    "子图标题": zitu.title if zitu is not None else None,
                                                    "子图副标题": zitu.sub_title if zitu is not None else None,
                                                    "剖面文本": poumian_wenben if poumian_wenben is not None else None,
                                                    "success": 7})
                            else:
                                print("未找到坐标轴但找到了符合条件的副标题!@#")
                        elif is_wl_dl_sim_format(every_sub_title):
                            print("检测到WL/DL格式副标题")
                            wl_dl_info_list = extract_wl_dl_numbers(every_sub_title)
                            print("提取的WL/DL信息列表：", wl_dl_info_list)
                            for wl_dl_info in wl_dl_info_list:
                                comp=find_text_in_main_figure(zhutu,wl_dl_info)
                                if comp is not None:
                                    print(f"找到文本组件: {wl_dl_info}")
                                    # 基于文本位置创建虚拟剖面符号对，复用现有剖面搜索逻辑
                                    virtual_signs = create_virtual_signs_from_text(comp, signs, poumian)
                                    if virtual_signs:
                                        print(f"成功创建虚拟剖面符号对")
                                    virtual_poumian = search_poumian_by_signs_wldl(virtual_signs, ori_components)

                                    try:
                                        #新需求实现
                                        other_info = get_other_info(virtual_signs, virtual_poumian, ori_components, all_insert_info)
                                        if len(other_info) > 0:
                                            virtual_poumian.extend(other_info)
                                    except:
                                        pass
                                    
                                    if virtual_poumian is not None:
                                        print("找到剖面", [i.data["handle"] for i in virtual_poumian])
                                        result.append({"相似场景": "WL/DL格式子图",
                                                    "主图标题": signs[0]["zhutu_title"],
                                                    "副标题": zhutu.sub_title if zhutu is not None else None,
                                                    "sign": signs,
                                                    "剖面": virtual_poumian,
                                                    "zitu": zitu,
                                                    "子图标题": zitu.title if zitu is not None else None,
                                                    "子图副标题": zitu.sub_title if zitu is not None else None,
                                                    "剖面文本": poumian_wenben if poumian_wenben is not None else None,
                                                    "success": 7})

                            
            ##<>
            if zitu is not None:
                if zhutu is not None:
                    result.append({"主图标题": signs[0]["zhutu_title"], "副标题": zhutu.sub_title, "sign": signs, "剖面": poumian, "zitu": zitu, 
                                "子图标题": zitu.title, "子图副标题":zitu.sub_title,"剖面文本": poumian_wenben, "success": 1})
                else:
                    result.append({"主图标题": signs[0]["zhutu_title"], "副标题": None, "sign": signs, "剖面": poumian, "zitu": zitu, 
                            "子图标题": zitu.title,"子图副标题":zitu.sub_title, "剖面文本": poumian_wenben, "success": 1})
            else:
                if zhutu is not None:
                    result.append({"主图标题": signs[0]["zhutu_title"], "副标题": zhutu.sub_title, "sign": signs, "剖面": poumian, "zitu": zitu, 
                                "子图标题": None, "子图副标题":None,"剖面文本": poumian_wenben, "success": 0})
                else:
                    result.append({"主图标题": signs[0]["zhutu_title"], "副标题": None, "sign": signs, "剖面": poumian, "zitu": zitu, 
                            "子图标题": None,"子图副标题":None, "剖面文本": poumian_wenben, "success": 0})

        #加入剩余剖面符号    
        remain_signs = get_remain_signs(signs_list, new_signs_list)
        for sign in remain_signs:
            result.append({"主图标题": None, "副标题": None, "sign": [sign], "剖面": None, "zitu": None, "子图标题": None, "子图副标题":None, "剖面文本": None, "success": 2})

    ################################
    #新需求—0615：场景一：相似剖面局部重绘
    xuxiankuang_list = []
    new_xuxiankuang_list = []

    for comp in ori_components:
        if comp.data["type"] == "lwpolyline" and len(comp.data["vertices"]) == 4 and \
           comp.data["color"] == 7 and comp.data["linetype"] in ["CENTER", "CENTER1", "CENTER2", "CENTER3", "CENTER4", "CENTER5", "JNRD_Center", "JNRD_PointLine","DDW035"]:
            #找到虚线框
            print("找到虚线框", comp.data["handle"])
            zhutu = find_sign_zhutu(comp, done_components_list)
            if zhutu is not None:
                xuxiankuang_list.append({"sign": comp, "zhutu_title": zhutu.title})
            else:
                xuxiankuang_list.append({"sign": comp, "zhutu_title": None})
            arrow = get_arrow_start_from_xuxiankuang(comp, ori_components)
            if arrow is not None:
                print("找到虚线框的箭头", arrow.data["handle"])
                #找到箭头指向的子图
                zitu = search_zitu_by_arrow(arrow, ori_components, human_components)
                if zitu is not None:
                    print("找到子图：", zitu.title)
                    leader_text = find_nearest_xuxiankuang_text_special_improved(arrow, ori_components)
                    zhutu_title= None
                    for i in xuxiankuang_list:
                        if i["sign"].data["handle"] == comp.data["handle"]:
                            zhutu_title = i["zhutu_title"]
                            break
                    zhutu_bbox = comp.bbox
                    result.append({"相似场景": "局部重绘",
                                "主图标题": zhutu_title,
                                "主图点划线矩形框bbox": zhutu_bbox,
                                "主图点划线矩形框句柄": comp.data["handle"],
                                "子图bbox": zitu.get_final_bbox()[0], 
                                "引线文字内容": leader_text if leader_text is not None else None,
                                "success": 7})
                
    

    #新需求：场景二：文本指向相似0616
    for comp in ori_components:
        if comp.data["type"] == "lwpolyline" and len(comp.data["vertices"]) == 4 and \
           comp.data["color"] == 7 and comp.data["linetype"] in ["CENTER", "CENTER1", "CENTER2", "CENTER3", "CENTER4", "CENTER5", "JNRD_Center", "JNRD_PointLine","DDW035"]:
            #找到虚线框
            print("找到虚线框", comp.data["handle"])
            zhutu = find_sign_zhutu(comp, done_components_list)
            if zhutu is not None:
                new_xuxiankuang_list.append({"sign": comp, "zhutu_title": zhutu.title})
            else:
                new_xuxiankuang_list.append({"sign": comp, "zhutu_title": None})
            arrow = get_line_start_from_xuxiankuang(comp, ori_components)
            arrows = get_line_start_from_xuxiankuang(comp, ori_components)
            if arrows is not None and len(arrows) > 0:
                # 遍历所有可能的箭头，直到找到一个有合适文本的
                found_valid_arrow = False
                for arrow in arrows:
                    print("找到虚线框的直线或多段线", arrow.data["handle"])
                    #找到直线或多段线附近的文本
                    # 修改0702
                    text_comp = find_nearest_xuxiankuang_text_special(arrow, ori_components,500)
                    if text_comp is not None and len(text_comp) > 0:
                        print("找到文本：", text_comp[0])
                        zhutu_title= None
                        for i in new_xuxiankuang_list:
                            if i["sign"].data["handle"] == comp.data["handle"]:
                                zhutu_title = i["zhutu_title"]
                                break
                        zhutu_bbox = comp.bbox
                        result.append({"相似场景": "文本指向相似",
                                    "主图标题": zhutu_title,
                                    "主图点划线框bbox": zhutu_bbox,
                                    "主图点划线框句柄": comp.data["handle"],
                                    "引线文字内容": text_comp,
                                    "success": 7})
                        found_valid_arrow = True
                        break  # 找到符合条件的箭头后跳出循环
                if not found_valid_arrow:
                    print("未找到符合条件的文本")
        if comp.data["type"] == "circle" and \
           comp.data["color"] == 7 and comp.data["linetype"] in ["CENTER", "CENTER1", "CENTER2", "CENTER3", "CENTER4", "CENTER5", "JNRD_Center", "JNRD_PointLine","DDW035"]:
            #找到虚线框
            print("找到虚线框", comp.data["handle"])
            zhutu = find_sign_zhutu(comp, done_components_list)
            if zhutu is not None:
                new_xuxiankuang_list.append({"sign": comp, "zhutu_title": zhutu.title})
            else:
                new_xuxiankuang_list.append({"sign": comp, "zhutu_title": None})
            arrow = get_line_start_from_xuxiankuang_circle(comp, ori_components)
            if arrow is not None:
                print("找到虚线框的直线或多段线", arrow.data["handle"])
                #找到直线或多段线附近的文本
                # 修改0702
                text_comp = find_nearest_xuxiankuang_text_special(arrow, ori_components,500)
                if text_comp is not None and len(text_comp) > 0:
                    print("找到文本：", text_comp[0])
                    zhutu_title= None
                    for i in new_xuxiankuang_list:
                        if i["sign"].data["handle"] == comp.data["handle"]:
                            zhutu_title = i["zhutu_title"]
                            break
                    zhutu_bbox = comp.bbox
                    result.append({"相似场景": "文本指向相似",
                                "主图标题": zhutu_title,
                                "主图点划线框bbox": zhutu_bbox,
                                "主图点划线框句柄": comp.data["handle"],
                                "引线文字内容": text_comp,
                                "success": 7})

    # 新需求：场景三0618局部放大
    used_leaders_in_multi_section = record_used_leaders_in_multi_section(new_signs_list, ori_components)
    results= []
    for comp in ori_components:
        if comp.data["type"]=="leader" and comp.data["color"]!=8:
            #先判定引线起点附近是否有虚线框，如果有，属于场景一，continue；否则属于场景三
            if comp.data["vertices"] is not None:
                start_point = [comp.data["vertices"][-1][0], comp.data["vertices"][-1][1]]
                end_point = [comp.data["vertices"][0][0], comp.data["vertices"][0][1]]
                flag= False
                # 检查是否属于场景一（虚线框相关）
                for xuxiankuang in xuxiankuang_list:
                    if judge_point_near_bound(start_point, xuxiankuang["sign"].bbox, 100) or \
                       judge_point_near_bound(end_point, xuxiankuang["sign"].bbox, 100):
                        print("引线起点或终点在虚线框附近，属于场景一，跳过")
                        print(comp.data["handle"])
                        flag = True
                        break

                # 检查是否已被多级剖图场景使用（精确过滤）
                if not flag and comp.data["handle"] in used_leaders_in_multi_section:
                    print("引线已被多级剖图场景使用，跳过")
                    print(comp.data["handle"])
                    flag = True
                
                if flag:
                    continue
                else:
                    #如果没有跳出循环，则说明引线不在虚线框附近
                    zhutu = find_sign_zhutu(comp, done_components_list)
                    print("找到引线场景三", comp.data["handle"])
                    #获取引线附近的文本
                    leader_text = find_nearest_xuxiankuang_text_special_improved(comp, ori_components)
                    print("引线附近文本：", leader_text if leader_text is not None else None)
                    # 获取主图标题
                    zhutu_title = None
                    zhutu = find_sign_zhutu(comp, done_components_list)
                    if zhutu is not None:
                        zhutu_title = zhutu.title
                    # 获取子图
                    zitu = search_zitu_by_arrow_special(comp, ori_components, human_components)
                    zhutustructure = find_nearest_none_text(comp, ori_components,threshold=100)
                    if zitu is not None:
                        print("找到子图：", zitu.title)
                        result.append({"相似场景": "局部放大",
                                    "主图标题": zhutu_title,
                                    "主图点划线圆形框或具体结构对象句柄": zhutustructure.data["handle"] if zhutustructure is not None else None,
                                    "主图点划线圆形框或具体结构对象bbox": zhutustructure.bbox if zhutustructure is not None else None,
                                    "子图bbox": zitu.get_final_bbox()[0], 
                                    "引线附近文本内容": leader_text if leader_text is not None else None,
                                    "success": 7})
                        
    ################################
    return result

##################################

#验证块内元素解析的函数
def verify_insert_elements(insert: component, all_insert_info: dict):
    actual_insert_name = insert.data["blockName"]
    print(f"\n=== 检测到Insert块: {actual_insert_name} ===")
    print(f"Insert句柄: {insert.data['handle']}")
    print(f"Insert插入点: {insert.data['insert']}")
    print(f"Insert缩放: {insert.data['scales']}")
    print(f"Insert旋转: {insert.data['rotation']}")
    
    # 查找对应的块定义
    actual_insert_elements = None
    for insert_name in all_insert_info.keys():
        if actual_insert_name == insert_name:
            actual_insert_elements = all_insert_info[insert_name]
            break
    
    if actual_insert_elements is None:
        print(f"未找到块定义: {actual_insert_name}")
        return
    
    print(f"块内元素总数: {len(actual_insert_elements)}")
    print("\n--- 块内元素详情 ---")
    
    # 输出前5个元素的信息（避免输出过多）
    max_elements = min(5, len(actual_insert_elements))
    for i, element in enumerate(actual_insert_elements[:max_elements]):
        print(f"\n元素 {i+1}:")
        print(f"  类型: {element.get('type', 'Unknown')}")
        print(f"  句柄: {element.get('handle', 'Unknown')}")
        
        # 输出局部坐标
        if 'bound' in element:
            local_bound = element['bound']
            print(f"  局部边界: ({local_bound.get('x1', 0):.2f}, {local_bound.get('y1', 0):.2f}) -> ({local_bound.get('x2', 0):.2f}, {local_bound.get('y2', 0):.2f})")
            
            # 计算全局坐标
            global_start = coordinatesmap(
                (local_bound.get('x1', 0), local_bound.get('y1', 0)), 
                insert.data["insert"], 
                insert.data["scales"], 
                insert.data["rotation"]
            )
            global_end = coordinatesmap(
                (local_bound.get('x2', 0), local_bound.get('y2', 0)), 
                insert.data["insert"], 
                insert.data["scales"], 
                insert.data["rotation"]
            )
            print(f"  全局边界: ({global_start[0]:.2f}, {global_start[1]:.2f}) -> ({global_end[0]:.2f}, {global_end[1]:.2f})")
        
        # 输出起点和终点（如果存在）
        if 'start' in element:
            local_start = element['start']
            global_start = coordinatesmap(local_start, insert.data["insert"], insert.data["scales"], insert.data["rotation"])
            print(f"  局部起点: ({local_start[0]:.2f}, {local_start[1]:.2f})")
            print(f"  全局起点: ({global_start[0]:.2f}, {global_start[1]:.2f})")
        
        if 'end' in element:
            local_end = element['end']
            global_end = coordinatesmap(local_end, insert.data["insert"], insert.data["scales"], insert.data["rotation"])
            print(f"  局部终点: ({local_end[0]:.2f}, {local_end[1]:.2f})")
            print(f"  全局终点: ({global_end[0]:.2f}, {global_end[1]:.2f})")
        
        # 输出顶点信息（如果存在）
        if 'vertices' in element and len(element['vertices']) > 0:
            print(f"  顶点数量: {len(element['vertices'])}")
            # 只输出前2个顶点
            max_vertices = min(2, len(element['vertices']))
            for j, vertex in enumerate(element['vertices'][:max_vertices]):
                if len(vertex) >= 2:
                    local_vertex = (vertex[0], vertex[1])
                    global_vertex = coordinatesmap(local_vertex, insert.data["insert"], insert.data["scales"], insert.data["rotation"])
                    print(f"    顶点{j+1}: 局部({local_vertex[0]:.2f}, {local_vertex[1]:.2f}) -> 全局({global_vertex[0]:.2f}, {global_vertex[1]:.2f})")
    
    if len(actual_insert_elements) > max_elements:
        print(f"\n... 还有 {len(actual_insert_elements) - max_elements} 个元素未显示")
    
    print("=" * 50)

# 检测坐标轴并计算比例尺
def detect_coordinate_axis(insert: component, all_insert_info: dict):
    """
    检测坐标轴并计算比例尺
    返回: {
        'is_axis': bool,
        'axis_info': {
            'main_line': 主坐标轴线,
            'scale_marks': 刻度线列表,
            'scale_texts': 刻度文本列表,
            'scale_ratio': 比例尺 (实际距离/图上距离),
            'scale_data': [{
                'text': 文本内容,
                'value': 刻度值

            }]
        }
    }
    """
    actual_insert_name = insert.data["blockName"]
    # 查找对应的块定义
    actual_insert_elements = None
    for insert_name in all_insert_info.keys():
        if actual_insert_name == insert_name:
            actual_insert_elements = all_insert_info[insert_name]
            break
    
    if actual_insert_elements is None:
        return {'is_axis': False, 'axis_info': None}
    
    # 分离线条和文本
    lines = []
    texts = []
    
    for element in actual_insert_elements:
        if element.get('type') == 'line':
            # 转换为全局坐标
            global_element = convert_from_local_to_world(insert, element)
            lines.append(global_element)
        elif element.get('type') == 'text':
            # 转换为全局坐标
            global_element = convert_from_local_to_world(insert, element)
            texts.append(global_element)
    
    # 检查是否符合坐标轴特征
    if len(lines) < 3 or len(texts) < 2:  # 至少需要主轴+2个刻度线，2个文本
        return {'is_axis': False, 'axis_info': None}
    
    # if actual_insert_name=="48.5KLEG":
    #     print("check1")
    # 查找主坐标轴线（最长的水平线）
    main_axis_line = find_main_axis_line(lines)
    if main_axis_line is None:
        return {'is_axis': False, 'axis_info': None}
    # if actual_insert_name=="48.5KLEG":
    #     print("check2")
    # 查找刻度线（垂直于主轴的短线）
    scale_marks = find_scale_marks(lines, main_axis_line)
    if len(scale_marks) < 2:
        return {'is_axis': False, 'axis_info': None}

    # 匹配刻度线和文本
    scale_data = match_scales_with_texts(scale_marks, texts)
    if len(scale_data) < 2:
        return {'is_axis': False, 'axis_info': None}
    # if actual_insert_name=="48.5KLEG":
    #     print("check3")
    # 计算比例尺
    scale_ratio = calculate_scale_ratio(scale_data)
    
    return {
        'is_axis': True,
        'axis_info': {
            'main_line': main_axis_line,
            'scale_marks': scale_marks,
            'scale_texts': texts,
            'scale_data': scale_data,
            'scale_ratio': scale_ratio
        }
    }

def find_main_axis_line(lines):
    """查找主坐标轴线（由多个水平线段组成）"""
    # 首先找到所有水平线段
    horizontal_lines = []
    
    for line in lines:
        start = line['start']
        end = line['end']
        
        # 计算线段长度
        length = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        
        # 检查是否为水平线（允许小的角度偏差）
        if length > 0 and abs(end[1] - start[1]) < length * 0.1:  # 垂直偏差小于10%
            horizontal_lines.append(line)
    
    if len(horizontal_lines) < 2:
        return None
    
    # 尝试将水平线段连接成主轴线
    # 按x坐标排序
    horizontal_lines.sort(key=lambda line: min(line['start'][0], line['end'][0]))
    
    # 找到在相同y坐标上的线段群组
    line_groups = []
    tolerance = 5  # y坐标容差
    
    for line in horizontal_lines:
        avg_y = (line['start'][1] + line['end'][1]) / 2
        
        # 查找是否有已存在的组可以加入
        added_to_group = False
        for group in line_groups:
            group_avg_y = sum((l['start'][1] + l['end'][1]) / 2 for l in group) / len(group)
            if abs(avg_y - group_avg_y) < tolerance:
                group.append(line)
                added_to_group = True
                break
        
        if not added_to_group:
            line_groups.append([line])
    
    # 找到最长的线段组（最可能是主轴）
    if not line_groups:
        return None
    
    main_group = max(line_groups, key=len)
    
    # 如果主轴组只有一条线段，直接返回
    if len(main_group) == 1:
        return main_group[0]
    
    # 构建主轴线的边界框
    all_x_coords = []
    all_y_coords = []
    
    for line in main_group:
        all_x_coords.extend([line['start'][0], line['end'][0]])
        all_y_coords.extend([line['start'][1], line['end'][1]])
    
    # 创建一个虚拟的主轴线，代表整个水平轴
    main_axis_line = {
        'start': [min(all_x_coords), sum(all_y_coords) / len(all_y_coords)],
        'end': [max(all_x_coords), sum(all_y_coords) / len(all_y_coords)],
        'segments': main_group  # 保存原始线段信息
    }
    
    return main_axis_line

def find_scale_marks(lines, main_axis_line):
    """查找刻度线（垂直于主轴的短线）"""
    if main_axis_line is None:
        return []
    
    main_start = main_axis_line['start']
    main_end = main_axis_line['end']
    main_length = math.sqrt((main_end[0] - main_start[0])**2 + (main_end[1] - main_start[1])**2)
    
    # 主轴方向向量
    main_direction = [(main_end[0] - main_start[0]) / main_length, (main_end[1] - main_start[1]) / main_length]
    
    # 获取主轴线段（如果是由多段组成）
    main_segments = main_axis_line.get('segments', [main_axis_line])
    
    scale_marks = []
    # print("主轴线段数量:", len(main_segments))
    # print("主轴长度:", main_length)
    # print("主轴方向:", main_direction)
    # print("刻度线数量:", len(lines))

    for line in lines:
        # 跳过主轴线段
        if line in main_segments:
            continue
            
        start = line['start']
        end = line['end']
        length = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        
        # 检查是否为短线（长度小于主轴的30%，因为主轴是由多段组成的）
        if length > main_length * 0.3:
            continue
        
        # 检查是否垂直于主轴
        if length > 0:
            line_direction = [(end[0] - start[0]) / length, (end[1] - start[1]) / length]
            # 计算点积，垂直线的点积应该接近0
            dot_product = abs(main_direction[0] * line_direction[0] + main_direction[1] * line_direction[1])
            if dot_product < 0.3:  # 允许一定的角度偏差
                # 检查刻度线是否在主轴附近
                if is_scale_mark_on_axis(line, main_axis_line):
                    scale_marks.append(line)
    
    return scale_marks

def is_scale_mark_on_axis(scale_line, main_axis_line):
    """检查刻度线是否在主轴上"""
    # 只需比对y坐标，因为主轴是水平的
    if main_axis_line is None or 'start' not in main_axis_line or 'end' not in main_axis_line:
        return False
    if 'start' not in scale_line or 'end' not in scale_line:
        return False
    # 主轴线的起点和终点
    main_start = main_axis_line['start']
    main_end = main_axis_line['end']
    scale_start = scale_line['start']
    scale_end = scale_line['end']
    return (abs(scale_start[1] - main_start[1]) < 10 or abs(scale_end[1] - main_end[1]) < 10)

def point_to_line_distance(point, line_start, line_end):
    """计算点到直线的距离"""
    # 使用点到直线距离公式
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # 避免除零错误
    if x1 == x2 and y1 == y2:
        return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
    
    # 点到直线距离公式
    distance = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    return distance

def match_scales_with_texts(scale_marks, texts):
    """匹配刻度线和文本 - 每个文本对应最近的刻度线"""
    scale_data = []
    
    for text in texts:
        # 获取文本位置
        bound = text.get('bound', {})
        text_x = (bound.get('x1', 0) + bound.get('x2', 0)) / 2
        
        # 查找最近的刻度线
        min_distance = float('inf')
        closest_scale_mark = None
        
        for scale_mark in scale_marks:
            # 获取刻度线的x坐标位置
            scale_x = (scale_mark['start'][0] + scale_mark['end'][0]) / 2
            
            # 计算x方向的距离
            distance = abs(text_x - scale_x)
            
            if distance < min_distance:
                min_distance = distance
                closest_scale_mark = scale_mark
        
        # # 如果找到了足够近的刻度线
        # print(f"文本: {text.get('content', '')}, 位置: {text_x}, 最近刻度线位置: {(closest_scale_mark['start'][0] + closest_scale_mark['end'][0]) / 2 if closest_scale_mark else '无'}, 距离: {min_distance}")
        
        if closest_scale_mark is not None and min_distance < 50:  # 可调整阈值
            try:
                # 尝试解析文本内容为数字
                text_content = text.get('content', '').strip()
                numeric_value = float(text_content)
                
                scale_data.append({
                    'position': (closest_scale_mark['start'][0] + closest_scale_mark['end'][0]) / 2,
                    'value': numeric_value,
                    'scale_mark': closest_scale_mark,
                    'text': text
                })
            except ValueError:
                # 如果文本不能转换为数字，跳过
                continue
    
    # 按位置排序
    scale_data.sort(key=lambda x: x['position'])
    
    return scale_data

def calculate_scale_ratio(scale_data):
    """计算比例尺"""
    if len(scale_data) < 2:
        return None
    
    # 使用第一个和最后一个刻度计算比例尺
    first_scale = scale_data[0]
    last_scale = scale_data[-1]
    
    # 图上距离（像素距离）
    drawing_distance = abs(last_scale['position'] - first_scale['position'])
    
    # 实际距离（根据刻度值）
    actual_distance = abs(last_scale['value'] - first_scale['value'])
    
    if drawing_distance == 0:
        return None
    
    # 比例尺 = 实际距离 / 图上距离
    scale_ratio = drawing_distance/actual_distance
    
    return {
        'ratio': scale_ratio,
        'description': f"图上1单位 = 实际{scale_ratio:.6f}单位",
        'first_scale': first_scale,
        'last_scale': last_scale,
        'drawing_distance': drawing_distance,
        'actual_distance': actual_distance
    }

def analyze_coordinate_axis(insert: component, all_insert_info: dict):
    #DEBUG for 坐标轴检测
    result = detect_coordinate_axis(insert, all_insert_info)
    
    if result['is_axis']:
        axis_info = result['axis_info']
        print(f"\n=== 检测到坐标轴: {insert.data['blockName']} ===")
        print(f"坐标轴句柄: {insert.data['handle']}")
        
        main_line = axis_info['main_line']
        print(f"主轴线: 起点({main_line['start'][0]:.2f}, {main_line['start'][1]:.2f}) -> 终点({main_line['end'][0]:.2f}, {main_line['end'][1]:.2f})")
        
        # 如果主轴线是由多段组成的，显示段数
        if 'segments' in main_line:
            print(f"主轴线由 {len(main_line['segments'])} 段组成")
            for i, segment in enumerate(main_line['segments']):
                print(f"  段{i+1}: ({segment['start'][0]:.2f}, {segment['start'][1]:.2f}) -> ({segment['end'][0]:.2f}, {segment['end'][1]:.2f})")
        
        print(f"刻度线数量: {len(axis_info['scale_marks'])}")
        print(f"文本数量: {len(axis_info['scale_texts'])}")
        
        if axis_info['scale_data']:
            print(f"有效刻度数据: {len(axis_info['scale_data'])}")
            for i, scale in enumerate(axis_info['scale_data']):
                print(f"  刻度{i+1}: 位置={scale['position']:.2f}, 值={scale['value']}")
        
        if axis_info['scale_ratio']:
            ratio_info = axis_info['scale_ratio']
            print(f"\n比例尺信息:")
            print(f"  {ratio_info['description']}")
            print(f"  图上距离: {ratio_info['drawing_distance']:.2f}")
            print(f"  实际距离: {ratio_info['actual_distance']:.2f}")
            print(f"  比例尺: {ratio_info['ratio']:.6f}")
        
        print("=" * 50)
        return result
    
#########################0717
######################### - FR SIM 格式检测和数字提取
def is_fr_sim_format(subtitle):
    """
    检测副标题是否符合FR开头SIM或SIM.结尾的格式
    """
    if not subtitle:
        return False
    
    subtitle_stripped = subtitle.strip()
    
    # 检查是否以FR开头
    if not subtitle_stripped.startswith('FR'):
        return False
    
    # 检查是否以SIM或SIM.结尾
    if subtitle_stripped.endswith('SIM.') or subtitle_stripped.endswith('SIM'):
        return True
    
    return False

def is_fr_sim_format_with_former_DLWL(every_subtitle,index,subtitle):
    if not subtitle:
        return False
    if index==0:
        return False
    if not is_fr_sim_format(every_subtitle):
        return False
    
    # 获取前一项
    previous_subtitle = subtitle[index - 1].strip()
    
    # 检查前一项是否为DL+数字或WL+数字格式，且不以SIM或SIM.结尾
    if previous_subtitle.endswith('SIM') or previous_subtitle.endswith('SIM.'):
        return False
    
    # 使用正则表达式检查是否匹配DL+数字或WL+数字格式
    dl_wl_pattern = r'^(DL|WL)\d+$'
    if re.match(dl_wl_pattern, previous_subtitle):
        return True
    
    return False

def extract_fr_numbers(subtitle):
    """
    subtitle: 副标题字符串，格式如"FR24-FR27 SIM"或"FR65,FR66/FR67~FR69 SIM."或"FR17+200 SIM"
    返回: FR信息列表，每个元素为字典 {'fr_num': int, 'offset': int}
    """
    if not is_fr_sim_format(subtitle):
        return []
    
    subtitle_stripped = subtitle.strip()
    
    # 去掉末尾的SIM或SIM.
    if subtitle_stripped.endswith('SIM.'):
        main_part = subtitle_stripped[:-4].strip()
    elif subtitle_stripped.endswith('SIM'):
        main_part = subtitle_stripped[:-3].strip()
    else:
        return []
    
    fr_info_list = []
    
    # 分割逗号和斜线
    items = []
    for part in main_part.split(','):
        items.extend([item.strip() for item in part.split('/')])
    items = [item for item in items if item]  # 过滤空项
    
    for item in items:
        if not item:
            continue
            
        # 检查是否包含加号（偏移量格式）
        if '+' in item:
            fr_info = parse_fr_offset_item(item)
            if fr_info:
                fr_info_list.append(fr_info)
        # 处理范围格式（短横线或波浪线）
        elif '-' in item or '~' in item:
            range_numbers = parse_fr_range_item(item)
            # 将范围数字转换为FR信息格式
            for num in range_numbers:
                fr_info_list.append({'fr_num': num, 'offset': 0})
        else:
            # 单项，直接提取数字
            number = extract_fr_number(item)
            if number is not None:
                fr_info_list.append({'fr_num': number, 'offset': 0})
    
    # 去重并按FR编号排序
    unique_fr_info = {}
    for info in fr_info_list:
        key = (info['fr_num'], info['offset'])
        unique_fr_info[key] = info
    
    return sorted(list(unique_fr_info.values()), key=lambda x: (x['fr_num'], x['offset']))


def parse_fr_offset_item(item):
    """
    解析带偏移量的FR项，如"FR17+200"
    
    Args:
        item: 包含偏移量的FR项字符串
        
    Returns:
        dict: {'fr_num': int, 'offset': int} 或 None
    """
    import re
    
    # 匹配FR数字+偏移量的格式
    match = re.match(r'FR(\d+)\+(\d+)', item.strip())
    if match:
        fr_num = int(match.group(1))
        offset = int(match.group(2))
        return {'fr_num': fr_num, 'offset': offset}
    
    return None


def parse_fr_range_item(item):
    """
        item: 范围项字符串，如"FR24-FR27"、"FR24-27"、"FR24~FR27"、"FR24~27"
    """
    import re
    
    if '~' in item:
        # 格式: FR24~FR27 或 FR24~27
        parts = item.split('~')
        if len(parts) == 2:
            start_part = parts[0].strip()
            end_part = parts[1].strip()
        else:
            return []
    elif '-' in item:
        # 格式: FR24-FR27 或 FR24-27
        parts = item.split('-')
        if len(parts) == 2:
            start_part = parts[0].strip()
            end_part = parts[1].strip()
        else:
            return []
    else:
        return []
    
    start_num = extract_number(start_part)
    end_num = extract_number(end_part)
    
    if start_num is not None and end_num is not None:
        return list(range(start_num, end_num + 1))
    else:
        return []


def extract_fr_number(text):
    import re
    
    # 查找FR后面的数字
    match = re.search(r'FR(\d+)', text)
    if match:
        return int(match.group(1))
    
    return None

def calculate_fr_x_positions(fr_info_list, result_axis):
    """
    计算FR开头的数字在坐标轴上的位置（支持偏移量）
    
    Args:
        fr_info_list: FR信息列表，每个元素为字典 {'fr_num': int, 'offset': int}
        result_axis: 坐标轴信息，包含比例尺和主轴线等
        
    Returns:
        list: 每个FR信息对应的x坐标位置列表
    """
    if not fr_info_list or not result_axis.get("is_axis"):
        return []
    
    axis_info = result_axis.get("axis_info", {})
    scale_data = axis_info.get("scale_data", [])
    
    if len(scale_data) < 2:
        print("警告：刻度数据不足，无法计算位置")
        return []
    
    # 按位置排序刻度数据
    scale_data_sorted = sorted(scale_data, key=lambda x: x['position'])
    
    # 计算比例尺
    scale_ratio = calculate_scale_ratio(scale_data_sorted)
    if scale_ratio is None:
        print("警告：无法计算比例尺")
        return []
    
    x_positions = []
    
    for fr_info in fr_info_list:
        fr_num = fr_info['fr_num']
        offset = fr_info['offset']
        
        # 首先获取FR编号对应的基础位置
        base_x_pos = interpolate_fr_position(fr_num, scale_data_sorted, scale_ratio)
        
        if base_x_pos is not None:
            # 如果有偏移量，需要将偏移量转换为像素距离并加到基础位置上
            if offset != 0:
                final_x_pos = base_x_pos + offset
            else:
                final_x_pos = base_x_pos
                
            x_positions.append(final_x_pos)
            if offset != 0:
                print(f"FR{fr_num}+{offset} 对应的x坐标: {final_x_pos} (基础位置: {base_x_pos}, 偏移: {offset})")
            else:
                print(f"FR{fr_num} 对应的x坐标: {final_x_pos}")
        else:
            print(f"警告：无法计算 FR{fr_num}+{offset} 的位置")
    
    return x_positions


def interpolate_fr_position(fr_number, scale_data, scale_ratio):
    if not scale_data or len(scale_data) < 2:
        return None
    
    # 查找最接近的两个刻度点
    left_scale = None
    right_scale = None
    
    for i in range(len(scale_data)):
        scale = scale_data[i]
        if scale['value'] <= fr_number:
            left_scale = scale
        if scale['value'] >= fr_number and right_scale is None:
            right_scale = scale
            break
    
    # 如果FR数字在刻度范围内，使用线性插值
    if left_scale and right_scale:
        if left_scale['value'] == right_scale['value']:
            return left_scale['position']
        
        value_diff = right_scale['value'] - left_scale['value']
        position_diff = right_scale['position'] - left_scale['position']
        
        ratio = (fr_number - left_scale['value']) / value_diff
        
        x_position = left_scale['position'] + ratio * position_diff
        
        return x_position
    
    elif left_scale and not right_scale:
        if len(scale_data) >= 2:
            last_scale = scale_data[-1]
            second_last_scale = scale_data[-2]
            
            value_diff = last_scale['value'] - second_last_scale['value']
            position_diff = last_scale['position'] - second_last_scale['position']
            
            if value_diff != 0:
                unit_position_change = position_diff / value_diff
                
                extrapolation_distance = (fr_number - last_scale['value']) * unit_position_change
                x_position = last_scale['position'] + extrapolation_distance
                
                return x_position
    
    elif not left_scale and right_scale:
        if len(scale_data) >= 2:
            first_scale = scale_data[0]
            second_scale = scale_data[1]
            
            value_diff = second_scale['value'] - first_scale['value']
            position_diff = second_scale['position'] - first_scale['position']
            
            if value_diff != 0:
                unit_position_change = position_diff / value_diff
                
                extrapolation_distance = (fr_number - first_scale['value']) * unit_position_change
                x_position = first_scale['position'] + extrapolation_distance
                
                return x_position
    
    return None


def is_component_vertical(comp, angle_threshold=15):
    """
    检查组件是否与x轴近似垂直
    
    Args:
        comp: 组件对象
        angle_threshold: 角度阈值（度），默认15度，即与垂直方向偏差不超过15度
        
    Returns:
        bool: 组件是否垂直
    """
    import math
    
    # 只对直线类型的组件进行垂直性检查
    if comp.data.get("type") != "line":
        return True  # 非直线组件不进行垂直性检查，默认通过
    
    # 获取直线的起点和终点
    if "start" not in comp.data or "end" not in comp.data:
        return True  # 缺少坐标信息，默认通过
    
    start = comp.data["start"]
    end = comp.data["end"]
    
    # 计算直线的方向向量
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    
    # 避免除零错误
    if abs(dx) < 1e-10:  # 完全垂直的情况
        return True
    
    # 计算直线与x轴的夹角
    angle_rad = math.atan(abs(dy) / abs(dx))
    angle_deg = math.degrees(angle_rad)
    
    # 检查是否接近垂直（即与x轴的夹角接近90度）
    vertical_angle_diff = abs(angle_deg - 90.0)
    
    return vertical_angle_diff <= angle_threshold

def find_elements_at_fr_positions(zhutu, x_positions, reference_poumian, threshold=50):
    """
    在主图中查找FR位置附近的元素
    
    Args:
        zhutu: 主图对象
        x_positions: FR位置列表，包含x坐标
        reference_poumian: 参考剖面列表
        threshold: 匹配阈值，默认为50
    """
    if not zhutu or not x_positions or not reference_poumian:
        return []
        
    result_elements = []
    result_by_position = {x_pos: [] for x_pos in x_positions}
    # 提取参考剖面的特征（图层、类型、颜色）
    reference_features = []
    for poumian_item in reference_poumian:
        if "layerName" in poumian_item.data and "type" in poumian_item.data and "color" in poumian_item.data:
            reference_features.append({
                "layer": poumian_item.data["layerName"],
                "type": poumian_item.data["type"],
                "color": poumian_item.data["color"]
            })
    print(f"参考剖面特征数量: {len(reference_features)}")
    
    for x_pos in x_positions:
        # 在主图的组件列表中查找符合条件的元素
        for comp in zhutu.component_list:
            if not comp.bbox:
                continue
                
            # 计算组件中心点的x坐标
            comp_center_x = (comp.bbox["x1"] + comp.bbox["x2"]) / 2
            
            # 检查组件是否在当前FR位置附近
            if abs(comp_center_x - x_pos) <= threshold:
                # 检查组件是否满足参考剖面的特征条件
                for feature in reference_features:
                    if ("layerName" in comp.data and comp.data["layerName"] == feature["layer"] and
                            "type" in comp.data and comp.data["type"] == feature["type"] and
                            "color" in comp.data and comp.data["color"] == feature["color"]):

                        if is_component_vertical(comp):
                            result_by_position[x_pos].append(comp)
                            print(f"找到匹配元素：{comp.data['handle']}, 位于FR位置: {x_pos:.2f}, x坐标: {comp_center_x:.2f}, 垂直性检查通过")
                        else:
                            print(f"元素不垂直，跳过：{comp.data['handle']}, 位于FR位置: {x_pos:.2f}, x坐标: {comp_center_x:.2f}")
                        break
    ##<!--NEWADD-->
    print("参考剖面元素数量:", len(reference_features))
    for x_pos in x_positions:
        if x_pos in result_by_position:
            elements = result_by_position[x_pos]
            if elements:
                print(f"FR位置 {x_pos:.2f} 找到 {len(elements)} 个元素")
            else:
                print(f"FR位置 {x_pos:.2f} 未找到匹配元素")
        else:
            print(f"FR位置 {x_pos:.2f} 未找到匹配元素")
    # 预先计算与参考剖面相交的所有直线，用于后续过滤
    reference_intersecting_lines = []
    if zhutu and zhutu.component_list:
        for ref_comp in reference_poumian:
            for comp in zhutu.component_list:
                if comp.data.get("type") == "line" and judge_intersect(ref_comp, comp):
                    reference_intersecting_lines.append(comp)
    
    print(f"找到与参考剖面相交的直线数量: {len(reference_intersecting_lines)}")
    
    # 对每个FR位置的结果进行过滤
    for x_pos, elements in result_by_position.items():
        if len(elements) > len(reference_poumian):
            print(f"FR位置 {x_pos:.2f} 找到 {len(elements)} 个元素，超过参考剖面元素数 {len(reference_poumian)}，开始过滤")
            
            # 应用过滤条件：保留与参考剖面有公共直线相交的元素
            filtered_elements = []
            for element in elements:
                # 检查当前元素是否与任何参考相交直线相交
                has_common_intersecting_line = False
                for intersecting_line in reference_intersecting_lines:
                    if judge_intersect(element, intersecting_line):
                        has_common_intersecting_line = True
                        print(f"元素 {element.data.get('handle', 'unknown')} 与参考相交直线有公共交点，保留")
                        break
                
                if has_common_intersecting_line:
                    filtered_elements.append(element)
                else:
                    print(f"元素 {element.data.get('handle', 'unknown')} 无公共相交直线，过滤掉")
            
            result_by_position[x_pos] = filtered_elements
            print(f"FR位置 {x_pos:.2f} 过滤后剩余 {len(filtered_elements)} 个元素")
    #</NEWADD>
    # # 打印每个FR位置找到的元素数量
    # for x_pos, elements in result_by_position.items():
    #     #print(f"FR位置 {x_pos:.2f} 找到 {len(elements)} 个元素")
    #     #将elements
    return result_by_position
#######################

#######################
##场景三场景四
def is_wl_dl_sim_format(subtitle):
    """
    检测副标题是否符合包含WL或DL并且以SIM或SIM.结尾的格式
    支持混合格式，如"WL12,DL13/WL14 SIM"
    """
    if not subtitle:
        return False
    
    subtitle_stripped = subtitle.strip()
    
    # 检查是否以SIM或SIM.结尾
    if not (subtitle_stripped.endswith('SIM.') or subtitle_stripped.endswith('SIM')):
        return False
    
    # 去掉末尾的SIM或SIM.
    if subtitle_stripped.endswith('SIM.'):
        main_part = subtitle_stripped[:-4].strip()
    elif subtitle_stripped.endswith('SIM'):
        main_part = subtitle_stripped[:-3].strip()
    else:
        return False
    
    # 检查主要部分是否包含WL或DL
    import re
    if re.search(r'\b(WL|DL)\d+', main_part):
        return True
    
    return False

def extract_wl_dl_numbers(subtitle):
    """
    从WL/DL格式的副标题中提取完整的标识符列表（包含前缀）
    subtitle: 副标题字符串，格式如"WL12,DL13/WL14 SIM"或"DL12,WL15 SIM."
    返回: 标识符列表，如["WL12", "DL13", "WL14"]
    """
    if not is_wl_dl_sim_format(subtitle):
        return []
    
    subtitle_stripped = subtitle.strip()
    
    # 去掉末尾的SIM或SIM.
    if subtitle_stripped.endswith('SIM.'):
        main_part = subtitle_stripped[:-4].strip()
    elif subtitle_stripped.endswith('SIM'):
        main_part = subtitle_stripped[:-3].strip()
    else:
        return []
    
    identifiers = []
    
    # 分割逗号和斜线
    items = []
    for part in main_part.split(','):
        items.extend([item.strip() for item in part.split('/')])
    items = [item for item in items if item]  # 过滤空项
    
    for item in items:
        if not item:
            continue
            
        # 处理范围格式（短横线）
        if '-' in item:
            range_identifiers = parse_wl_dl_range_item(item)
            identifiers.extend(range_identifiers)
        else:
            # 单项，检查是否包含WL或DL前缀
            if item.startswith('WL') or item.startswith('DL'):
                # 验证格式是否正确
                number = extract_wl_dl_number(item)
                if number is not None:
                    identifiers.append(item)
    
    # 去重并按前缀和数字排序
    unique_identifiers = list(set(identifiers))
    
    return unique_identifiers

def parse_wl_dl_range_item(item):
    """
    解析WL/DL范围项，如"WL12-WL15"、"WL12-15"、"DL12-DL15"、"DL12-15"
    返回包含前缀的标识符列表，如["WL12", "WL13", "WL14", "WL15"]
    """
    import re
    
    if '-' in item:
        parts = item.split('-')
        if len(parts) == 2:
            start_part = parts[0].strip()
            end_part = parts[1].strip()
        else:
            return []
    elif '~' in item:
        parts = item.split('~')
        if len(parts) == 2:
            start_part = parts[0].strip()
            end_part = parts[1].strip()
        else:
            return []
    else:
        return []
    
    # 提取起始部分的前缀和数字
    start_prefix = None
    if start_part.startswith('WL'):
        start_prefix = 'WL'
    elif start_part.startswith('DL'):
        start_prefix = 'DL'
    else:
        return []
    
    start_num = extract_wl_dl_number(start_part)
    
    # 处理结束部分
    if end_part.startswith('WL') or end_part.startswith('DL'):
        # 完整格式，如 "WL12-WL15"
        end_num = extract_wl_dl_number(end_part)
        end_prefix = 'WL' if end_part.startswith('WL') else 'DL'
        
        # 检查前缀是否一致
        if start_prefix != end_prefix:
            return []  # 前缀不一致，不支持跨前缀范围
    else:
        # 简化格式，如 "WL12-15"
        end_num = extract_number(end_part)
        end_prefix = start_prefix
    
    if start_num is not None and end_num is not None:
        # 生成范围内的所有标识符
        identifiers = []
        for num in range(start_num, end_num + 1):
            identifiers.append(f"{start_prefix}{num}")
        return identifiers
    else:
        return []

def extract_wl_dl_number(text):
    """
    从WL或DL文本中提取数字
    """
    import re
    
    # 查找WL或DL后面的数字
    match = re.search(r'(?:WL|DL)(\d+)', text)
    if match:
        return int(match.group(1))
    
    return None

def find_text_in_main_figure(zhutu, target_text):
    """
    在主图中查找指定的文本
    Args:
        zhutu: 主图对象
        target_text: 要查找的文本，如"DL12"
    Returns:
        找到的文本组件，如果没找到返回None
    """
    if not zhutu or not target_text:
        return None
    
    for comp in zhutu.component_list:
        if comp.data.get("type") == "text" and comp.data.get("content"):
            content = comp.data["content"].strip()
            if content == target_text:
                print(f"找到文本: {target_text} at handle: {comp.data.get('handle')}")
                return comp
    
    print(f"未找到文本: {target_text}")
    return None


def create_virtual_signs_from_text(text_comp, reference_signs, reference_profiles):
    """
    基于文本位置和参考剖面符号创建虚拟剖面符号对
    Args:
        text_comp: 文本组件（如DL12的文本）
        reference_signs: 参考剖面符号对
        reference_profiles: 参考剖面（用于验证方向）
    Returns:
        虚拟剖面符号对，格式与原始signs相同
    """
    if not text_comp or not reference_signs or len(reference_signs) < 2:
        return None
    
    # 获取文本中心点
    text_bbox = text_comp.bbox
    text_center = [(text_bbox["x1"] + text_bbox["x2"]) / 2, (text_bbox["y1"] + text_bbox["y2"]) / 2]
    
    # 找到离文本最近的剖面符号
    min_distance = float('inf')
    nearest_sign_idx = 0
    
    for i, sign_dict in enumerate(reference_signs):
        sign = sign_dict["sign"]
        sign_center = [(sign.bbox["x1"] + sign.bbox["x2"]) / 2, (sign.bbox["y1"] + sign.bbox["y2"]) / 2]
        distance = ((text_center[0] - sign_center[0])**2 + (text_center[1] - sign_center[1])**2)**0.5
        if distance < min_distance:
            min_distance = distance
            nearest_sign_idx = i
    
    # 获取最近的剖面符号和其配对的剖面符号
    nearest_sign = reference_signs[nearest_sign_idx]
    other_sign = reference_signs[1 - nearest_sign_idx]  # 配对的另一个剖面符号
    
    # 计算两个原始剖面符号之间的偏移向量
    nearest_center = [
        (nearest_sign["sign"].bbox["x1"] + nearest_sign["sign"].bbox["x2"]) / 2,
        (nearest_sign["sign"].bbox["y1"] + nearest_sign["sign"].bbox["y2"]) / 2
    ]
    other_center = [
        (other_sign["sign"].bbox["x1"] + other_sign["sign"].bbox["x2"]) / 2,
        (other_sign["sign"].bbox["y1"] + other_sign["sign"].bbox["y2"]) / 2
    ]
    
    offset_vector = [other_center[0] - nearest_center[0], other_center[1] - nearest_center[1]]
    
    # 在文本位置创建第一个虚拟剖面符号（对应最近的剖面符号）
    virtual_sign1 = create_virtual_sign_at_position(text_center, nearest_sign["sign"], nearest_sign["zhutu_title"])
    
    # 应用偏移创建第二个虚拟剖面符号
    virtual_sign2_center = [text_center[0] + offset_vector[0], text_center[1] + offset_vector[1]]
    virtual_sign2 = create_virtual_sign_at_position(virtual_sign2_center, other_sign["sign"], other_sign["zhutu_title"])
    
    print(f"创建虚拟剖面符号: 第一个在 {text_center}, 第二个在 {virtual_sign2_center}")
    print(f"偏移向量: {offset_vector}, 距离: {(offset_vector[0]**2 + offset_vector[1]**2)**0.5:.2f}")
    
    return [virtual_sign1, virtual_sign2]

def create_virtual_signs_from_text_reverse(text_comp, reference_signs, reference_profiles):
    """
    基于文本位置和参考剖面符号创建虚拟剖面符号对
    Args:
        text_comp: 文本组件（如DL12的文本）
        reference_signs: 参考剖面符号对
        reference_profiles: 参考剖面（用于验证方向）
    Returns:
        虚拟剖面符号对，格式与原始signs相同
    """
    if not text_comp or not reference_signs or len(reference_signs) < 2:
        return None
    
    # 获取文本中心点
    text_bbox = text_comp.bbox
    text_center = [(text_bbox["x1"] + text_bbox["x2"]) / 2, (text_bbox["y1"] + text_bbox["y2"]) / 2]
    
    # 找到离文本最近的剖面符号
    min_distance = float('inf')
    nearest_sign_idx = 0
    
    for i, sign_dict in enumerate(reference_signs):
        sign = sign_dict["sign"]
        sign_center = [(sign.bbox["x1"] + sign.bbox["x2"]) / 2, (sign.bbox["y1"] + sign.bbox["y2"]) / 2]
        distance = ((text_center[0] - sign_center[0])**2 + (text_center[1] - sign_center[1])**2)**0.5
        if distance < min_distance:
            min_distance = distance
            nearest_sign_idx = i
    
    # 获取最近的剖面符号和其配对的剖面符号
    nearest_sign = reference_signs[1-nearest_sign_idx]
    other_sign = reference_signs[nearest_sign_idx]  # 配对的另一个剖面符号
    
    # 计算两个原始剖面符号之间的偏移向量
    nearest_center = [
        (nearest_sign["sign"].bbox["x1"] + nearest_sign["sign"].bbox["x2"]) / 2,
        (nearest_sign["sign"].bbox["y1"] + nearest_sign["sign"].bbox["y2"]) / 2
    ]
    other_center = [
        (other_sign["sign"].bbox["x1"] + other_sign["sign"].bbox["x2"]) / 2,
        (other_sign["sign"].bbox["y1"] + other_sign["sign"].bbox["y2"]) / 2
    ]
    
    offset_vector = [other_center[0] - nearest_center[0], other_center[1] - nearest_center[1]]
    
    # 在文本位置创建第一个虚拟剖面符号（对应最近的剖面符号）
    virtual_sign1 = create_virtual_sign_at_position(text_center, nearest_sign["sign"], nearest_sign["zhutu_title"])
    
    # 应用偏移创建第二个虚拟剖面符号
    virtual_sign2_center = [text_center[0] + offset_vector[0], text_center[1] + offset_vector[1]]
    virtual_sign2 = create_virtual_sign_at_position(virtual_sign2_center, other_sign["sign"], other_sign["zhutu_title"])
    
    print(f"创建虚拟剖面符号: 第一个在 {text_center}, 第二个在 {virtual_sign2_center}")
    print(f"偏移向量: {offset_vector}, 距离: {(offset_vector[0]**2 + offset_vector[1]**2)**0.5:.2f}")
    
    return [virtual_sign1, virtual_sign2]
def create_virtual_sign_at_position(center, reference_sign, zhutu_title):
    """
    在指定位置创建虚拟剖面符号
    """
    # 获取参考剖面符号的尺寸和方向
    ref_vertices = reference_sign.data["vertices"][0]  # [x1, y1, x2, y2]
    width = (ref_vertices[2] - ref_vertices[0])
    height = (ref_vertices[3] - ref_vertices[1])
    
    # 在新位置创建相同尺寸的虚拟剖面符号
    virtual_vertices = [
        center[0] - width/2, center[1] - height/2,  # x1, y1
        center[0] + width/2, center[1] + height/2   # x2, y2
    ]
    
    # 创建虚拟组件，模拟剖面符号的数据结构
    class VirtualSign:
        def __init__(self, vertices, bbox):
            self.data = {
                "type": "lwpolyline",
                "vertices": [vertices],
                "handle": f"virtual_{id(self)}",
                "layerName": reference_sign.data.get("layerName", "0"),
                "color": reference_sign.data.get("color", 7)
            }
            self.bbox = bbox
            self.type = "lwpolyline"
    
    virtual_bbox = {
        "x1": min(virtual_vertices[0], virtual_vertices[2]),
        "y1": min(virtual_vertices[1], virtual_vertices[3]),
        "x2": max(virtual_vertices[0], virtual_vertices[2]),
        "y2": max(virtual_vertices[1], virtual_vertices[3])
    }
    
    virtual_sign = VirtualSign(virtual_vertices, virtual_bbox)
    
    return {"sign": virtual_sign, "zhutu_title": zhutu_title}


def search_poumian_by_signs_wldl(sign_pair, orl_components):
    signs_bbox = []
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = float('-inf'), float('-inf')
    for i in sign_pair: #获取两个剖面符号所在的最大bbox
        x_min = min(x_min, i["sign"].bbox["x1"])
        y_min = min(y_min, i["sign"].bbox["y1"])
        x_max = max(x_max, i["sign"].bbox["x2"])
        y_max = max(y_max, i["sign"].bbox["y2"])

    max_sign_bbox = { #进行一些扩大
        "x1": x_min - 80,
        "y1": y_min - 80,
        "x2": x_max + 80,
        "y2": y_max + 80
    }

    poumian_comps = []
    finished_poumian_handle = []
    #找到最近且平行于剖面符号的剖面
    for sign in sign_pair:
        tmp_comp = None
        min_distance = float("inf")
        for comp in orl_components:
            if comp.type == "line" and (comp.data["layerName"] in poumian_tuceng):
                start = comp.data["start"]
                end = comp.data["end"]
                distance = get_point_distance_to_bound(start, sign["sign"].bbox) + \
                          get_point_distance_to_bound(end, sign["sign"].bbox)
                if distance < min_distance and (judge_poumian_parallel_sign(sign, comp, max_sign_bbox,threshold=0.95)):
                    min_distance = distance
                    tmp_comp = comp

        if tmp_comp is not None and tmp_comp.data["handle"] not in finished_poumian_handle:
            poumian_comps.append(tmp_comp)
            finished_poumian_handle.append(tmp_comp.data["handle"])

    #获取其它平行剖面
    if poumian_comps is not None:
        all_poumian_comps = poumian_comps.copy()
        poumian_comps_handles = [i.data["handle"] for i in poumian_comps]
        for poumian_comp in poumian_comps:
            for comp in orl_components:
                if comp.type == "line" and (comp.data["layerName"] in poumian_tuceng) and \
                   comp.data["handle"] not in poumian_comps_handles:
                    if judge_coline_poumians(comp, poumian_comp, max_sign_bbox):
                        all_poumian_comps.append(comp)
        return all_poumian_comps
    return None

def find_text_in_FRDLWL_scene(wl_dl_title, wl_dl_info, ori_components, done_components_list):
    """
    在FRDLWL场景四中查找指定的文本
    Args:
        wl_dl_title: 要查找的文本内容（如"DL12"）
        wl_dl_info: FR信息，用于确定主图标题
        ori_components: 所有组件列表
        done_components_list: 已处理的组件列表，用于确定主图
    Returns:
        找到的文本组件，如果没找到返回None
    """
    if not wl_dl_title or not wl_dl_info or not ori_components or not done_components_list:
        return None
    
    # 遍历所有元素
    for comp in ori_components:
        # 验证当前的type为文本
        if comp.data.get("type") == "text" and comp.data.get("content"):
            content = comp.data["content"].strip()
            # 当前文本内容为wl_dl_title
            if content == wl_dl_title:
                # 获取当前文本所在的主图
                zhutu = find_sign_zhutu(comp, done_components_list)
                if zhutu is not None:
                    # 当前文本所在主图标题为wl_dl_info
                    if zhutu.title == str(wl_dl_info):
                        print(f"在FRDLWL场景四中找到文本: {wl_dl_title} 在主图 {wl_dl_info} 中, handle: {comp.data.get('handle')}")
                        return comp
    
    print(f"在FRDLWL场景四中未找到文本: {wl_dl_title} 在主图 {wl_dl_info} 中")
    return None



def find_text_in_FRDLWL_scene_cross_frame(wl_dl_title, wl_dl_info, result_components_list):
    """
    跨图框查找FRDLWL场景四中的文本
    Args:
        wl_dl_title: 要查找的文本内容（如"DL12"）
        wl_dl_info: FR信息，用于确定主图标题
        result_components_list: 所有图框的组件列表
    Returns:
        找到的文本组件，如果没找到返回None
    """
    if not wl_dl_title or not wl_dl_info or not result_components_list:
        return None,None
    
    # 遍历所有图框中的所有元素
    for comp_list in result_components_list:
        if hasattr(comp_list, 'component_list'):
            for comp in comp_list.component_list:
                # 验证当前的type为文本
                if comp.data.get("type") == "text" and comp.data.get("content"):
                    content = comp.data["content"].strip()
                    # 当前文本内容为wl_dl_title
                    if content == wl_dl_title:
                        # 检查当前文本所在的主图标题是否为wl_dl_info
                        if comp_list.title == str(wl_dl_info):
                            print(f"跨图框找到文本: {wl_dl_title} 在主图 {wl_dl_info} 中, handle: {comp.data.get('handle')}")
                            return comp,comp_list
    
    print(f"跨图框未找到文本: {wl_dl_title} 在主图 {wl_dl_info} 中")
    return None,None


def reconstruct_signs_from_handles(sign_handles, result_components_list):
    """
    根据handle列表重构signs对象
    Args:
        sign_handles: sign组件的handle列表
        result_components_list: 所有图框的组件列表
    Returns:
        重构的signs列表，格式与原始signs相同
    """
    if not sign_handles or not result_components_list:
        return []
    
    signs = []
    for comp_list in result_components_list:
        if hasattr(comp_list, 'component_list'):
            for comp in comp_list.component_list:
                if comp.data.get("handle") in sign_handles:
                    signs.append({"sign": comp, "zhutu_title": comp_list.title})
    
    return signs
def merge_duplicate_lists(dict_list, tuple_list):
    """
    对字典列表和元组列表进行归并，去除重复项
    
    Args:
        dict_list: 包含字典的列表
        tuple_list: 包含元组的列表，元组格式为(dict, number)
        
    Returns:
        tuple: (去重后的字典列表, 去重后的元组列表)
    """
    # 去重字典列表
    unique_dicts = []
    seen_dicts = set()
    
    for d in dict_list:
        # 将字典转换为可哈希的字符串形式用于比较
        dict_str = str(sorted(d.items()))
        if dict_str not in seen_dicts:
            seen_dicts.add(dict_str)
            unique_dicts.append(d)
    
    # 去重元组列表（元组格式为(dict, number)）
    unique_tuples = []
    seen_tuples = set()
    
    for item in tuple_list:
        if isinstance(item, tuple) and len(item) == 2:
            bbox_dict, num = item
            # 将字典转换为可哈希的字符串形式
            if isinstance(bbox_dict, dict):
                dict_str = str(sorted(bbox_dict.items()))
                tuple_identifier = (dict_str, num)
            else:
                tuple_identifier = (str(bbox_dict), num)
            
            if tuple_identifier not in seen_tuples:
                seen_tuples.add(tuple_identifier)
                unique_tuples.append(item)
        else:
            # 对于非标准格式的项，直接添加（如果不重复）
            item_str = str(item)
            if item_str not in seen_tuples:
                seen_tuples.add(item_str)
                unique_tuples.append(item)
    
    return unique_dicts, unique_tuples
#################################
##################################0703
def record_used_leaders_in_multi_section(new_signs_list: list, ori_components: all_components):
    """
    记录已经被多级剖图场景使用的引线句柄
    """
    used_leaders = set()
    
    for signs_tuple in new_signs_list:
        # 获取从剖面符号开始的箭头
        arrow = get_arrow_start_from_sign(signs_tuple, ori_components)
        if arrow is not None:
            used_leaders.add(arrow.data["handle"])
            print(f"记录多级剖图使用的引线: {arrow.data['handle']}")
    
    return used_leaders

#判断两条直线是否角度合理（平行或接近平行）
def judge_line_angle_reasonable(line1: component, line2: component, threshold=0.7):
    """
    判断两条直线的角度是否合理
    Args:
        line1, line2: 两个直线组件
        threshold: 角度阈值，cos值，默认0.8对应约37度以内
    Returns:
        bool: 角度是否合理
    """
    try:
        # 获取两条直线的方向向量
        p1 = np.array(line1.data["start"])
        p2 = np.array(line1.data["end"])
        dir1 = p2 - p1
        
        p3 = np.array(line2.data["start"])
        p4 = np.array(line2.data["end"])
        dir2 = p4 - p3
        
        # 计算向量长度
        len1 = math.sqrt(dir1[0] ** 2 + dir1[1] ** 2)
        len2 = math.sqrt(dir2[0] ** 2 + dir2[1] ** 2)
        
        if len1 == 0 or len2 == 0:
            return False
            
        # 计算夹角余弦值
        cos_theta = (dir1[0] * dir2[0] + dir1[1] * dir2[1]) / (len1 * len2)
        
        # 角度合理：平行（cos接近1或-1）或接近平行
        return abs(cos_theta) > threshold
        
    except Exception as e:
        print(f"角度检查异常: {e}")
        return False
    
##################################0703
#后处理
def after_deal_with_title(result_components_list: list, json_result: list, bbox_result: list):
    false_count = 0
    for single_result in json_result:
        if "相似场景" in single_result.keys() and single_result["相似场景"] == "局部重绘":
            continue
        if "相似场景" in single_result.keys() and single_result["相似场景"] == "文本指向相似":
            continue
        if "相似场景" in single_result.keys() and single_result["相似场景"] == "局部放大":
            continue
        if "相似场景" in single_result.keys() and single_result["相似场景"] == "FR格式子图":
            continue
        if "相似场景" in single_result.keys() and single_result["相似场景"] == "WL/DL格式子图":
            continue
        if "相似场景" in single_result.keys() and single_result["相似场景"] == "FRDLWL格式子图-待处理":
            continue
        if single_result["剖面文本"] is not None and (single_result["剖面标题"] is None or single_result["剖面标题"] == ""):
            for comps_list in result_components_list:
                if comps_list.title == single_result["剖面文本"]:
                    comp_bbox = comps_list.get_final_bbox()[0]
                    single_result["剖面标题"] = single_result["剖面文本"]

                    arrow_bbox = {
                        "x1": single_result["剖面符号bbox"][0]['x1'],
                        "y1": single_result["剖面符号bbox"][0]['y1'],
                        "x2": comp_bbox["x2"],
                        "y2": comp_bbox["y2"]
                    } 
                    bbox_result.append((arrow_bbox, 4))
                    bbox_result.append((comp_bbox, 3))
                    single_result["子图bbox"] = comp_bbox
                    for i in range(len(single_result["剖面符号bbox"])):
                        bbox_result.append((single_result["剖面符号bbox"][i], 1))
                        false_count += 1

###################
    # FRDLWL场景四跨图框处理
    frdlwl_results_to_add = []
    frdlwl_results_to_remove = []
    
    for single_result in json_result:
        if "相似场景" in single_result.keys() and single_result["相似场景"] == "FRDLWL格式子图-待处理":
            print(f"处理FRDLWL待处理项: wl_dl_title={single_result['wl_dl_title']}, wl_dl_info={single_result['wl_dl_info']}")
            
            # 跨图框查找文本
            comp,comp_list = find_text_in_FRDLWL_scene_cross_frame(
                single_result['wl_dl_title'], 
                single_result['wl_dl_info'], 
                result_components_list
            )
            
            if comp is not None:
                print(f"跨图框找到文本组件: {single_result['wl_dl_info']}")
                
                # 重构signs对象（需要通过handle查找原始sign对象）
                signs = reconstruct_signs_from_handles(single_result['signs_info']['sign_handles'], result_components_list)
                
                if signs:
                    # 创建完整的WL/DL格式子图结果
                    poumian =single_result["剖面"]
                    all_insert_info = single_result["all_insert_info"]
                    virtual_signs = create_virtual_signs_from_text(comp, signs, poumian)
                    if virtual_signs:
                        print(f"成功创建虚拟剖面符号对")
                    virtual_poumian = search_poumian_by_signs_wldl(virtual_signs, comp_list.component_list)

                    try:
                        #新需求实现
                        other_info = get_other_info(virtual_signs, virtual_poumian, comp_list.component_list, all_insert_info)
                        if len(other_info) > 0:
                            virtual_poumian.extend(other_info)
                    except:
                        pass
                    if virtual_poumian is None or len(virtual_poumian) == 0:
                        virtual_signs = create_virtual_signs_from_text_reverse(comp, signs, poumian)
                        if virtual_signs:
                            print(f"成功创建另一组虚拟剖面符号对")
                        virtual_poumian = search_poumian_by_signs_wldl(virtual_signs, comp_list.component_list)

                        try:
                            #新需求实现
                            other_info = get_other_info(virtual_signs, virtual_poumian, comp_list.component_list, all_insert_info)
                            if len(other_info) > 0:
                                virtual_poumian.extend(other_info)
                        except:
                            pass
                    if virtual_poumian is not None:
                        print("找到剖面", [i.data["handle"] for i in virtual_poumian])
                        new_result={"相似场景": "WL/DL格式子图-场景四",
                        "主图标题": single_result['wl_dl_info'],
                        "副标题": comp_list.sub_title,
                        "子图标题": single_result["子图标题"],
                        "子图副标题": single_result["子图副标题"],
                        "剖面文本": single_result['剖面文本'],
                        "子图bbox": single_result["子图bbox"],}
                        poumian_list = []
                        new_result["剖面"] = []
                        for poumian in virtual_poumian:
                            poumian_dict = {}
                            poumian_dict["剖面向量句柄"] = poumian.data["handle"]
                            poumian_dict["剖面向量起点"] = poumian.data["start"]
                            poumian_dict["剖面向量终点"] = poumian.data["end"]
                            poumian_bbox = {
                                "x1": poumian.data["start"][0],
                                "y1": poumian.data["start"][1],
                                "x2": poumian.data["end"][0],
                                "y2": poumian.data["end"][1]
                            }
                            poumian_list.append(poumian_bbox)
                            bbox_result.append((poumian_bbox, 12))
                            new_result["剖面"].append(poumian_dict)
                    
                        frdlwl_results_to_add.append(new_result)
            
            # 标记原待处理项为需要移除
            frdlwl_results_to_remove.append(single_result)
    
    # 移除待处理项并添加完成的结果
    for item in frdlwl_results_to_remove:
        json_result.remove(item)
    json_result.extend(frdlwl_results_to_add)
    json_result, bbox_result=merge_duplicate_lists(json_result, bbox_result)
    ###################
    # 新需求0615：统计每个子图剖面符号数量，即每个子图调用次数，汇总调用次数为一个字典，包含子图标题，主图标题，调用次数。并将字典添加到结果列表中
    # 0709 0719修订，添加新的计数
    subgraph_count = dict()
    subgraph_info = {}  # 用于存储每个子图的详细信息
    ##<NEWADD>
    for comp_list in result_components_list:
        if hasattr(comp_list, 'title'):
            try:
                comp_bbox = comp_list.get_final_bbox()[0]
                bbox_key = str(comp_bbox)
                
                # 初始化子图信息，调用次数为0
                if bbox_key not in subgraph_info:
                    main_title = comp_list.title if hasattr(comp_list, 'title') else "未知标题"
                    sub_title_list = comp_list.sub_title if hasattr(comp_list, 'sub_title') and comp_list.sub_title else []
                    sub_title_call_count = compute_subtitle_call_count(sub_title_list)
                    
                    subgraph_info[bbox_key] = {
                        "主图标题": main_title,
                        "剖面副标题": sub_title_list,
                        "剖切符号调用次数": 0,  # 初始化为0
                        "副标题调用次数": sub_title_call_count
                    }
                    subgraph_count[bbox_key] = 0  # 初始化调用次数为0
            except Exception as e:
                print(f"初始化子图信息时出错: {e}")
                continue
        ##<NEWADD/>
    for result in json_result:
        if "相似场景" in result.keys() and result["相似场景"] == "局部重绘":
            continue
        if "相似场景" in result.keys() and result["相似场景"] == "文本指向相似":
            continue
        if "相似场景" in result.keys() and result["相似场景"] == "局部放大":
            continue
        
        if "相似场景" in result.keys() and result["相似场景"] == "FR格式子图":
            continue
        if "相似场景" in result.keys() and result["相似场景"] == "WL/DL格式子图":
            continue
        #NEWADD
        if "相似场景" in result.keys() and result["相似场景"] == "WL/DL格式子图-场景四":
            continue
        #NEWADD/
        if "子图bbox" in result.keys():
            subgraph_bbox = result["子图bbox"]
            bbox_key = str(subgraph_bbox)
            
            # 统计调用次数
            if bbox_key not in subgraph_count:
                subgraph_count[bbox_key] = 1
            else:
                subgraph_count[bbox_key] += 1
            
            # 存储子图的详细信息（主图标题来自剖面标题）
            main_title = result.get("剖面标题", "未知标题")
            sub_title_list = result.get("剖面副标题", [])
            sub_title_call_count = compute_subtitle_call_count(sub_title_list)
            subgraph_info[bbox_key] = {
                "主图标题": main_title,
                "剖面副标题": sub_title_list,
                "剖切符号调用次数": subgraph_count[bbox_key],
                "副标题调用次数": sub_title_call_count
            }
    json_result.append({"子图调用次数": subgraph_info})
    json_result = [ json_result]
    ###################
    return json_result, bbox_result, false_count

if __name__ == "__main__":

    pass