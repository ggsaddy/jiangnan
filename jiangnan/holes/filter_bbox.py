import json

# 计算包围盒的面积
def calculate_area(bbox):
    width = bbox['x2'] - bbox['x1']
    height = bbox['y2'] - bbox['y1']
    return width * height

# 判断 box1 是否完全包含 box2
def is_fully_contained_with_outlier(box1, box2):
    # 放宽条件，在 box2 的上下左右四个边界分别加上或减去 1 的缓冲
    return (box1['x1'] <= box2['x1'] + 1 and box1['x2'] >= box2['x2'] - 1 and
            box1['y1'] <= box2['y1'] + 1 and box1['y2'] >= box2['y2'] - 1)
    
def judge_double_contain(box1, box2):
    return is_fully_contained_with_outlier(box1, box2) or is_fully_contained_with_outlier(box2, box1)

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

def load_data_and_get_main_bbox(filepath):
    main_bbox_dict = {}
    count = 0
    area_threshold = 1222342000
    
    with open(filepath, 'r', encoding='utf8') as fp:
        json_data = json.load(fp)
        if len(json_data) == 2:
            json_data = json_data[0]
        for data in json_data:
            
            area = calculate_area(data["bound"])
            if area >= area_threshold * 0.99 and area < area_threshold * 20:
                main_bbox_dict[count] = data["bound"]
                count += 1
                   
        if len(main_bbox_dict) == 0:
            raise ValueError("未检测到图框, 请检查图框类型和尺寸.")
        
        main_bbox_dict, _ = remove_duplicate_tukuang(main_bbox_dict)
        
        
    result = list(main_bbox_dict.values())
    # print(result)
    return result


if __name__ == "__main__":
    # segment_v1("/disk1/user4/work/造船厂/结构AI/test1018/test1018_0.json","result1024.png")
    # segment_v2("/disk1/user4/work/造船厂/结构AI/example/test1114模拟.json","result/result1125.png")
    main_bbox = load_data_and_get_main_bbox("data1114_v2.json")
    print(main_bbox)
    # visualize_filtered_bbox("/disk1/user4/work/造船厂/结构AI/example/test1202模拟.json", "result/bbox.png")