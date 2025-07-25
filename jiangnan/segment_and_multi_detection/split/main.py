import sys
sys.path.append('..')
from split.utils import *
# from split.contain import *
from split.multi import detect, after_deal_with_title
import time
from multiprocessing import Pool
from tqdm import tqdm

#画图函数
def visualize_filtered_bbox(filepath, output_path):
    # 加载数据
    components = load_data(filepath)
    # 过滤的混合
    filtered_camps, x_min, y_min, x_max, y_max = filter_bounding_boxes(components, threshold=10)
    # 可视化结果
    visualize_bounding_boxes(filtered_camps, x_min, y_min, x_max, y_max, output_path)

def process_components(components):
    """静态处理一个 components 的函数，用于多进程处理。"""
    
    #分main和human，human用于人工画框纠正
    main_components = components["main"]
    human_components = components["human"]
    #
    filtered_all_comps, x_min_tmp, y_min_tmp, x_max_tmp, y_max_tmp = filter_bounding_boxes(main_components, threshold=0.1)
    
    # 查找标题并根据标题查找子图
    print("查找标题")
    sub_camps_list, re_search_index = search_titles(filtered_all_comps)

    print("去除数据元素")
    unnerged_components_list = remove_duplicate_comps(sub_camps_list)
    #归并剩余子图
    merged_components_list = merge_nearest_comp(unnerged_components_list, filtered_all_comps, re_search_index)
    # 解决大图包含小图问题
    final_components_list = solve_big_contain_small(merged_components_list)
    # 返回结果和当前的范围（用于画图）
    return final_components_list, (x_min_tmp, y_min_tmp, x_max_tmp, y_max_tmp)

def segment_separately(components_list):
    """使用多进程处理 components_list 的函数。"""
    # 初始化结果和边界值
    result = []
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = float('-inf'), float('-inf')

    print("分割中...")
    # 使用多进程处理
    with Pool() as pool:
        # 使用过程的处理每个组件
        results = list(tqdm(pool.imap(process_components, components_list.values()), total=len(components_list)))
    # 汇总结果
    for final_components_list, (x_min_tmp, y_min_tmp, x_max_tmp, y_max_tmp) in results:
        result.append(final_components_list)
        x_min = min(x_min, x_min_tmp)
        x_max = max(x_max, x_max_tmp)
        y_min = min(y_min, y_min_tmp)
        y_max = max(y_max, y_max_tmp)
    visual_axis = (x_min, y_min, x_max, y_max)
    return result, visual_axis

#过滤list
def filter_dict(lst):
    result = []
    for i in range(len(lst)):
        if lst[i] not in result:
            result.append(lst[i])
    return result

def process_multi(components, insert_info):
    # 过滤创建点
    main_components = components["main"]
    human_components = components["human"]
    
    filtered_all_comps, x_min_tmp, y_min_tmp, x_max_tmp, y_max_tmp = filter_bounding_boxes(main_components, threshold=0.9)

    # 查找数据的主要
    print("查找数据中")
    sub_comps_list, re_search_index = search_titles(filtered_all_comps)
    print("去除重复元素")
    unnerged_components_list = remove_duplicate_comps(sub_comps_list)
    merged_components_list = merge_nearest_comp(unnerged_components_list, filtered_all_comps, re_search_index)


    #加入人工画框结果
    merged_combine_human_list, ori_components = merge_human_result(merged_components_list, human_components, filtered_all_comps)
    
    # 解决大图包含小图的问题
    final_components_list = solve_big_contain_small(merged_combine_human_list)

    #多级剖图主函数
    result_json, result_bbox = detect(final_components_list, ori_components, insert_info, human_components)

    # 返回处理结果和当前的包覆盖范围
    return result_json, result_bbox, final_components_list, (x_min_tmp, y_min_tmp, x_max_tmp, y_max_tmp)



def segment_of_multi(components_list, insert_info):
    # 初始化结果和分析值
    json_result = []
    bbox_result = []
    result_components_list = []
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = float('-inf'), float('-inf')

    print('分割中...')

    # 使用控制模型
    with Pool() as pool:
       # 构造参数列表：每个元素是 (单个组件, 全部插入信息)
        args_list = [(component, insert_info) for component in components_list.values()]

        # 使用 starmap 并加上 tqdm 进度条
        results = list(tqdm(pool.starmap(process_multi, args_list), total=len(args_list)))


        for json_res, bbox_res, components_list, (x_min_tmp, y_min_tmp, x_max_tmp, y_max_tmp) in results:
            json_result.append(json_res)
            bbox_result.append(bbox_res)
            result_components_list.append(components_list)
            x_min = min(x_min, x_min_tmp)
            x_max = max(x_max, x_max_tmp)
            y_min = min(y_min, y_min_tmp)
            y_max = max(y_max, y_max_tmp)

    result_components_list = [item for sublist in result_components_list for item in (sublist if isinstance(sublist, list) else [sublist])]
    json_result = [item for sublist in json_result for item in (sublist if isinstance(sublist, list) else [sublist])]
    bbox_result = [item for sublist in bbox_result for item in (sublist if isinstance(sublist, list) else [sublist])]

    #后处理，用于跨图框匹配
    json_result, bbox_result, false_count = after_deal_with_title(result_components_list, json_result, bbox_result)
    
    visual_axis = (x_min, y_min, x_max, y_max)
    return json_result, bbox_result, visual_axis, false_count


def segment_v4(filepath, output_path=None):
    """图面分割函数入口
    filepath: json文件路径"""
    start = time.perf_counter()
    print("正在加载数据，检测阈值...")
    components, _ = load_data_and_get_main_bbox(filepath)
    # components, _, = load_data_and_get_main_bbox_local(filepath)
    
    final_components_list, v1s_axis = segment_separately(components)

    print("可视化中...")
    final_components_list = [item for sublist in final_components_list for item in (sublist if isinstance(sublist, list) else [sublist])]
    visualize_many_bbox(final_components_list, *v1s_axis, output_path)
    print(final_components_list)
    print("正在保存结果至文件夹...")
    # save_to_json(final_components_list, filepath)

    # final_bboxs = get_bbox(final_components_list)
    # final_bboxs = filter_dict(final_bboxs)

    #获取多段线包围盒
    final_bboxs = get_polygon_bbox(final_components_list)
    
    
    print("分割完毕，请查看分割结果")
    end = time.perf_counter()
    print("时间花费: {}s".format(round(end - start, 2)))
    return final_bboxs

def segment_v4_local(filepath, output_path=None):
    """图面分割本地测试函数
        filepath: json文件路径
        output_path: 输出路径
    """
    start = time.perf_counter()
    print("正在加载数据，检测器值...")
    # components, _ = load_data_and_get_main_bbox(filepath)
    components, _, = load_data_and_get_main_bbox_local(filepath)
    
    final_components_list, v1s_axis = segment_separately(components)
    print("可视化中...")
    visualize_many_bbox(final_components_list, *v1s_axis, output_path)
    final_components_list = [item for sublist in final_components_list for item in (sublist if isinstance(sublist, list) else [sublist])]
    print(final_components_list)
    print("正在保存结果至文件夹...")
    # save_to_json(final_components_list, filepath)

    # final_bboxs = get_bbox(final_components_list)
    # final_bboxs = filter_dict(final_bboxs)
    final_bboxs = get_polygon_bbox(final_components_list)
    
    print("分割完毕，请查看分割结果")
    end = time.perf_counter()
    print("时间花费: {}s".format(round(end - start, 2)))
    return final_bboxs
def segment_v0725(filepath, output_folder=None):
    start = time.perf_counter()
    print("正在加载数据，检测器值...")
    output_path =output_folder+"/segment.png"
    # components, _ = load_data_and_get_main_bbox(filepath)
    components, _, = load_data_and_get_main_bbox_local(filepath)
    
    final_components_list, v1s_axis = segment_separately(components)
    print("可视化中...")
    visualize_many_bbox(final_components_list, *v1s_axis, output_path)
    final_components_list = [item for sublist in final_components_list for item in (sublist if isinstance(sublist, list) else [sublist])]
    print(final_components_list)
    print("正在保存结果至文件夹...")
    # save_to_json(final_components_list, filepath)

    # final_bboxs = get_bbox(final_components_list)
    # final_bboxs = filter_dict(final_bboxs)
    final_bboxs = get_polygon_bbox(final_components_list)
    
    print("分割完毕，请查看分割结果")
    end = time.perf_counter()
    print("时间花费: {}s".format(round(end - start, 2)))
    return final_bboxs
def multi_detect(filepath, output_path=None):
    """多级剖面图函数入口
    filepath: json文件路径"""
    start = time.perf_counter()
    print("正在加载数据，检测器值...")
    components, insert_info = load_data_and_get_main_bbox(filepath)
    
    json_result, bbox_result, vis_axis, false_count = segment_of_multi(components, insert_info)
    print("可视化中...")
    return json_result, bbox_result, vis_axis, false_count

def multi_detect_local(filepath, output_path=None):
    """图面分割本地测试函数"""
    start = time.perf_counter()
    print("正在加载数据，检测器值...")
   
    components, insert_info, = load_data_and_get_main_bbox(filepath)
    
    json_result, bbox_result, vis_axis, false_count = segment_of_multi(components, insert_info)
    print("可视化中...")
    return json_result, bbox_result, vis_axis, false_count

if __name__ == "__main__":
    pass
    # segment_v1("/disk1/user4/work/dbdbd/模拟DAI/test1018/test1018_0.json","result1025.png")
    # segment_v2("/disk1/user4/work/dbdbd/模拟DAI/comap1a/test1114模拟.json","result1/result1125.png")
    # start = time.perf_counter()
    segment_v4_local("/disk1/user4/work/造船厂/结构AI/qzr/output/test_0725v6.json","/disk1/user4/work/造船厂/结构AI/qzr/output/segment.png")
    # end = time.perf_counter()
    # print(end - start)
    # visualize_filtered_bbox("/disk1/user4/work/dbdbd/模拟DAI/comap1a/test1202模拟.json","result/bbox.png")
    # json_result, bbox_result, vis_axis, false_count = multi_detect_local("/disk1/user4/work/造船厂/结构AI/example/多级剖图20250424.json")
    # visualize_multi(bbox_result,"/disk1/user4/work/造船厂/结构AI/v5/output/result.png", *vis_axis)