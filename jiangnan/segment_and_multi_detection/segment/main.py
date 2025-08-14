from segment_and_multi_detection.segment.utils import *
import time
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
    filtered_all_comps, x_min_tmp, y_min_tmp, x_max_tmp, y_max_tmp = filter_bounding_boxes(components, threshold=0.9)
    
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
    filtered_all_comps, x_min_tmp, y_min_tmp, x_max_tmp, y_max_tmp = filter_bounding_boxes(components, threshold=0.9)

    # 查找数据的主要
    print("查找数据中")
    sub_comps_list, re_search_index = search_titles(filtered_all_comps)
    print("去除重复元素")
    unmerged_components_list = remove_duplicate_comps(sub_comps_list)
    merged_components_list = merge_nearest_comp(unmerged_components_list, filtered_all_comps, re_search_index)

    # 解决大图包含小图的问题
    final_components_list = solve_big_contain_small(merged_components_list)

    #多级剖图主函数
    result_json, result_bbox = detect(final_components_list,filtered_all_comps)

    # 返回处理结果和当前的包覆盖范围
    return result_json, result_bbox, final_components_list

def segment_v4(filepath, output_path=None):
    """图面分割函数入口
    filepath: json文件路径"""
    start = time.perf_counter()
    print("正在加载数据，检测阈值...")
    components= load_data_and_get_main_bbox(filepath)
    # components, _, = load_data_and_get_main_bbox_local(filepath)
    
    final_components_list, vis_axis = segment_separately(components)

    print("可视化中...")
    visualize_many_bbox(final_components_list, *vis_axis, output_path)
    final_components_list = [item for sublist in final_components_list for item in (sublist if isinstance(sublist, list) else [sublist])]
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
#OLD0805
def segment_v4_bound(filepath, output_path=None,bound_json=None):
    """图面分割函数入口
    filepath: json文件路径"""
    start = time.perf_counter()
    print("正在加载数据，检测阈值...")
    components = load_data_and_get_main_bbox_boound(filepath, threshold=0.1,bound_json=bound_json)
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
#OLD0805
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