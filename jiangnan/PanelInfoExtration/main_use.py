from PanelInfoExtration.element import *
import random
import matplotlib.pyplot as plt
from matplotlib.path import Path
from PanelInfoExtration.utils import *
from PanelInfoExtration.infoextraction2 import *
import numpy as np
from PanelInfoExtration.plot_geo import *
from PanelInfoExtration.config import *
from tqdm import tqdm
from PanelInfoExtration.classifier import *
from PanelInfoExtration.draw_dxf import *
import pandas as pd
from PanelInfoExtration.find_panel import *
import json
import os
from io import StringIO
from PanelInfoExtration.example import *
from bracket.BraketDetection.load import dxf2json



def main(temp_dir, dxf_name, excel_name_1, excel_name_2):


    dxf_name = dxf_name.split(".")[0]
    segmentation_config=SegmentationConfig()
    verbose=segmentation_config.verbose
    dxf2json(temp_dir,dxf_name,temp_dir)
    json_path = os.path.join(temp_dir, f"{dxf_name}.json")
    # json_path = input("请输入json路径: ")
    segmentation_config.json_path = json_path
    # base, ext = os.path.splitext(json_path)
    # segmentation_config.multi_json_path = f"{base}_multi.json"
    
    # 输出准备
    json_stream = StringIO()
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    if segmentation_config.verbose:
        print("读取json文件")
    # 读取图纸，文件中线段元素的读取和根据颜色过滤
    elements,segments,ori_segments,stiffeners,sign_handles,polyline_handles, hatch_polys,jg_s = readJson(json_path,segmentation_config)

    hole_polys = get_hole_text_coor(json_path, segmentation_config.hole_layer)
    # print(f"图案填充个数：{len(hatch_polys)}")
    # print(sign_handles)
    ori_block=build_initial_block(ori_segments,segmentation_config)

    texts, dimensions = findAllTextsAndDimensions(elements)
    
    ori_dimensions = dimensions
    dimensions = processDimensions(dimensions)
    texts = processTexts(texts)
    bk_code_pos = find_bkcode(texts)
    if segmentation_config.verbose:
        print("json文件读取完毕")
        
    
    # 读取excel文件 信息
    
    # panel_path = input("请输入panel_info路径: ")
    # segment_path = input("请输入segment_info路径: ")
    panel_path = os.path.join(temp_dir, excel_name_1)
    segment_path = os.path.join(temp_dir, excel_name_2)
    # 读取回路曲线组成文件
    panel_info = pd.read_excel(panel_path, engine='openpyxl')
    # 读取线条句柄和坐标文件
    segments_info = pd.read_excel(segment_path, engine='openpyxl')
    

    #找出所有包含角隅孔圆弧的基本回路
    #polys, new_segments, point_map, star_pos_map, cornor_holes, text_map, removed_handles = findClosedPolys_via_BFS(elements,texts,dimensions,segments,sign_handles,segmentation_config)
    #polys, new_segments, point_map, star_pos_map, cornor_holes, text_map, removed_handles = findPolys_via_excel(panel_info, segments_info, elements,texts,dimensions,segments,sign_handles,segmentation_config)

    # #test输出Polys
    # save_polys_as_csv(polys, "output_polygons.csv")
    
    # vertices = polys.vertices  # 获取顶点坐标
    # np.savetxt('polygon_vertices.csv', vertices, delimiter=',')  # 保存为CSV

    #结构化输出每个板信息
    # edges_infos,poly_centroids,hint_infos,meta_infos=[],[],[],[]
    # indices=[]
    # pbar=tqdm(total=len(polys),desc="正在输出结构化信息")
    
    # for i, poly in enumerate(polys):
    #     segments_nearby=ori_block.segments_near_poly(poly)
    #     res = calculate_poly_features(poly, segments_nearby, segmentation_config, point_map, i, star_pos_map, cornor_holes,texts,dimensions,text_map,stiffeners, hatch_polys,hole_polys,jg_s)
    #     pbar.update()
    #     if res is not None:
    #         # print(res)
    #         edges_info,poly_centroid,hint_info,meta_info=res
    #         edges_infos.append(edges_info)
    #         poly_centroids.append(poly_centroid)
    #         hint_infos.append(hint_info)
    #         meta_infos.append(meta_info)
    #         indices.append(i)
    # pbar.close()
    
    # code_map=calculate_codemap(edges_infos,poly_centroids,hint_infos,meta_infos,bk_code_pos)

    # edges_infos,poly_centroids,hint_infos,meta_infos=hint_search_step(edges_infos,poly_centroids,hint_infos,meta_infos,code_map)
  
    # edges_infos,poly_centroids,hint_infos,meta_infos=diffusion_step(edges_infos,poly_centroids,hint_infos,meta_infos)

    # polys_info,classi_res,flags=classificationAndOutputStep(indices,edges_infos,poly_centroids,hint_infos,meta_infos,segmentation_config,polys,polyline_handles)

   
    # bboxs = []
    # actual_bboxs=[]
    # actual_ids=[]
    # for idx,(poly_refs,cls) in enumerate(zip(polys_info,classi_res)):
    #     max_x = float('-inf')
    #     min_x = float('inf')
    #     max_y = float('-inf')
    #     min_y = float('inf')
    #     for seg in poly_refs:
    #         # 提取起点和终点的横纵坐标
    #         x_coords = [seg.start_point[0], seg.end_point[0]]
    #         y_coords = [seg.start_point[1], seg.end_point[1]]

    #         # 更新最大最小值
    #         max_x = max(max_x, *x_coords)
    #         min_x = min(min_x, *x_coords)
    #         max_y = max(max_y, *y_coords)
    #         min_y = min(min_y, *y_coords)

    #     bbox = [[min_x, min_y], [max_x, max_y]]
    #     bboxs.append(bbox)
    
    #     if cls=='Unclassified' or cls=='Unstandard'  or ',' in cls  or 'ustd' in cls:
    #         continue
    #     actual_bboxs.append((min_x-20,max_x+20,min_y-20,max_y+20))
    #     actual_ids.append(indices[idx])
    # write_bboxes_with_ids(os.path.join(segmentation_config.dxf_output_folder, f"polys.txt"),actual_bboxs,actual_ids,len(bboxs))
    
    # dxf_path = os.path.splitext(segmentation_config.json_path)[0] + '.dxf'
    # dxf_output_folder = segmentation_config.dxf_output_folder
    # draw_rectangle_in_dxf(dxf_path, dxf_output_folder, bboxs, classi_res,indices, free_edge_handles,non_free_edge_handles,all_handles,not_all_handles,removed_handles,delete_bracket_ids)




    # 输出文件
    # # 返回JSON流
    json_stream = create_example_json_stream()
    json_stream.seek(0)  # 确保指针回到开头
    jsonlist = []
    jsonlist.append(json.load(json_stream))  # 直接解析

    # print(jsonlist)
    
    # path_1 = os.path.join(temp_dir, "output/example_output.json")
    # saved_path = save_json_to_file(json_stream, path_1)
    # json_stream.seek(0)  # 重置指针以便读取
    
    # 创建包含标注数据的表格对象
    Panel_Anno_Info = PanelAnno()
    # 生成Excel文件
    # path_2 = os.path.join(temp_dir, "output/example.xlsx")
    # create_anno_excel(Panel_Anno_Info, path_2)

    return jsonlist
    
    
    
    

    
if __name__ == '__main__':
    # 示例用法
    file_path = "./test_path"
    dxf_file = "example_v3.dxf"
    panel_excel = "STR_DRW_CURVE_INFO.xlsx"
    segment_excel = "STR_DRW_PANEL_LIMIT_INFO.xlsx"
    
    result = main(file_path, dxf_file, panel_excel, segment_excel)