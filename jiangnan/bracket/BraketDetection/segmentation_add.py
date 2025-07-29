from element import *
import random
import matplotlib.pyplot as plt
from utils import *
from infoextraction2 import *
import numpy as np
from plot_geo import *
from config import *
# from DGCNN_model import *
from tqdm import tqdm
from classifier import *
from draw_dxf import *


def read_json_(json_path, bracket_layer):
    bboxs=[]

    try:  
        with open(json_path, 'r', encoding='utf-8') as file:  
            data_list = json.load(file)
        block_elements=data_list[0]
        for ele in block_elements:
            if ele["layerName"]!=bracket_layer:
                continue
            if ele["type"]=="lwpolyline":
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
                max_x = float('-inf')
                min_x = float('inf')
                max_y = float('-inf')
                min_y = float('inf')
                for seg in poly:
                    # 提取起点和终点的横纵坐标
                    x_coords = [seg.start_point[0], seg.end_point[0]]
                    y_coords = [seg.start_point[1], seg.end_point[1]]

                    # 更新最大最小值
                    max_x = max(max_x, *x_coords)
                    min_x = min(min_x, *x_coords)
                    max_y = max(max_y, *y_coords)
                    min_y = min(min_y, *y_coords)
                bboxs.append((min_x,max_x,min_y,max_y))
                


        
    except FileNotFoundError:  
        print("The file does not exist.")
    except json.JSONDecodeError:  
        print("Error decoding JSON.")
    
    return bboxs
# 读取指定图层bbox
def get_bbox(json_path, bracket_layer):
    texts=[]
    polys=[]
    poly_ids=[]
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
        
    except FileNotFoundError:  
        print("The file does not exist.")
    except json.JSONDecodeError:  
        print("Error decoding JSON.")
    
    return polys

if __name__ == '__main__':
    segmentation_config=SegmentationConfig()
    verbose=segmentation_config.verbose
    json_path = input("请输入路径: ")
    poly_path=input("请输入poly文件路径: ")
    add_bracket_layer_name = input("请输入补充肘板边界图层名：")
    round=input("请输入轮次:")
    segmentation_config.json_path = json_path
    base, ext = os.path.splitext(json_path)
    segmentation_config.multi_json_path = f"{base}_multi.json"
    segmentation_config.remove_layername.append(add_bracket_layer_name)
    output_path=f"{segmentation_config.poly_info_dir}/round{round}"
    segmentation_config.line_image_path = os.path.join(output_path, "line.png")
    segmentation_config.poly_image_dir = os.path.join(output_path, "poly_image")
    segmentation_config.poly_info_dir = os.path.join(output_path)
    segmentation_config.res_image_path = os.path.join(output_path, "res.png")
    segmentation_config.dxf_output_folder = os.path.join(output_path)
    create_folder_safe(f"{segmentation_config.poly_info_dir}")
    create_folder_safe(f"{segmentation_config.poly_info_dir}/标准肘板详细信息参考图")
    create_folder_safe(f"{segmentation_config.poly_info_dir}/所有肘板图像(仅限开发模式)")
    create_folder_safe(f"{segmentation_config.poly_info_dir}/所有有效回路图像")
    create_folder_safe(f"{segmentation_config.poly_info_dir}/非标准肘板")
    create_folder_safe(f"{segmentation_config.poly_info_dir}/标准肘板")
    create_folder_safe(f"{segmentation_config.poly_info_dir}/标准肘板(无分类)")
    if segmentation_config.verbose:
        print("读取json文件")
    # 获取补充肘板的边界
    bb_polys_seg = get_bbox(json_path, add_bracket_layer_name)
    bb_polys = []
    for poly_seg in bb_polys_seg:
        poly = []
        for seg in poly_seg:
            poly.append([seg.start_point.x, seg.start_point.y])
    #文件中线段元素的读取和根据颜色过滤
    elements,segments,ori_segments,stiffeners,sign_handles,polyline_handles,hatch_polys,jg_s=readJson_inbbpolys(json_path,segmentation_config, bb_polys_seg)
    hole_polys = get_hole_text_coor(json_path, segmentation_config.hole_layer)
    ori_block=build_initial_block(ori_segments,segmentation_config)
    # grid,meta=segments_in_blocks(ori_segments,segmentation_config)
    # for row in grid:
    #     rows=[]
    #     for col in row:
    #         rows.append(len(col))
    #     print(rows)


    texts ,dimensions=findAllTextsAndDimensions(elements)
    
    ori_dimensions=dimensions
    dimensions=processDimensions(dimensions)
    texts=processTexts(texts)
    bk_code_pos=find_bkcode(texts)
    if segmentation_config.verbose:
        print("json文件读取完毕")
    

    #找出所有包含角隅孔圆弧的基本环
    polys, new_segments, point_map,star_pos_map,cornor_holes,text_map,removed_handles=findClosedPolys_via_BFS(elements,texts,dimensions,segments,sign_handles,segmentation_config)

    # #预训练的几何分类模型筛选肘板
    # model_path = "/home/user4/BraketDetection/DGCNN/cpkt/geometry_classifier.pth"
    # polys = filter_by_pretrained_DGCNN_Model(polys, model_path)

    # print("DGCNN筛选后剩余回路: ", len(polys))
    # outputPolysAndGeometry(polys,segmentation_config.poly_image_dir,segmentation_config.draw_polys,segmentation_config.draw_geometry,segmentation_config.draw_poly_nums)

    #结构化输出每个肘板信息
    edges_infos,poly_centroids,hint_infos,meta_infos=[],[],[],[]
    indices=[]
    pbar=tqdm(total=len(polys),desc="正在输出结构化信息")
    for i, poly in enumerate(polys):
        # try:
        #     res = outputPolyInfo(poly, ori_segments, segmentation_config, point_map, i, star_pos_map, cornor_holes,texts,dimensions,text_pos_map)
        # except Exception as e:
        #     res=None

        #     print(e)
        # segments_nearby,blocks=segments_near_poly(poly,grid,meta)
        
        # visualize_grid_and_segment(segments_nearby, poly,meta[0],meta[1],meta[2], blocks)
        # print(len(segments_nearby))
        # try:
        #     segments_nearby=ori_block.segments_near_poly(poly)
        #     res = outputPolyInfo(poly, segments_nearby, segmentation_config, point_map, i, star_pos_map, cornor_holes,texts,dimensions,text_map,stiffeners)
        # except Exception as e:
        #     res=None
        segments_nearby=ori_block.segments_near_poly(poly)
        res = calculate_poly_features(poly, segments_nearby, segmentation_config, point_map, i, star_pos_map, cornor_holes,texts,dimensions,text_map,stiffeners,hatch_polys,hole_polys,jg_s)
        pbar.update()
        if res is not None:
            # print(res)
            edges_info,poly_centroid,hint_info,meta_info=res
            edges_infos.append(edges_info)
            poly_centroids.append(poly_centroid)
            hint_infos.append(hint_info)
            meta_infos.append(meta_info)
            indices.append(i)
    pbar.close()
    
    code_map=calculate_codemap(edges_infos,poly_centroids,hint_infos,meta_infos,bk_code_pos)

    edges_infos,poly_centroids,hint_infos,meta_infos=hint_search_step(edges_infos,poly_centroids,hint_infos,meta_infos,code_map)
  
    edges_infos,poly_centroids,hint_infos,meta_infos=diffusion_step(edges_infos,poly_centroids,hint_infos,meta_infos)

    polys_info,classi_res,flags,all_json_data=classificationAndOutputStep(indices,edges_infos,poly_centroids,hint_infos,meta_infos,segmentation_config,polys,polyline_handles)
    
    # 获得需要去重肘板的id
    delete_bracket_ids = find_dump_bracket_ids(polys_info, classi_res, indices)
    
    free_edge_handles = []
    all_handles=[]
    not_all_handles=[]
    non_free_edge_handles = []
    for idx,(poly_refs,cls,flag) in enumerate(zip(polys_info,classi_res,flags)):
        if cls=='Unclassified' or cls=='Unstandard'  or ',' in cls  or 'ustd' in cls:
            continue
        else:
            for seg in poly_refs:
                if seg.isConstraint == False and seg.isCornerhole == False:
                    free_edge_handles.append(seg.ref.handle)
                else:
                    non_free_edge_handles.append(seg.ref.handle)
            if len(cls.split(','))==1 and flag:
                for seg in poly_refs:
                    all_handles.append(seg.ref.handle)
            else:
                for seg in poly_refs:
                    not_all_handles.append(seg.ref.handle)
    bboxs = []
    add_bboxs=[]
    add_ids=[]
    for idx,(poly_refs,cls) in enumerate(zip(polys_info,classi_res)):
        max_x = float('-inf')
        min_x = float('inf')
        max_y = float('-inf')
        min_y = float('inf')
        for seg in poly_refs:
            # 提取起点和终点的横纵坐标
            x_coords = [seg.start_point[0], seg.end_point[0]]
            y_coords = [seg.start_point[1], seg.end_point[1]]

            # 更新最大最小值
            max_x = max(max_x, *x_coords)
            min_x = min(min_x, *x_coords)
            max_y = max(max_y, *y_coords)
            min_y = min(min_y, *y_coords)

        bbox = [[min_x, min_y], [max_x, max_y]]
        bboxs.append(bbox)
    
        if cls=='Unclassified' or cls=='Unstandard'  or ',' in cls  or 'ustd' in cls:
            continue
        add_bboxs.append((min_x-20,max_x+20,min_y-20,max_y+20))
        add_ids.append(indices[idx])
    
    remain_bboxs=read_json_(json_path,"Braket")
    count,old_bboxs,old_ids=read_bboxes_with_ids(poly_path)
    b_set=set()
    for bbox in remain_bboxs:
        x_min,x_max,y_min,y_max=bbox
        b_set.add((int(x_min),int(x_max),int(y_min),int(y_max)))
    actual_bboxs=[]
    actual_ids=[]
    for i,bbox in enumerate(old_bboxs):
        x_min,x_max,y_min,y_max=bbox
        if (int(x_min),int(x_max),int(y_min),int(y_max)) not in b_set:
            #删除包围盒

            id=old_ids[i]
        else:
            id=old_ids[i]
            actual_bboxs.append(bbox)
            actual_ids.append(id)

    print(len(remain_bboxs),len(old_bboxs),len(add_bboxs))
    actual_bboxs=actual_bboxs+add_bboxs
    for id in add_ids:
        actual_ids.append(id+count)
    write_bboxes_with_ids(os.path.join(segmentation_config.dxf_output_folder, f"polys.txt"),actual_bboxs,actual_ids,count+len(bboxs))
    
    dxf_path = os.path.splitext(segmentation_config.json_path)[0] + '.dxf'
    dxf_output_folder = segmentation_config.dxf_output_folder
    draw_rectangle_in_dxf(dxf_path, dxf_output_folder, bboxs, classi_res,indices, free_edge_handles,non_free_edge_handles,all_handles,not_all_handles,removed_handles,delete_bracket_ids)
