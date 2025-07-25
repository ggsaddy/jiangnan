from  element import *
from plot_geo import plot_geometry,plot_polys, plot_info_poly,p_minus,p_add,p_mul
import os
from utils import segment_intersection_line,segment_intersection,computeBoundingBox,is_parallel,conpute_angle_of_two_segments,point_segment_position,shrinkFixedLength,check_points_against_segments,check_points_against_free_segments,check_parallel_anno,check_vertical_anno,check_non_parallel_anno,check_whole
from classifier import poly_classifier,match_template,load_classification_table,eva_c_f,poly_classifier_ustd
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString
from bracket_parameter_extraction import parse_elbow_plate
import json
import re
import numpy as np
import csv
import codecs
import sys
import ast

def is_point_in_polygon(point, polygon_edges):
    
    polygon_points = set()  # Concave polygon example
    for edge in polygon_edges:
        vs,ve=edge.start_point,edge.end_point
        polygon_points.add((vs.x,vs.y))
        polygon_points.add((ve.x,ve.y))

    polygon_points = list(polygon_points)

    polygon = Polygon(polygon_points)

    point = Point(point.x, point.y)

    # Check if the point is inside the polygon
    is_inside = polygon.contains(point)

    return is_inside
def log_to_file(filename, content):
    """将内容写入指定的文件。"""
    with open(filename, 'a', encoding='utf-8') as file:  # 以追加模式打开文件
        file.write(content + '\n')  # 写入内容并换行

def clear_file(file_path):
    """清空指定的文件"""
    try:
        with open(file_path, 'w') as file:
            file.truncate(0)  # 清空文件内容
    except Exception as e:
        print(f"无法清空文件 '{file_path}'。错误: {e}")


def polygon_area(points):
    """
    计算多边形的面积，基于顶点的顺序。
    :param points: 多边形顶点列表 [(x1, y1), (x2, y2), ...]。
    :return: 多边形面积（绝对值）。
    """
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - y1 * x2
    return abs(area) / 2.0

def is_near_convex(points, i, tolerance=0.1):
    """
    判断多边形是否是近似凸多边形（基于面积差）。
    :param points: 多边形顶点列表，按顺序排列 [(x1, y1), (x2, y2), ...]。
    :param i: 当前多边形的索引，用于命名图片文件。
    :param tolerance: 面积差的允许百分比，默认为 5%。
    :return: True 如果是近似凸多边形，否则 False。
    """
    if len(points) < 3:
        return False  # 不构成多边形

    # 确保输入点集是一个二维数组
    points = [(float(x), float(y)) for x, y in points]

    # 计算输入多边形面积
    poly_area = polygon_area(points)

    # 计算凸包的面积
    try:
        convex_hull = ConvexHull(points)
        hull_area = convex_hull.volume  # ConvexHull 的面积
        # 检查面积差
        #print(poly_area,hull_area,abs(poly_area - hull_area) / hull_area)
        if abs(poly_area - hull_area) / hull_area > tolerance:
            return False
    except Exception as e:
        pass

    return True

def is_near_rectangle(points, area, tolerance = 0.01):
    if len(points) < 3:
        return False  # 不构成多边形

    # 确保输入点集是一个二维数组
    points = [(float(x), float(y)) for x, y in points]

    # 计算输入多边形面积
    poly_area = polygon_area(points)
    
    if abs(poly_area - area) / area > tolerance:
        return False

    return True


def calculate_angle(point1, point2, point3):
    """
    计算由三点确定的两向量之间的夹角（以度为单位）。

    Parameters:
        point1, point2, point3: 各为一个元组，表示点的坐标 (x, y)。

    Returns:
        angle: 两向量之间的夹角（较小的角，范围为 0-180°）。
    """
    def vector(a, b):
        return (b[0] - a[0], b[1] - a[1])

    def magnitude(v):
        return math.sqrt(v[0]**2 + v[1]**2)

    def dot_product(v1, v2):
        return v1[0] * v2[0] + v1[1] * v2[1]

    v1 = vector(point2, point1)
    v2 = vector(point2, point3)

    mag_v1 = magnitude(v1)
    mag_v2 = magnitude(v2)

    if mag_v1 == 0 or mag_v2 == 0:
        return 0  # 防止除零错误

    cos_theta = dot_product(v1, v2) / (mag_v1 * mag_v2)
    cos_theta = max(-1, min(1, cos_theta))  # 修正浮点误差

    angle = math.acos(cos_theta) * (180 / math.pi)  # 弧度转角度
    return angle

# 计算是否位于阴影区域
def is_intersect_hatch(poly, hatch_polys):
    # 提取poly的所有端点坐标
    poly_geom = computePolygon(poly)

    # 遍历所有 hatch 区域，判断是否相交
    try:
        for hatch in hatch_polys:
            hatch_points = [tuple(p) for p in hatch]
            if len(hatch_points) < 3:
                continue  # 不能构成面
            if hatch_points[0] != hatch_points[-1]:
                hatch_points.append(hatch_points[0])  # 确保闭合

            hatch_geom = Polygon(hatch_points)

            intersection = poly_geom.intersection(hatch_geom)

            if intersection.area / poly_geom.area > 0.5:
                return True
    except Exception as e:
        pass

    return False

def is_intersect_hole(poly, hole_text_coors):
    # 提取poly的所有端点坐标
    poly_geom = computePolygon(poly)
    for coor in hole_text_coors:
        if poly_geom.contains(Point(coor[0], coor[1])):
            return True
        
    return False

# 计算几何中心坐标
def calculate_poly_centroid(poly,):
    points = []
    for segment in poly:
        points.append(segment.start_point)
        points.append(segment.end_point)
        x = sum(point.x for point in points) / len(points)
        y = sum(point.y for point in points) / len(points)
    return (x, y)

def calculate_combined_ref(segments,segmentation_config):
    for i,s in enumerate(segments):
        if s.ref.color in segmentation_config.constraint_color:
            return s.ref
    return segments[0].ref
def combine_the_same_line(poly,segmentation_config,ref_map):
    n=len(poly)
    new_poly=[]
    for i in range(len(poly)):
        if isinstance(poly[i].ref,DArc):
            continue
        for j in range(len(poly)):
            if isinstance(poly[j].ref,DArc):
                continue
            if i>=j:
                continue
            #j>i
            

            s1,s2=poly[i],poly[j]
            #print(i,j,point_segment_position(s1.start_point,s2,anno=False),point_segment_position(s1.end_point,s2,anno=False))
            if point_segment_position(s1.start_point,s2,epsilon=0.2,anno=False)!="not_on_line" and point_segment_position(s1.end_point,s2,epsilon=0.2,anno=False)!="not_on_line":
                # print(i,j)
                if j==i+1:
                    if s1.end_point ==s2.start_point:
                        ref=calculate_combined_ref([s1,s2],segmentation_config)
                        new_segment=DSegment(s1.start_point,s2.end_point,ref)
                        for k in range(i):
                            new_poly.append(poly[k])
                        new_poly.append(new_segment)
                        ref_map[new_segment]=[]
                        if s1 in ref_map:
                            ref_map[new_segment].extend(ref_map[s1])
                        else:
                            ref_map[new_segment].append(s1)
                        if s2 in ref_map:
                            ref_map[new_segment].extend(ref_map[s2])
                        else:
                            ref_map[new_segment].append(s2)
                        for k in range(j+1,n):
                            new_poly.append(poly[k])
                        return new_poly,True
                    else:
                        ref=calculate_combined_ref([s1,s2],segmentation_config)
                        new_segment=DSegment(s2.start_point,s1.end_point,ref)

                        for k in range(i):
                            new_poly.append(poly[k])
                        new_poly.append(new_segment)
                        ref_map[new_segment]=[]
                        if s1 in ref_map:
                            ref_map[new_segment].extend(ref_map[s1])
                        else:
                            ref_map[new_segment].append(s1)
                        if s2 in ref_map:
                            ref_map[new_segment].extend(ref_map[s2])
                        else:
                            ref_map[new_segment].append(s2)
                        for k in range(j+1,n):
                            new_poly.append(poly[k])
                        return new_poly,True
                if i==0 and j==n-1:
                    if s1.end_point ==s2.start_point:
                        ref=calculate_combined_ref([s1,s2],segmentation_config)
                        new_segment=DSegment(s1.start_point,s2.end_point,ref)
                        new_poly.append(new_segment)
                        ref_map[new_segment]=[]
                        if s1 in ref_map:
                            ref_map[new_segment].extend(ref_map[s1])
                        else:
                            ref_map[new_segment].append(s1)
                        if s2 in ref_map:
                            ref_map[new_segment].extend(ref_map[s2])
                        else:
                            ref_map[new_segment].append(s2)
                        for k in range(1,n-1):
                            new_poly.append(poly[k])
                        return new_poly,True
                    else:
                        ref=calculate_combined_ref([s1,s2],segmentation_config)
                        new_segment=DSegment(s2.start_point,s1.end_point,ref)

                        new_poly.append(new_segment)
                        ref_map[new_segment]=[]
                        if s1 in ref_map:
                            ref_map[new_segment].extend(ref_map[s1])
                        else:
                            ref_map[new_segment].append(s1)
                        if s2 in ref_map:
                            ref_map[new_segment].extend(ref_map[s2])
                        else:
                            ref_map[new_segment].append(s2)
                        for k in range(1,n-1):
                            new_poly.append(poly[k])
                        return new_poly,True
                
                # if s1.length()<150 or s2.length()<150:
                #     continue
                # if point_segment_position(s1.start_point,s2,epsilon=0.1,anno=False)=="not_on_line" or point_segment_position(s1.end_point,s2,epsilon=0.1,anno=False)=="not_on_line":
                #     continue
                # inner_l,outer_l=0,0
                # for k in range(i+1,j):
                #     inner_l+=poly[k].length()
                # for k in range(i):
                #     outer_l+=poly[k].length()
                # for k in range(j+1,n):
                #     outer_l+=poly[k].length()
                # if inner_l<outer_l:
                #     #remove inner
                #     prev,next=poly[(i-1+n)%n],poly[(j+1)%n]
                #     if prev.end_point == s1.start_point:
                #         ref=calculate_combined_ref([s1,s2],segmentation_config)
                #         new_segment=DSegment(s1.start_point,s2.end_point,ref)
                #         for k in range(i):
                #             new_poly.append(poly[k])
                #         new_poly.append(new_segment)
                #         for k in range(j+1,n):
                #             new_poly.append(poly[k])
                #         return new_poly,True
                #     else:
                #         #prev.start_point==s1.end_point
                #         ref=calculate_combined_ref([s1,s2],segmentation_config)
                #         new_segment=DSegment(s2.start_point,s1.end_point,ref)
                #         for k in range(i):
                #             new_poly.append(poly[k])
                #         new_poly.append(new_segment)
                #         for k in range(j+1,n):
                #             new_poly.append(poly[k])
                #         return new_poly,True
                # else:
                #     #remove outer
                #     prev,next=poly[(i-1+n)%n],poly[(j+1)%n]
                #     if prev.end_point == s1.start_point:
                #         ref=calculate_combined_ref([s1,s2],segmentation_config)
                #         new_segment=DSegment(s2.start_point,s1.end_point,ref)
                #         for k in range(i+1,j):
                #             new_poly.append(poly[k])
                #         new_poly.append(new_segment)
                #         return new_poly,True
                #     else:
                #         ref=calculate_combined_ref([s1,s2],segmentation_config)
                #         new_segment=DSegment(s1.start_point,s2.end_point,ref)
                #         for k in range(i+1,j):
                #             new_poly.append(poly[k])
                #         new_poly.append(new_segment)
                #         return new_poly,True
    return poly,False
def calculate_poly_refs(poly,segmentation_config):
    new_poly=poly
    old_poly=poly
    ref_map={}
    #combine the same line in the poly
    # print("===================")
    # print(len(poly))
    while True:

        
        new_poly,flag=combine_the_same_line(old_poly,segmentation_config,ref_map)
        # print(len(new_poly))
        if flag==False:
            break
        old_poly=new_poly
    # print("===========================")

    refs = []
    
    
    for segment in new_poly:
        if isinstance(segment.ref, DArc):
            if len(refs) != 0 and isinstance(refs[-1].ref, DArc):
                if (segment.ref.start_angle, segment.ref.end_angle, segment.ref.center, segment.ref.radius) == (
                    refs[-1].ref.start_angle, refs[-1].ref.end_angle, refs[-1].ref.center, refs[-1].ref.radius):
                    ref_map[refs[-1]].append(segment)
                    continue
            refs.append(segment)
            ref_map[segment]=[segment]
        else:
            # if len(refs) != 0:
            #     last_segment = refs[-1]
            #     # 判断是否平行
            #     if is_parallel(last_segment, segment,segmentation_config.is_parallel_tolerance) and not isinstance(last_segment.ref, DArc):
            #         seg=last_segment if last_segment.length()>segment.length() else segment
            #         new_segment = DSegment(
            #             start_point=last_segment.start_point,
            #             end_point=segment.end_point,
            #             ref=seg.ref
            #         )
            #         refs[-1] = new_segment
            #         continue
            refs.append(segment)

    # 判断首尾是否可以合并
    if len(refs) > 1:
        first_segment = refs[0]
        last_segment = refs[-1]

        if isinstance(first_segment.ref, DArc) and isinstance(last_segment.ref, DArc):
            if (first_segment.ref.start_angle, first_segment.ref.end_angle, first_segment.ref.center, first_segment.ref.radius) == (
                last_segment.ref.start_angle, last_segment.ref.end_angle, last_segment.ref.center, last_segment.ref.radius):
                refs.pop()
                ref_map[first_segment].append(last_segment)
        # elif not isinstance(first_segment.ref, DArc) and not isinstance(last_segment.ref, DArc):
        #     if is_parallel(first_segment, last_segment,segmentation_config.is_parallel_tolerance):
        #         new_segment = DSegment(
        #             start_point=last_segment.start_point,
        #             end_point=first_segment.end_point,
        #             ref=last_segment.ref
        #         )
        #         refs[0] = new_segment
        #         refs.pop()

    return refs,ref_map


def computePolygon(poly,tolerance = 0.1):
    polygon_points = list()  # Concave polygon example
    for edge in poly:
        vs,ve=edge.start_point,edge.end_point
        polygon_points.append((vs.x,vs.y))
        polygon_points.append((ve.x,ve.y))
    polygon = Polygon(polygon_points)
    
    # polygon_with_tolerance = polygon.buffer(tolerance)

    # return polygon_with_tolerance
    return polygon
def point_is_inside(point,polygon):
    point_=Point(point.x,point.y)
    return polygon.contains(point_)


def stiffenersInPoly(stiffeners,poly,segmentation_config):
    # Define your polygon (list of vertices)
    sf=[]
    x_min,x_max,y_min,y_max=computeBoundingBox(poly)
    if segmentation_config.bracket_bbox_expand_is_ratio:
        xx=(x_max-x_min)*segmentation_config.bracket_bbox_expand_ratio
        yy=(y_max-y_min)*segmentation_config.bracket_bbox_expand_ratio
    else:
        xx=segmentation_config.bracket_bbox_expand_length
        yy=segmentation_config.bracket_bbox_expand_length
    x_min=x_min-xx
    x_max=x_max+xx
    y_max=y_max+yy
    y_min=y_min-yy
    # polygon_points = set()  # Concave polygon example
    # for edge in poly:
    #     vs,ve=edge.start_point,edge.end_point
    #     polygon_points.add((vs.x,vs.y))
    #     polygon_points.add((ve.x,ve.y))
    # polygon_points = list(polygon_points)
    # polygon = Polygon(polygon_points)
    # tolerance = 0.1 # 误差值
    # polygon_with_tolerance = polygon.buffer(tolerance)
    for stiffener in stiffeners:
        mid=stiffener.mid_point()
        # point = Point(mid.x,mid.y)
        
        # if polygon_with_tolerance.contains(point):
        #     sf.append(stiffener)
        if x_min <= mid.x and mid.x <=x_max and y_min <=mid.y and y_max>=mid.y:
            sf.append(stiffener)
    return sf
def textsInPoly(text_map,poly,segmentation_config,is_fb,polygon):
    x_min,x_max,y_min,y_max=computeBoundingBox(poly)
    if segmentation_config.bracket_bbox_expand_is_ratio:
        xx=(x_max-x_min)*segmentation_config.bracket_bbox_expand_ratio
        yy=(y_max-y_min)*segmentation_config.bracket_bbox_expand_ratio
    else:
        xx=segmentation_config.bracket_bbox_expand_length
        yy=segmentation_config.bracket_bbox_expand_length
    x_min=x_min-xx
    x_max=x_max+xx
    y_max=y_max+yy
    y_min=y_min-yy
    ts=[]
    for pos,texts in text_map.items():
        for t in texts:
            if t[1]["Type"]=="FB" or t[1]["Type"]=="FL" or t[1]["Type"]=="B" or t[1]["Type"]=="BK" or (t[1]["Type"]=="R" and t[1]["Ref"]=="ref"):

                if point_is_inside(pos,polygon):
                    if t[1]["Type"]=="FB" or t[1]["Type"]=="FL":
                        result=parse_elbow_plate(t[0].content, "bottom",is_fb)
                    else:
                        result=t[1]
                    ts.append([t[0],pos,result,t[2]])#element,position,result,anotation
            elif x_min <= pos.x and pos.x <=x_max and y_min <=pos.y and y_max>=pos.y:
                result=t[1]
                ts.append([t[0],pos,result,t[2]])#element,position,result,anotation


    #filter text
    text_set=set()
    new_ts=[]
    for t_t in ts:
        t=t_t[0]
        if (t.handle,t.content) not in text_set:
            new_ts.append(t_t)
            text_set.add((t.handle,t.content))
    ts=new_ts
    new_ts=[]
    b_count=0
    fb_count=0

    for t_t in ts:
        content=t_t[0].content.strip()
        if "FB" in content or "FL" in content:
            result=parse_elbow_plate(content, "bottom",is_fb)
            if result is None:
                new_ts.append([t_t[0],t_t[1],{"Type":"None"},t_t[3]])
            else:
                new_ts.append([t_t[0],t_t[1],result,t_t[3]])
                fb_count+=1
        elif t_t[2]["Type"]=="B":
            new_ts.append(t_t)
            b_count+=1
        elif t_t[2]["Type"]=="FB" or t_t[2]["Type"]=="FL":
            new_ts.append(t_t)
            fb_count+=1
        else:
            new_ts.append(t_t)
    ts=new_ts
    new_ts=[]
    for t_t in ts:
        if t_t[2]["Type"]=="None" and (b_count==0 or fb_count==0):
            content=t_t[0].content.strip()
            if (content[0]=="$" or content[0]=="~" or content[0]=="&" or content[0]=="%" or content[0]=="#") and len(content)>1:
                if b_count==0:
                    result=parse_elbow_plate(content, "top",is_fb)
                    if result is None:
                        new_ts.append([t_t[0],t_t[1],{"Type":"None"},t_t[3]])
                    else:
                        new_ts.append([t_t[0],t_t[1],result,t_t[3]])
                        b_count+=1
                elif fb_count==0:
                    result=parse_elbow_plate(content, "bottom",is_fb)
                    if result is None:
                        new_ts.append([t_t[0],t_t[1],{"Type":"None"},t_t[3]])
                    else:
                        new_ts.append([t_t[0],t_t[1],result,t_t[3]])
                        fb_count+=1
            elif ("$" in content or "~" in content or "&" in content or "%" in content  or "#" in content) and len(content)>2:
                if b_count==0:
                    result=parse_elbow_plate(content, "top",is_fb)
                    if result is None:
                        new_ts.append([t_t[0],t_t[1],{"Type":"None"},t_t[3]])
                    else:
                        new_ts.append([t_t[0],t_t[1],result,t_t[3]])
                        b_count+=1
                elif fb_count==0:
                    result=parse_elbow_plate(content, "bottom",is_fb)
                    if result is None:
                        new_ts.append([t_t[0],t_t[1],{"Type":"None"},t_t[3]])
                    else:
                        new_ts.append([t_t[0],t_t[1],result,t_t[3]])
                        fb_count+=1
            else:
                new_ts.append(t_t)
        else:
            new_ts.append(t_t)

    return new_ts

def braketTextInPoly(braket_texts,braket_pos,poly,segmentation_config):
    x_min,x_max,y_min,y_max=computeBoundingBox(poly)
    if segmentation_config.bracket_bbox_expand_is_ratio:
        xx=(x_max-x_min)*segmentation_config.bracket_bbox_expand_ratio
        yy=(y_max-y_min)*segmentation_config.bracket_bbox_expand_ratio
    else:
        xx=segmentation_config.bracket_bbox_expand_length
        yy=segmentation_config.bracket_bbox_expand_length
    x_min=x_min-xx
    x_max=x_max+xx
    y_max=y_max+yy
    y_min=y_min-yy
    ts=[]
    bps=[]
    for i,t in enumerate(braket_pos):
        pos=t
        if x_min <= pos.x and pos.x <=x_max and y_min <=pos.y and y_max>=pos.y:
            ts.append(braket_texts[i])
            bps.append(braket_pos[i])
    return ts,bps


def dimensionsInPoly(dimensions,poly,segmentation_config):
    x_min,x_max,y_min,y_max=computeBoundingBox(poly)
    if segmentation_config.bracket_bbox_expand_is_ratio:
        xx=(x_max-x_min)*segmentation_config.bracket_bbox_expand_ratio
        yy=(y_max-y_min)*segmentation_config.bracket_bbox_expand_ratio
    else:
        xx=segmentation_config.bracket_bbox_expand_length
        yy=segmentation_config.bracket_bbox_expand_length
    x_min=x_min-xx
    x_max=x_max+xx
    y_max=y_max+yy
    y_min=y_min-yy
    ds=[]
    for d_t in dimensions:
        pos=d_t[1]
        d=d_t[0]
        if x_min <= pos.x and pos.x <=x_max and y_min <=pos.y and y_max>=pos.y:
            ds.append([d,pos])
    return ds
def match_r_anno(r_anno,free_edges):
    r_map={}
    d_map={}
    radius_anno=[]
    corner_edges=[]
    s_free_edges=[]
    for s in free_edges[0]:
        if (not s.isConstraint) and (not s.isCornerhole):
            s_free_edges.append(s)
        elif (s.isCornerhole) and (not s.isConstraint):
            corner_edges.append(s)
    #标注值>90距离在1000以内的自由边半径标注初步分配(最邻近)
    for r_t in r_anno:
        pos,content=r_t
        result=parse_elbow_plate(content.content.strip())
        if result is not None and result["Type"]=="R" and result["Radius"]>89:
            min_distance=float("inf")
            target=None
            for s in s_free_edges:
                if isinstance(s.ref,DArc):
                    center=DSegment(s.ref.start_point,s.ref.end_point).mid_point()
                    distance=DSegment(center,pos).length()
                    if distance>1000:
                        continue
                    if target is None:
                        target=s
                        min_distance=min(min_distance,distance)
                    else:
                        if distance<min_distance:
                            min_distance=distance
                            target=s
            if target is not None:
                if target not in r_map:
                    r_map[target]=content
                    d_map[target]=min_distance
                elif min_distance<d_map[target]:
                    r_map[target]=content
                    d_map[target]=min_distance
    for seg,content in  r_map.items():
        radius_anno.append(content)

    #缺省的自由边半径标注


    arc_free_edges=[]
    for s in s_free_edges:
        if isinstance(s.ref,DArc):
            arc_free_edges.append(s)

    pairs=[]
    for i,s1 in enumerate(arc_free_edges):
        for j,s2 in enumerate(arc_free_edges):
            if j<=i:
                continue
            pairs.append((s1,s2))
    for pair in pairs:
        s1,s2=pair
        if s1.ref is None or s1.ref.radius is None or s2.ref is None or s2.ref.radius is None:
            continue
        if abs(s1.ref.radius-s2.ref.radius)>20:
            continue
        if s1 in r_map and s2 in r_map:
            continue
        if s1 not in r_map and s2 not in r_map:
            continue
        if s1 in r_map and s2 not in r_map:
            r_map[s2]=r_map[s1]
        if s2 in r_map and s1 not in r_map:
            r_map[s1]=r_map[s2]
    # print(str(r_map))
    # print(r_anno)
    return r_map,radius_anno

def computeDistance(s,v1,v2):
    vs,ve=s.start_point,s.end_point
    d1,d2=DSegment(vs,v1).length(),DSegment(vs,v2).length()
    if d1<d2:
        distance=d1+DSegment(ve,v2).length()
    else:
        distance=d2+DSegment(ve,v1).length()
    return distance
def tidy_anno_output(edge_anno):
    s=""
    for t in edge_anno:
        v1,v2,l=t
        s+=f"起始点:{v1}、终止点:{v2}、标注长度:{l.text}、标注句柄:{l.handle}\n"
    return s[:-1]
def tidy_anno_output2(edge_anno):
    s=""
    for t in edge_anno:
        s+=f"标注长度:{t.text}、标注句柄:{t.handle}\n"
    return s[:-1]
def tidy_anno_output3(edge_anno):
    s=""
    for t in edge_anno:
        p1,p2,inter,d=t
        s+=f"起始点:{p1}、终止点:{p2}、圆心:{inter}、标注角度:{d.text}、标注句柄:{d.handle}\n"
    return s[:-1]
def are_equal_with_tolerance_(a, b, tolerance=1e-6):
    return abs(a - b) < tolerance

def is_parallel_(seg1, seg2, tolerance=0.05):
    """判断两条线段是否平行"""
    dx1 = seg1.end_point.x - seg1.start_point.x
    dy1 = seg1.end_point.y - seg1.start_point.y
    dx2 = seg2.end_point.x - seg2.start_point.x
    dy2 = seg2.end_point.y - seg2.start_point.y

    # 计算两个方向向量的模长
    length1 = math.sqrt(dx1**2 + dy1**2)
    length2 = math.sqrt(dx2**2 + dy2**2)
    
    # 防止长度为0的线段
    if length1 == 0 or length2 == 0:
        return False
    
    # 归一化叉积
    cross_product = (dx1 * dy2 - dy1 * dx2) / (length1 * length2)
    # print(seg1)
    # print(seg2)
    # print(dx1,dx2,dy1,dy2)
    # print(cross_product)
    # 返回是否接近0
    #print(cross_product)
    return are_equal_with_tolerance_(cross_product, 0, tolerance)

def is_vertical_(point1,point2,segment,epsilon=0.05):
    if segment is None:
        return False
    v1=DPoint(point1.x-point2.x,point1.y-point2.y)
    v2=DPoint(segment.start_point.x-segment.end_point.x,segment.start_point.y-segment.end_point.y)
    cross_product=(v1.x*v2.x+v1.y*v2.y)/(DSegment(point1,point2).length()*segment.length()+1e-4)
    if  abs(cross_product) <epsilon:
        return True
    return False 

def is_toe(free_edge,last_free_edge,cons_edge,max_free_edge_length):
    if last_free_edge is not None and isinstance(last_free_edge.ref,DArc) and is_tangent_(free_edge,last_free_edge):
        return False
    if (free_edge.length()<56 or free_edge.length()<=0.105*max_free_edge_length) and is_vertical_(free_edge.start_point,free_edge.end_point,cons_edge,epsilon=0.1):
        return True
    return False
def is_ks_corner(free_edge,last_free_edge,cons_edge,max_free_edge_length):
    if (not is_toe(free_edge,last_free_edge,cons_edge,max_free_edge_length)) and (not is_vertical_(free_edge.start_point,free_edge.end_point,cons_edge,epsilon=0.1)) and isinstance(last_free_edge.ref,DLine) and free_edge.length() <= 100:
        return True
    return False

def find_cons_edge(poly_refs,seg):
    for s in poly_refs:
        if (not s.isCornerhole) and (not s.isConstraint):
            continue
        if s.start_point==seg.start_point or s.start_point == seg.end_point or s.end_point ==seg.start_point or s.end_point==seg.end_point:
            return s
    return None
        
def match_l_anno(l_anno,poly_refs,constraint_edges,free_edges,segmentation_config):
    
    l_whole_map={}
    l_half_map={}
    l_cornor_map={}
    l_para_map={}
    l_para_single_map={}
    l_n_para_map={}
    l_n_para_single_map={}
    l_ver_map={}
    l_ver_single_map={}
    s_constraint_edges=[]
    ori_cons_edges=[]
    whole_anno=[]
    half_anno=[]
    cornor_anno=[]
    parallel_anno=[]
    non_parallel_anno=[]
    vertical_anno=[]

    d_map={}
    d_anno=[]
    for constraint_edge in constraint_edges:
        
        s_constraint_edge=shrinkFixedLength(constraint_edge,10)
        s_constraint_edges.extend(s_constraint_edge)
        ori_cons_edges.extend(constraint_edge)
    s_free_edges=free_edges[0]
    ss_free_edges=shrinkFixedLength(s_free_edges,6)
    for l_t in l_anno:
        v1,v2,l=l_t
        
        ty,idx=check_points_against_segments(v1,v2,s_constraint_edges)
        if ty is not None:
            edge=ori_cons_edges[idx]
            if ty=="whole":
                if check_whole(v1,v2,edge,s_constraint_edges,ori_cons_edges):
                    if edge not in l_whole_map:
                        l_whole_map[edge]=[]    
                    l_whole_map[edge].append((v1,v2,l))
                    whole_anno.append(l)
                    continue
            elif ty=="half":
                
                if edge not in l_half_map:
                    l_half_map[edge]=[]    
                l_half_map[edge].append((v1,v2,l))
                half_anno.append(l)
                continue
            elif ty=="cornor":
                
                if edge not in l_cornor_map:
                    l_cornor_map[edge]=[]    
                l_cornor_map[edge].append((v1,v2,l))
                cornor_anno.append(l)
                continue
            
        ty,idx=check_points_against_free_segments(v1,v2,ss_free_edges)
        
        if ty is not None:
            edge=s_free_edges[idx]

            if edge not in d_map:
                d_map[edge]=[]
            d_map[edge].append((v1,v2,l))
            d_anno.append(l)
            continue
        key=check_parallel_anno(v1,v2,ori_cons_edges,s_free_edges)
        if key is not None:
            constraint_edge,free_edge=key
            if  is_parallel_(constraint_edge,free_edge,0.1):
                
                if key not in l_para_map:
                    l_para_map[key]=[]
                l_para_map[key].append(l)
                if constraint_edge not in l_para_single_map:
                    l_para_single_map[constraint_edge]=[]
                if free_edge not in l_para_single_map:
                    l_para_single_map[free_edge]=[]
                l_para_single_map[constraint_edge].append(l)
                l_para_single_map[free_edge].append(l)
                parallel_anno.append(l)
                continue 
        key=check_non_parallel_anno(v1,v2,ori_cons_edges,s_free_edges)
        if key is not None:
            constraint_edge,free_edge=key
            if key not in l_n_para_map:
                l_n_para_map[key]=[]
            l_n_para_map[key].append(l)
            if constraint_edge not in l_n_para_single_map:
                l_n_para_single_map[constraint_edge]=[]
            if free_edge not in l_n_para_single_map:
                l_n_para_single_map[free_edge]=[]
            l_n_para_single_map[constraint_edge].append(l)
            l_n_para_single_map[free_edge].append(l)
            non_parallel_anno.append(l)
            continue 
        key=check_vertical_anno(v1,v2,ori_cons_edges)
        if key is not None:
            start_edge,end_edge=key
            if key not in l_ver_map:
                l_ver_map[key]=[]
            l_ver_map[key].append(l)
            if start_edge not in l_ver_single_map:
                l_ver_single_map[start_edge]=[]
            if end_edge not in l_ver_single_map:
                l_ver_single_map[end_edge]=[]
            l_ver_single_map[start_edge].append(l)
            l_ver_single_map[end_edge].append(l)
            vertical_anno.append(l)
            continue 

    return l_whole_map,l_half_map,l_cornor_map,l_para_map,l_para_single_map,l_n_para_map,l_n_para_single_map,l_ver_map,l_ver_single_map,d_map,whole_anno,half_anno,cornor_anno,parallel_anno,non_parallel_anno,vertical_anno,d_anno

def match_a_anno(a_anno,free_edges,constraint_edges):
    s_free_edges=free_edges[0]
    a_map={}
    angle_anno=[]
    toe_angle_anno=[]
    max_free_edge_length=float("-inf")
    for seg in free_edges[0]:
        max_free_edge_length=max(max_free_edge_length,seg.length())
    cons_edges=[]
    for constraint_edge in constraint_edges:
        for edge in constraint_edge:
            cons_edges.append(edge)  
    for a in a_anno:
        p1,p2,inter,d=a
        l1,l2=DSegment(inter,p1),DSegment(inter,p2)
        edges=[]
        for edge in s_free_edges:
            if is_parallel(edge,l1,0.15) or is_parallel(edge,l2,0.15):
                edges.append(edge)
        target=None
        l=float('inf')
        for edge in edges:
            if isinstance(edge.ref,DArc):
                continue
            ll=edge.length()
           
            if ll<l:
                l=ll
                target=edge
        if target is not None:
            if target not in a_map:
                a_map[target]=[]
            a_map[target].append((p1,p2,inter,d))
            cons_edge=find_cons_edge(cons_edges,target)
            if target==s_free_edges[0] and len(s_free_edges)>1:
                last_free_edge=s_free_edges[1]
                if is_toe(target,last_free_edge,cons_edge,max_free_edge_length):
                    toe_angle_anno.append(d)
                else:
                    angle_anno.append(d)
            elif  target==s_free_edges[-1] and len(s_free_edges)>1:
                last_free_edge=s_free_edges[-2]
                if is_toe(target,last_free_edge,cons_edge,max_free_edge_length):
                    toe_angle_anno.append(d)
                else:
                    angle_anno.append(d)
            else:
                angle_anno.append(d)




    return a_map,angle_anno,toe_angle_anno

def affine(d1,d2,d3,d4,ori_cons_edges):
    p1,p2,p3,p4=d1,d2,d3,d4
    for s in ori_cons_edges:
        for p in [s.start_point,s.end_point]:
            if DSegment(p,p3).length()<5:
                p3=p
            if DSegment(p,p4).length()<5:
                p4=p
    return p1,p2,p3,p4
def compute_accurate_position(d1,d2,d3,d4,constraint_edges):
    ori_cons_edges=[]
    for constraint_edge in constraint_edges:
        
        # s_constraint_edge=shrinkFixedLength(constraint_edge,10)
        # s_constraint_edges.extend(s_constraint_edge)
        ori_cons_edges.extend(constraint_edge)
    ref_segment=DSegment(d1,d2)
    for i,segment in enumerate(ori_cons_edges):

        if is_parallel(ref_segment,segment,0.15):
            pos1 = point_segment_position(d3, segment)
            pos2 = point_segment_position(d4, segment)
            if pos1 !="not_on_line" and pos2 !="not_on_line":
                return affine(d1,d2,d3,d4,ori_cons_edges)
            if pos1!= "not_on_line":
                return affine(d1,d2,d3,p_minus(p_add(d1,d3),d2),ori_cons_edges)
            if pos2 !="not_on_line":
                return affine(d1,d2,p_minus(p_add(d2,d4),d1),d4,ori_cons_edges)
    l1,l2=DSegment(d1,d4).length(),DSegment(d2,d3).length()
    if l1<l2:
        return affine(d1,d2,d3,p_minus(p_add(d1,d3),d2),ori_cons_edges)
    else:
        return affine(d1,d2,p_minus(p_add(d2,d4),d1),d4,ori_cons_edges)


def compara_free_order(free_edges,free_edges_seq):
    detect_free_edges_seq=[]
    for edge in free_edges:
        if isinstance(edge.ref,DArc):
            detect_free_edges_seq.append("arc")
        else:
            detect_free_edges_seq.append("line")
    for i in range(len(free_edges_seq)):
        if detect_free_edges_seq[i]!=free_edges_seq[i]:
            return False
    return True


def get_anno_position(d,constraint_edges):
    l0=p_minus(d.defpoints[0],d.defpoints[2])
    l1=p_minus(d.defpoints[1],d.defpoints[2])
    d10=l0.x*l1.x+l0.y*l1.y
    d00=l0.x*l0.x+l0.y*l0.y
    if d00 <1e-4:
        x=d.defpoints[1]
    else:
        x=p_minus(p_add(d.defpoints[1],l0),p_mul(l0,d10/d00))
    d1,d2,d3,d4=d.defpoints[0], x,d.defpoints[1],d.defpoints[2]
    d1,d2,d3,d4=compute_accurate_position(d1,d2,d3,d4,constraint_edges)
    return d3,d4
def is_tangent_(line,arc):
    s1,s2=DSegment(arc.ref.center,arc.ref.start_point),DSegment(arc.ref.center,arc.ref.end_point)
    v1,v2=line.start_point,line.end_point
    if v1==arc.ref.start_point or v2==arc.ref.start_point:
        return is_vertical_(v1,v2,s1,0.1)
    if v1==arc.ref.end_point or v2==arc.ref.end_point:
        return is_vertical_(v1,v2,s2,0.1)
    return True
def is_vu(edge):
    if len(edge)!=2:
        return False
    if (isinstance(edge[0].ref,DLine) and isinstance(edge[1].ref,DArc) ) or (isinstance(edge[1].ref,DLine) and isinstance(edge[0].ref,DArc) ) :
        return True
    return False
def is_near(cons_edge,short_edge):
    d1=DSegment(cons_edge.start_point,short_edge.start_point).length()
    d3=DSegment(cons_edge.end_point,short_edge.start_point).length()
    d2=DSegment(cons_edge.start_point,short_edge.end_point).length()
    d4=DSegment(cons_edge.end_point,short_edge.end_point).length()

    if (d1<100 and d2<100) or (d3<100 and d4<100):
        return True
    return False

def is_near_vu(cons_edge,short_edge):
    d1=DSegment(cons_edge.start_point,short_edge.start_point).length()
    d3=DSegment(cons_edge.end_point,short_edge.start_point).length()
    d2=DSegment(cons_edge.start_point,short_edge.end_point).length()
    d4=DSegment(cons_edge.end_point,short_edge.end_point).length()

    min_dis=min(d1,d2,d3,d4)
    max_dis=max(d1,d2,d3,d4)
    if min_dis<25 and max_dis<250:
        return True
    return False
def find_reference_edge(seg,fr_edges,cons_edges,p1,p2,inter):




    para_edges=[]
    l1,l2=DSegment(inter,p1),DSegment(inter,p2)
    edge=None
    if is_parallel_(seg,l1,0.15):
        edge=l2
    else:
        edge=l1
    for cons_edge in cons_edges:
        if is_parallel_(edge,cons_edge,0.15):
            para_edges.append(cons_edge)
    distance=float("inf")
    ref_edge=None
    for s in para_edges:
        dis=DSegment(s.mid_point(),inter).length()
        if dis<distance:
            distance=dis
            ref_edge=s
    if ref_edge is not None:
        return True,ref_edge
    return False,edge

def match_edge_anno(segments,constraint_edges,free_edges,edges,all_anno,all_map):
    features=set()
    feature_map={}
    corner_holes=[]
    corner_hole_start_edge=[]
    constraint_edge_no={}
    corner_hole_arc={}
    k=1
    for i,edge in enumerate(edges):
        if (not edge[0].isConstraint) and (not edge[0].isCornerhole):
            continue
        if edge[0].isCornerhole:
            
            corner_holes.append(edge)
            corner_hole_start_edge.append(edge[0])
            for s in edge:
                if isinstance(s.ref,DArc):
                    corner_hole_arc[s]=edge[0]
                feature_map[s]=set()
        else:
            for s in  edge:
                constraint_edge_no[s]=k
                feature_map[s]=set()
            k+=1
    cons_edges=[]
    fr_edges=free_edges[0]
    free_edge_no={}
    for i,s in enumerate(fr_edges):
        feature_map[s]=set()
        free_edge_no[s]=i+1
    for constraint_edge in constraint_edges:
        for edge in constraint_edge:
            cons_edges.append(edge)




    max_free_edge_length=float("-inf")
    for seg in free_edges[0]:
        max_free_edge_length=max(max_free_edge_length,seg.length())
    c_anno,radius_anno,whole_anno,half_anno,cornor_anno,parallel_anno,non_parallel_anno,vertical_anno,d_anno,angle_anno,toe_angle_anno=all_anno
    r_map,l_whole_map,l_half_map,l_cornor_map,l_para_map,l_para_single_map,l_n_para_map,l_n_para_single_map,l_ver_map,l_ver_single_map,d_map,a_map=all_map
    edge_type={}
    for i, seg in enumerate(free_edges[0]):
        if isinstance(seg.ref, DLine) or isinstance(seg.ref, DLwpolyline):
            if (i == 0 or i == len(free_edges[0]) - 1):
                if i==0:
                    last_free_edge=free_edges[0][1]
                else:
                    last_free_edge=free_edges[0][-2]
                cons_edge=find_cons_edge(cons_edges,seg)
                # print(cons_edge)
                if is_toe(seg,last_free_edge,cons_edge,max_free_edge_length):
                    edge_type[seg]="toe"
                elif is_ks_corner(seg,last_free_edge,cons_edge,max_free_edge_length):
                    edge_type[seg]="KS_corner"
                else:
                    edge_type[seg]="line"
            else:
                edge_type[seg]="line"
        elif isinstance(seg.ref, DArc):
            edge_type[seg]="arc"

    



    all_toe=[]
    for free_edge,ty in edge_type.items():
        if ty=="toe":
            all_toe.append(free_edge)
    all_edge_map={}

    for  edge in cons_edges:
        all_edge_map[edge]={}
        all_edge_map[edge]["边长标注"]=[]
        all_edge_map[edge]["垂直标注"]=[]

    for seg, ds in l_whole_map.items():
        for d_t in ds:
            p1,p2,d=d_t
            all_edge_map[seg]["边长标注"].append([d,f"句柄：{d.handle}，值：{d.text}，箭头起点：{p1}，箭头终点：{p2}"])
    k=0
    for key, ds in l_ver_map.items():
        start_edge,base_edge=key
        for d in ds:
            k+=1
            v1,v2 = get_anno_position(d, constraint_edges)
            all_edge_map[start_edge]["垂直标注"].append([d,f"句柄：{d.handle}，值：{d.text}，箭头起点：{v1}，箭头终点：{v2}，垂边，参考边：约束边",base_edge])
            all_edge_map[base_edge]["垂直标注"].append([d,f"句柄：{d.handle}，值：{d.text}，箭头起点：{v1}，箭头终点：{v2}，底边，参考边：约束边",start_edge])
    # if k==0:
    #     features.add('ff')
    # elif k==1:
    #     features.add('tf')
    # else:
    #     features.add('tt')
    # cons_maps=(l_whole_map,l_ver_map,l_ver_single_map)
    # cons_annos=(whole_anno,vertical_anno)
    #cons_edges
    #l_whole_map   seg->[dimesion]
    #l_ver_map     (start_seg,base——seg)->[dimesion]
    #l_ver_single_map   seg->[dimesion]


    #seg -->  {   type --> [dimension,dimension_des]s }

    for edge in fr_edges:
        all_edge_map[edge]={}
        if edge_type[edge]=="line":
            all_edge_map[edge]["边长标注"]=[]
            all_edge_map[edge]["延长线与约束边交点标注"]=[]
            all_edge_map[edge]["与趾端夹角标注"]=[]
            all_edge_map[edge]["与约束边及其平行线夹角标注"]=[]
            all_edge_map[edge]["与约束边垂直标注"]=[]
            all_edge_map[edge]["是否与约束边平行"]=False
            all_edge_map[edge]["是否与相邻约束边夹角为90度"]=False
            all_edge_map[edge]["平行标注"]=[]
            all_edge_map[edge]["平行于"]=[]
            all_edge_map[edge]["垂直于相邻边"]=[]
        elif edge_type[edge]=="arc":
            all_edge_map[edge]["是否相切"]=False
            all_edge_map[edge]["圆心是否在趾端延长线上"]=False
            all_edge_map[edge]["半径尺寸标注"]=[]
        elif edge_type[edge]=="toe":
            all_edge_map[edge]["边长标注"]=[]
            all_edge_map[edge]["存在正面或背面结构约束"]=[]
            all_edge_map[edge]["Gap"]=[]
            all_edge_map[edge]["Gap尺寸标注"]=[]
        elif edge_type[edge]=="KS_corner":
            all_edge_map[edge]["与自由边夹角标注"]=[]
            all_edge_map[edge]["与自由边长度标注"]=[]
    
    for corner_hole in corner_holes:
        for edge in corner_hole:
            all_edge_map[edge]={}
            all_edge_map[edge]["短边尺寸标注"]=[]
            all_edge_map[edge]["半径尺寸标注"]=[]
            all_edge_map[edge]["短边是否平行于相邻边"]=False
            all_edge_map[edge]["尺寸参数"]=[]
    

    toes =[]
    toe_annos=[]
    toe_points={}
    for edge in fr_edges:
        if edge_type[edge]=="toe":
            toes.append(edge)
            toe_points[edge.start_point]=edge
            toe_points[edge.end_point]=edge

            cons_edge=find_cons_edge(cons_edges,edge)
            if cons_edge is None:
                all_edge_map[edge]["存在正面或背面结构约束"].append([DDimension(),"无"])
                toe_annos.append("无")
                continue
            p=None
            if cons_edge.start_point==edge.start_point or cons_edge.end_point==edge.start_point:
                p=edge.start_point
            else:
                p=edge.end_point
            q=None
            if p==cons_edge.start_point:
                q=cons_edge.end_point
            else:
                q=cons_edge.start_point
            anno_="无"

            v=DPoint(edge.end_point.x-edge.start_point.x,edge.end_point.y-edge.start_point.y)
            l=math.sqrt(v.x*v.x+v.y*v.y)
            v=DPoint(-v.y/l*50,v.x/l*50)
            m=edge.mid_point()
            seg=DSegment(DPoint(m.x+v.x,m.y+v.y),DPoint(m.x-v.x,m.y-v.y))
            for s in segments:
                inter=segment_intersection(seg.start_point, seg.end_point, s.start_point, s.end_point)
                if inter is not None:
                    distance=DSegment(inter,m).length()
                    if distance>=10 and distance<=50:
                        anno_="正面"
                        break
            if anno_=="无":
                for s in segments:
                    if point_segment_position(s.start_point,cons_edge,epsilon=0.05,anno=False)!="not_on_line":
                        if DSegment(s.start_point,p).length()<=50 and DSegment(s.start_point,p).length()>=10:
                            anno_="背面"
                            break
                    elif point_segment_position(s.end_point,cons_edge,epsilon=0.05,anno=False)!="not_on_line":
                        if DSegment(s.end_point,p).length()<=50 and DSegment(s.end_point,p).length()>=10:
                            anno_="背面"
                            break
            toe_annos.append(anno_)
            all_edge_map[edge]["存在正面或背面结构约束"].append([DDimension(),anno_])


    #趾端周围结构    
    #自由边直线以及趾端边长标注
    for seg,ds in d_map.items():
        if seg not in edge_type:
            continue
        for d_t in ds:
            v1,v2,d =d_t
            if edge_type[seg]=="line":
                features.add('D')
                all_edge_map[seg]["边长标注"].append([d,f"句柄：{d.handle}，值：{d.text}，箭头起点：{v1}，箭头终点：{v2}"])
            elif edge_type[seg]=="toe":
                features.add('D')
                all_edge_map[seg]["边长标注"].append([d,f"句柄：{d.handle}，值：{d.text}，箭头起点：{v1}，箭头终点：{v2}"])
  

    #自由边直线延长线与约束边交点标注/背部结构gap参数
    for seg,ds in l_half_map.items():
        for d_t in ds:
            v1,v2,d=d_t
            for idx,free_edge in enumerate(fr_edges):
                if edge_type[free_edge]!="line":
                    continue
                if isinstance(free_edge.ref,DArc):
                    continue
                if point_segment_position(v1,free_edge,epsilon=0.25,anno=False)!="not_on_line" or point_segment_position(v2,free_edge,epsilon=0.25,anno=False)!="not_on_line":
                    if point_segment_position(v1,free_edge,epsilon=0.25,anno=False)!="not_on_line":
                        v=v2
                    else:
                        v=v1
                    if DSegment(v,seg.start_point).length() < DSegment(v,seg.end_point).length():
                        point1,point2=seg.start_point,seg.end_point
                    else:
                        point1,point2=seg.end_point,seg.start_point

                    
                    all_edge_map[free_edge]["延长线与约束边交点标注"].append([d,f"句柄：{d.handle}，值：{d.text}，箭头起点：{v1}，箭头终点：{v2}，参考边：约束边",seg,(seg,point1,point2)])
                    features.add('short_anno')
                    feature_map[free_edge].add('short_anno')
                    break
                else:
                    if v1 in toe_points:
                        all_edge_map[toe_points[v1]]["Gap尺寸标注"].append([d,f"句柄：{d.handle}，值：{d.text}，箭头起点：{v1}，箭头终点：{v2}"])
                        all_edge_map[toe_points[v1]]["Gap"].append([DDimension(),d.text.strip()])
                    elif v2 in toe_points:
                        all_edge_map[toe_points[v2]]["Gap尺寸标注"].append([d,f"句柄：{d.handle}，值：{d.text}，箭头起点：{v1}，箭头终点：{v2}"])
                        all_edge_map[toe_points[v2]]["Gap"].append([DDimension(),d.text.strip()])
            # point_segment_position(p1,,epsilon=0.2,anno=False)!="not_on_line"
    

    #自由边直线与趾端夹角标注/自由边倒角与自由边夹角标注/自由边直线与约束边或平行线夹角标注
    for seg,ds in a_map.items():
        for d_t in ds:
            p1,p2,inter,d=d_t
            if edge_type[seg]=="toe":
                idx=free_edge_no[seg]
                if idx==1:
                    for s in fr_edges:
                        if edge_type[s]=="line":
                            #句柄：FF280， 值：14.5，参考边：2，箭头起点：Point(1126156.00605, -283084.451961)，箭头终点：Point(1126156.00605, -283084.451961
                            all_edge_map[s]["与趾端夹角标注"].append([d,f"句柄：{d.handle}，值：{d.text}，参考点1：{p1}，参考点2：{p2}，参考中心： {inter}，参考边： (趾端)自由边",seg])
                            features.add('toe_angle')
                            feature_map[s].add('toe_angle')
                            break
                if idx==len(fr_edges):
                    for s in fr_edges[::-1]:
                        if edge_type[s]=="line":
                            #句柄：FF280， 值：14.5，参考边：2，箭头起点：Point(1126156.00605, -283084.451961)，箭头终点：Point(1126156.00605, -283084.451961
                            all_edge_map[s]["与趾端夹角标注"].append([d,f"句柄：{d.handle}，值：{d.text}，参考点1：{p1}，参考点2：{p2}，参考中心： {inter}，参考边： (趾端)自由边",seg])
                            features.add('toe_angle')
                            feature_map[s].add('toe_angle')
                            break
            elif edge_type[seg]=="line":

                flag,ref_edge=find_reference_edge(seg,fr_edges,cons_edges,p1,p2,inter)
                features.add('angl')
                if flag:
                    all_edge_map[seg]["与约束边及其平行线夹角标注"].append([d,f"句柄：{d.handle}，值：{d.text}，参考点1：{p1}，参考点2：{p2}，参考中心： {inter}，参考边：约束边",ref_edge])
                else:
                    all_edge_map[seg]["与约束边及其平行线夹角标注"].append([d,f"句柄：{d.handle}，值：{d.text}，参考点1：{p1}，参考点2：{p2}，参考中心： {inter}，参考边：{ref_edge}"])
                feature_map[seg].add('angl')
            elif edge_type[seg]=="KS_corner":

                flag,ref_edge=find_reference_edge(seg,fr_edges,cons_edges,p1,p2,inter)
                features.add('angl')
                feature_map[seg].add('angl')
                if flag:
                    all_edge_map[seg]["与自由边夹角标注"].append([d,f"句柄：{d.handle}，值：{d.text}，参考点1：{p1}，参考点2：{p2}，参考中心： {inter}，参考边：约束边",ref_edge])
                else:
                    all_edge_map[seg]["与自由边夹角标注"].append([d,f"句柄：{d.handle}，值：{d.text}，参考点1：{p1}，参考点2：{p2}，参考中心： {inter}，参考边：{ref_edge}"])
    #自由边直线与约束边垂直标注/自由边倒角与约束边距离标注
    for key,ds in l_n_para_map.items():
        cons_edge,free_edge=key
        for d in ds:
            v1,v2=get_anno_position(d,constraint_edges)
            if edge_type[free_edge]=="line":
                features.add('cons_vert')
                feature_map[free_edge].add('cons_vert')
                all_edge_map[free_edge]["与约束边垂直标注"].append([d,f"句柄：{d.handle}，值：{d.text}，箭头起点：{v1}，箭头终点：{v2}，参考边：约束边",cons_edge])
            elif edge_type[free_edge]=="KS_corner":
                features.add('cons_vert')
                feature_map[free_edge].add('cons_vert')
                all_edge_map[free_edge]["与自由边长度标注"].append([d,f"句柄：{d.handle}，值：{d.text}，箭头起点：{v1}，箭头终点：{v2}，参考边：约束边",cons_edge])
    #自由边直线几何特征（平行与垂直）/自由边圆弧几何特征
    for seg in fr_edges:
        if edge_type[seg]=="line":
            flag=False
            para_edge=None
            for cons_edge in cons_edges:
                if flag:
                    break
                flag=is_parallel_(seg,cons_edge,0.1)
                if flag:
                    para_edge=cons_edge
      
            all_edge_map[seg]["是否与约束边平行"]=flag
            if flag:
            
                all_edge_map[seg]["平行于"].append([DDimension(),f"约束边",para_edge])
                features.add('is_para')
                feature_map[seg].add('is_para')
            if free_edge_no[seg]==1 or free_edge_no[seg]==len(fr_edges):
                cons_edge=find_cons_edge(cons_edges,seg)
                flag=is_vertical_(seg.start_point,seg.end_point,cons_edge,0.005)
                all_edge_map[seg]["是否与相邻约束边夹角为90度"]=flag
                if flag:
                    all_edge_map[seg]["垂直于相邻边"].append([DDimension(),f"约束边",cons_edge])
                    features.add('is_ver')
                    feature_map[seg].add('is_ver')
            else:
                all_edge_map[seg]["是否与相邻约束边夹角为90度"]=False
        elif edge_type[seg]=="arc":
            v1,v2=seg.ref.start_point,seg.ref.end_point
            o=seg.ref.center
            idx=free_edge_no[seg]
            free_edges_nearby=[]
            if idx==1:
                free_edges_nearby.append(fr_edges[idx])
            elif idx==len(fr_edges):
                free_edges_nearby.append(fr_edges[-2])
            else:
                idx=idx-1
                free_edges_nearby.append(fr_edges[idx-1])
                free_edges_nearby.append(fr_edges[idx+1])
            flag=True
            for free_edge_nearby in free_edges_nearby:
                if is_tangent_(free_edge_nearby,seg)==False:
                    flag=False
                if flag==False:
                    break
            all_edge_map[seg]["是否相切"]=flag
            if flag==False:
                features.add('no_tangent')
                feature_map[seg].add('no_tangent')
            flag=False
            for toe_line in all_toe:
                if point_segment_position(o,toe_line,epsilon=0.2,anno=False)!="not_on_line":
                    flag=True
                if flag:
                    break
            all_edge_map[seg]["圆心是否在趾端延长线上"]=flag
            if flag:
                features.add('is_ontoe')
                feature_map[seg].add('is_ontoe')

    #vu角隅孔几何特征
    for i,seg in enumerate(corner_hole_start_edge):
        edge=corner_holes[i]
        if is_vu(edge):
            short_edge=edge[0] if isinstance(edge[0].ref,DLine) else edge[1] 
            flag=False
            for cons_edge in cons_edges:
                if is_near(cons_edge,short_edge) and is_parallel_(short_edge,cons_edge,0.1):
                    flag=True
                    break
            all_edge_map[edge[0]]["短边是否平行于相邻边"]=flag
            all_edge_map[edge[1]]["短边是否平行于相邻边"]=flag
    #vu角隅孔短边尺寸标注/gap正面
    for seg,ds in l_cornor_map.items():
        for d_t in ds:
            v1,v2,d=d_t
            flag=False
            for i,start_edge in enumerate(corner_hole_start_edge):
                edge=corner_holes[i]
                if is_vu(edge):
                    short_edge=edge[0] if isinstance(edge[0].ref,DLine) else edge[1] 
                    if is_near_vu(DSegment(v1,v2),short_edge):
                        all_edge_map[edge[0]]["短边尺寸标注"].append([d,f"句柄：{d.handle}，值：{d.text}，箭头起点：{v1}，箭头终点：{v2}"])
                        all_edge_map[edge[1]]["短边尺寸标注"].append([d,f"句柄：{d.handle}，值：{d.text}，箭头起点：{v1}，箭头终点：{v2}"])
                        # features.add('vuf')
                        flag=True
                        break
            if flag==False:
                if v1 in toe_points:
                    all_edge_map[toe_points[v1]]["Gap尺寸标注"].append([d,f"句柄：{d.handle}，值：{d.text}，箭头起点：{v1}，箭头终点：{v2}"])
                    all_edge_map[toe_points[v1]]["Gap"].append([DDimension(),d.text.strip()])
                elif v2 in toe_points:
                    all_edge_map[toe_points[v2]]["Gap尺寸标注"].append([d,f"句柄：{d.handle}，值：{d.text}，箭头起点：{v1}，箭头终点：{v2}"])
                    all_edge_map[toe_points[v2]]["Gap"].append([DDimension(),d.text.strip()])

    #自由边直线平行标注
    for key,ds in l_para_map.items():
        cons_edge,free_edge=key
        for d in ds:
            v1,v2=get_anno_position(d,constraint_edges)
            if edge_type[free_edge]=="line":
                features.add('short_anno_para')
                feature_map[free_edge].add('short_anno_para')
                all_edge_map[free_edge]["平行标注"].append([d,f"句柄：{d.handle}，值：{d.text}，箭头起点：{v1}，箭头终点：{v2}，参考边：约束边",cons_edge])
            elif edge_type[free_edge]=="toe":
                cons_edge=find_cons_edge(cons_edges,free_edge)
                all_edge_map[cons_edge]["边长标注"].append([d,f"句柄：{d.handle}，值：{d.text}，箭头起点：{v1}，箭头终点：{v2}"])

    #半径尺寸标注
    for seg,t in r_map.items():

        if seg in edge_type and edge_type[seg]=="arc":
            all_edge_map[seg]["半径尺寸标注"].append([t,f"句柄：{t.handle}，值：{t.content}"])
        elif seg in corner_hole_arc:
            all_edge_map[seg]["半径尺寸标注"].append([t,f"句柄：{t.handle}，值：{t.content}"])
    

    #角隅孔尺寸标注
    for c_t in c_anno:
        p,t =c_t
        distance=float('inf')
        idx=None
        for i,seg in enumerate(corner_hole_start_edge):
            q=seg.mid_point()
            dis=DSegment(p,q).length()
            if dis<500 and dis<distance:
                distance=dis
                idx=i
        if idx is not None:
            corner_hole =corner_holes[idx]
            ct=[]
            for edge in corner_hole:
                if isinstance(edge.ref,DLine):
                    ct.append("line")
                elif isinstance(edge.ref,DArc):
                    ct.append("arc")
            if t.content.strip()[0]=="R" and ct !=["arc"]:
                continue
            if (len(t.content.strip().split("X"))==2 or len(t.content.strip().split("x"))==2) and (not(ct==["arc","line"] or ct==["line","arc"])):
                continue

            for edge in corner_hole:
                all_edge_map[edge]["尺寸参数"].append([t,f"句柄：{t.handle}，值：{t.content}"])
    for s,fs in feature_map.items():
        feature_map[s]=list(fs)
    #补充gap值
    for edge in toes:
        if len(all_edge_map[edge]["Gap"])>1:
            all_edge_map[edge]["Gap"]=[all_edge_map[edge]["Gap"][0]]
        elif len(all_edge_map[edge]["Gap"])==0:
            if all_edge_map[edge]["存在正面或背面结构约束"]=="无":
                all_edge_map[edge]["Gap"].append([DDimension(),"无"])
            elif all_edge_map[edge]["存在正面或背面结构约束"]=="正面":
                all_edge_map[edge]["Gap"].append([DDimension(),"35"])
            else:
                all_edge_map[edge]["Gap"].append([DDimension(),"15"])
    return all_edge_map,edge_type,list(features),feature_map,constraint_edge_no,free_edge_no

def  get_edge_des(edge):
    des=""
    for s in edge:
        if isinstance(s.ref,DArc):
            des+="arc,"
        else:
            des+="line,"
    return des[:-1]
def get_free_edge_des(edge,edge_types):
    des=""
    for s in edge:
       
        des+=edge_types[s]+","
    return des[:-1]

def calculate_poly_features(poly, segments, segmentation_config, point_map, index,star_pos_map,cornor_holes,texts,dimensions,text_map,stiffeners, hatch_polys,hole_text_coors, jg_s):
    # step1: 计算几何中心坐标
    poly_centroid = calculate_poly_centroid(poly)
    # 特判：判断几何是否在阴影中
    if is_intersect_hatch(poly, hatch_polys):
        print(f"回路{index}位于阴影区域")
        return None
    
    # 特判：判断几何中是否包含开孔
    if is_intersect_hole(poly, hole_text_coors):
        print(f"回路{index}中包含开孔")
        return None
    poly_map={}
    polygon=computePolygon(poly)
    # step2: 合并边界线
    poly_refs,ref_map= calculate_poly_refs(poly,segmentation_config)
    
    new_poly_ref=[]
    for ref in poly_refs:
        new_poly_ref.append(DSegment(ref.start_point,ref.end_point,ref.ref))
    poly_refs=new_poly_ref
    for ref in poly_refs:
        ref.initialize()

    # plot_polys({},poly_refs,"./check")
    # step3: 标记角隅孔
    # for corner_hole in cornor_holes:
    #     for seg in corner_hole.segments:
    #         seg.isCornerHole=True
    # print(len(cornor_holes))
    # for cornor_hole in cornor_holes:
    #     print(cornor_hole.segments)
    cornor_holes_map={}
    for cornor_hole in cornor_holes:
        for s in cornor_hole.segments:
            cornor_holes_map[s]=cornor_hole.ID
            cornor_holes_map[DSegment(s.end_point,s.start_point,s.ref)]=cornor_hole.ID
    for seg in poly_refs:
        if seg in cornor_holes_map:
            seg.isCornerhole=True
            #print(seg)
            

    
    #  根据星形角隅孔的位置，将角隅孔的坐标标记到相邻segment的StarCornerhole属性中，同时将该边标记为固定边
    # star_set=set()
    # for s in poly:
    #     if s in star_pos_map:
    #         for ss in star_pos_map[s]:
    #             star_set.add(ss)
    # stars_pos=list(star_set)
    
    s_stars_set=set()
    m_starts_set=set()
    s_stars_r_map={}
    m_stars_r_map={}

    for pos,texts in text_map.items():
        point=Point(pos.x,pos.y)
        if not polygon.contains(point):
            continue
        for t in texts:
            if t[1]["Type"]=="s_star":
                s_stars_set.add(pos)
                if pos not in s_stars_r_map:
                    s_stars_r_map[pos]=[]
                s_stars_r_map[pos].append(t[0])
            elif t[1]["Type"]=="m_star":
                m_starts_set.add(pos)
                if pos not in m_stars_r_map:
                    m_stars_r_map[pos]=[]
                m_stars_r_map[pos].append(t[0])
    star_pos=list(s_stars_set)
    for p in star_pos:
        x,y=p.x,p.y
        lines=[]
        lines.append(DSegment(DPoint(x,y),DPoint(x-5000,y)))
        lines.append(DSegment(DPoint(x,y),DPoint(x+5000,y)))
        lines.append(DSegment(DPoint(x,y),DPoint(x,y+5000)))
        lines.append(DSegment(DPoint(x,y),DPoint(x,y-5000)))
        cornor=[]
        for i, seg1 in enumerate(lines):
            dist=None
            s=None
            for j, seg2 in enumerate(poly_refs):
                p1, p2 = seg1.start_point, seg1.end_point
                q1, q2 = seg2.start_point, seg2.end_point
                intersection = segment_intersection(p1, p2, q1, q2)
                if intersection:
                    if dist is None:
                        dist=(intersection[0]-p1.x)*(intersection[0]-p1.x)+(intersection[1]-p1.y)*(intersection[1]-p1.y)
                        s=seg2
                    else:
                        if dist>(intersection[0]-p1.x)*(intersection[0]-p1.x)+(intersection[1]-p1.y)*(intersection[1]-p1.y):
                            dist=(intersection[0]-p1.x)*(intersection[0]-p1.x)+(intersection[1]-p1.y)*(intersection[1]-p1.y)
                            s=seg2
            if s is not None and s is not None and dist is not None and isinstance(s.ref,DLine) and dist<1000000 and s.length()>50:
               cornor.append((dist,s,p))
        cornor=sorted(cornor,key=lambda p:p[0])
        if len(cornor)>=2:
            cornor[0][1].isConstraint=True
            cornor[0][1].isCornerhole=False
            # cornor[0][1].StarCornerhole=cornor[0][2]
            cornor[1][1].isConstraint=True
            cornor[1][1].isCornerhole=False
            # cornor[1][1].StarCornerhole=cornor[1][2]
        elif len(cornor)==1:
            cornor[0][1].isConstraint=True
            cornor[0][1].isCornerhole=False 
    
    star_pos=list(m_starts_set)
    for p in star_pos:
        x,y=p.x,p.y
        lines=[]
        if p in m_stars_r_map and len(m_stars_r_map[p])>0:
            r=m_stars_r_map[p][0].rotation
        else:
            r=0
        vx,vy=math.cos(r/180*math.pi)*5000,math.sin(r/180*math.pi)*5000
        lines.append(DSegment(DPoint(x,y),DPoint(x+vx,y+vy)))
        lines.append(DSegment(DPoint(x,y),DPoint(x-vx,y-vy)))
        # lines.append(DSegment(DPoint(x,y),DPoint(x,y+5000)))
        # lines.append(DSegment(DPoint(x,y),DPoint(x,y-5000)))
        cornor=[]
        for i, seg1 in enumerate(lines):
            dist=None
            s=None
            for j, seg2 in enumerate(poly_refs):
                p1, p2 = seg1.start_point, seg1.end_point
                q1, q2 = seg2.start_point, seg2.end_point
                intersection = segment_intersection(p1, p2, q1, q2)
                if intersection:
                    if dist is None:
                        dist=(intersection[0]-p1.x)*(intersection[0]-p1.x)+(intersection[1]-p1.y)*(intersection[1]-p1.y)
                        s=seg2
                    else:
                        if dist>(intersection[0]-p1.x)*(intersection[0]-p1.x)+(intersection[1]-p1.y)*(intersection[1]-p1.y):
                            dist=(intersection[0]-p1.x)*(intersection[0]-p1.x)+(intersection[1]-p1.y)*(intersection[1]-p1.y)
                            s=seg2
            if s is not None and s is not None and dist is not None and isinstance(s.ref,DLine) and dist<1000000 and s.length()>50:
               cornor.append((dist,s,p))
        cornor=sorted(cornor,key=lambda p:p[0])
        if len(cornor)==2:
            cornor[0][1].isConstraint=True
            cornor[0][1].isCornerhole=False
            # cornor[0][1].StarCornerhole=cornor[0][2]
            cornor[1][1].isConstraint=True
            cornor[1][1].isCornerhole=False
            # cornor[1][1].StarCornerhole=cornor[1][2]
        elif len(cornor)==1:
            cornor[0][1].isConstraint=True
            cornor[0][1].isCornerhole=False 
   
    # step4: 标记固定边
    for i, segment in enumerate(poly_refs):
        # 颜色确定
        if segment.ref.color in segmentation_config.constraint_color and segment.length()>segmentation_config.toe_length:
            segment.isConstraint = True
            poly_refs[i].isConstraint = True
        # 角隅孔确定
        elif poly_refs[(i - 1) % len(poly_refs)].isCornerhole or poly_refs[(i + 1) % len(poly_refs)].isCornerhole:
            segment.isConstraint = True
            poly_refs[i].isConstraint = True
        elif isinstance(segment.ref,DLine):
            o_vs,o_ve=segment.ref.start_point,segment.ref.end_point
            vs,ve=segment.start_point,segment.end_point
            pairs=[(o_vs,vs),(o_vs,ve),(o_ve,vs),(o_ve,ve)]
            flag=True
            for pair in pairs:
                if DSegment(pair[0],pair[1]).length()<15:
                    flag=False
                    break
            if flag:
                segment.isConstraint = True
                poly_refs[i].isConstraint = True
        # elif isinstance(segment.ref,DLine) and segment.length()<0.85*DSegment(segment.ref.start_point,segment.ref.end_point).length():
        #     segment.isConstraint = True
        #     poly_refs[i].isConstraint = True
        # 平行线确定
        # else:
        #     dx_1 = segment.end_point.x - segment.start_point.x
        #     dy_1 = segment.end_point.y - segment.start_point.y
        #     mid_point=DPoint((segment.end_point.x+segment.start_point.x)/2,(segment.end_point.y+segment.start_point.y)/2)
        #     l = (dx_1**2 + dy_1**2)**0.5
        #     v_1 = (dy_1 / l * segmentation_config.parallel_max_distance, -dx_1 / l * segmentation_config.parallel_max_distance)
        #     point1,point2,point3=DSegment(segment.start_point,mid_point).mid_point(),DSegment(segment.end_point,mid_point).mid_point(),mid_point
        #     for j, other in enumerate(segments):
        #         if segment == other:
        #             continue
                
        #         if is_parallel(segment, other,segmentation_config.is_parallel_tolerance):
        #             #print(segment,other)
        #             s1 = DSegment(
        #                 DPoint(point1.x + v_1[0], point1.y + v_1[1]),
        #                 DPoint(point1.x - v_1[0], point1.y - v_1[1])
        #             )
        #             s2 = DSegment(
        #                 DPoint(point2.x + v_1[0], point2.y + v_1[1]),
        #                 DPoint(point2.x - v_1[0], point2.y - v_1[1])
        #             )
        #             s3 = DSegment(
        #                 DPoint(mid_point.x + v_1[0], mid_point.y + v_1[1]),
        #                 DPoint(mid_point.x - v_1[0], mid_point.y - v_1[1])
        #             )

        #             i1 = segment_intersection(s1.start_point, s1.end_point, other.start_point, other.end_point)
        #             if i1 == other.end_point or i1 == other.start_point:
        #                 i1 = None
                    
        #             i2 = segment_intersection(s2.start_point, s2.end_point, other.start_point, other.end_point)
        #             if i2 == other.end_point or i2 == other.start_point:
        #                 i2 = None
        #             i3 = segment_intersection(s3.start_point, s3.end_point, other.start_point, other.end_point)
        #             if i3 == other.end_point or i3 == other.start_point:
        #                 i3 = None
        #             if i1 is not None and DSegment(i1,point1).length()<segmentation_config.parallel_min_distance:
        #                 i1=None
        #             if i2 is not None and DSegment(i2,point2).length()<segmentation_config.parallel_min_distance:
        #                 i2=None
        #             if i3 is not None and DSegment(i3,point3).length()<segmentation_config.parallel_min_distance:
        #                 i3=None
        #             if i1 is not None and i2 is not None and i3 is not None:
        #                 segment.isConstraint = True
        #                 poly_refs[i].isConstraint = True
        #                 # if poly_refs[i].length()<=27 and poly_refs[i].length()>=25:
        #                 #     print(poly_refs[i])
        #                 #     print(other)
        #                 #     print(i1,i2,i3)
        #                 #print(segment,other)
        #                 break
    
            
   
    
    #加强结构
    sfs=stiffenersInPoly(stiffeners,poly,segmentation_config)
    is_fb=False
    others=set()
    st_segments=set()
    fb_segments=set()
    fl_segments=set()

    

    other_refs=[]
    distances=[]
    #查找相邻结构
    for i,segment in enumerate(poly_refs):
        # if not isinstance(segment.ref,DArc):
        dx_1 = segment.end_point.x - segment.start_point.x
        dy_1 = segment.end_point.y - segment.start_point.y
        mid_point=DPoint((segment.end_point.x+segment.start_point.x)/2,(segment.end_point.y+segment.start_point.y)/2)
        l = (dx_1**2 + dy_1**2)**0.5
        v_1 = (dy_1 / l * segmentation_config.parallel_max_distance_relax, -dx_1 / l * segmentation_config.parallel_max_distance_relax)
        point1,point2,point3=DSegment(segment.start_point,mid_point).mid_point(),DSegment(segment.end_point,mid_point).mid_point(),mid_point
        point4,point5=DSegment(DSegment(point1,point3).mid_point(),segment.start_point).mid_point(),DSegment(DSegment(point2,point3).mid_point(),segment.end_point).mid_point()
        point1=point4
        point2=point5
        # print(segment)
        # print(point1,point2,point3)
        d1,d2,d3=None,None,None
        seg1,seg2,seg3=None,None,None
        for j, other in enumerate(segments):
            if segment == other:
                continue

            if other.ref.handle in jg_s and segment.ref.handle not in jg_s:
                continue
            
            if is_parallel(segment, other,segmentation_config.is_parallel_tolerance):
                #print(segment,other)
                s1 = DSegment(
                    DPoint(point1.x + v_1[0], point1.y + v_1[1]),
                    DPoint(point1.x - v_1[0], point1.y - v_1[1])
                )
                s2 = DSegment(
                    DPoint(point2.x + v_1[0], point2.y + v_1[1]),
                    DPoint(point2.x - v_1[0], point2.y - v_1[1])
                )
                s3 = DSegment(
                    DPoint(point3.x + v_1[0], point3.y + v_1[1]),
                    DPoint(point3.x - v_1[0], point3.y - v_1[1])
                )

                i1 = segment_intersection(s1.start_point, s1.end_point, other.start_point, other.end_point)
                if i1 == other.end_point or i1 == other.start_point:
                    i1 = None
                
                i2 = segment_intersection(s2.start_point, s2.end_point, other.start_point, other.end_point)
                if i2 == other.end_point or i2 == other.start_point:
                    i2 = None
                i3 = segment_intersection(s3.start_point, s3.end_point, other.start_point, other.end_point)
                if i3 == other.end_point or i3 == other.start_point:
                    i3 = None
                # if i1 is not None and DSegment(i1,point1).length()<segmentation_config.parallel_min_distance_relax:
                #     i1=None
                # if i2 is not None and DSegment(i2,point2).length()<segmentation_config.parallel_min_distance_relax:
                #     i2=None
                # if i3 is not None and DSegment(i3,point3).length()<segmentation_config.parallel_min_distance_relax:
                #     i3=None

                if i1 is not None:
                    d=DSegment(i1,point1).length()
                    if d>=segmentation_config.parallel_min_distance_relax and d<=segmentation_config.parallel_max_distance_relax:
                        if seg1 is None:
                            seg1=other
                            d1=d
                        else:
                            
                            if d<d1:
                                seg1=other
                                d1=d
                        
                if i2 is not None:
                    d=DSegment(i2,point2).length()
                    if d>=segmentation_config.parallel_min_distance_relax and d<=segmentation_config.parallel_max_distance_relax:
                        if seg2 is None:
                            seg2=other
                            d2=d
                        else:
                            if d<d2:
                                seg2=other
                                d2=d
                if i3 is not None:
                    d=DSegment(i3,point3).length()
                    if d>=segmentation_config.parallel_min_distance_relax and d<=segmentation_config.parallel_max_distance_relax:
                        if seg3 is None:
                            seg3=other
                            d3=d
                        else:
                            if d<d3:
                                seg3=other
                                d3=d
            
                # if i1 is not None and i2 is not None and i3 is not None:
                #     distance=DSegment(i3,point3).length()
                #     l1=segment.length()
                #     l2=other.length()
                #     if  distance<segmentation_config.parallel_min_distance:
                #         continue
                #     q1,q2=other.start_point,other.end_point
                #     q3=other.mid_point()
                #     q4,q5=DSegment(q3,q1).mid_point(),DSegment(q3,q2).mid_point()
                #     count=0
                #     for q in [q1,q2,q3,q4,q5]:
                #         if point_is_inside(q,polygon) :
                #             count+=1
                #     if count>2:
                #         is_inside=True
                #     else:
                #         is_inside=False
                #     if distance < segmentation_config.parallel_max_distance and is_inside==False:
                #         #contraint
                #         others.add(other)
                #         segment.isPart=True
                #         poly_refs[i].isPart=True
                #         segment.isConstraint = True
                #         poly_refs[i].isConstraint = True
                #     elif l1<l2 *segmentation_config.contraint_factor and is_inside==False:
                #         #constraint
                #         others.add(other)
                #         segment.isPart=True
                #         poly_refs[i].isPart=True
                #         segment.isConstraint = True
                #         poly_refs[i].isConstraint = True
                #     elif l1<segmentation_config.free_edge_min_length:
                #         others.add(other)
                #         segment.isPart=True
                #         poly_refs[i].isPart=True
                #         segment.isConstraint = True
                #         poly_refs[i].isConstraint = True
                #     else:
                #         if is_inside:
                #             is_fb=True
                #             fb_segments.add(other)
                #             other.isFb=True
                #         else:
                #             fl_segments.add(other)

                #         # others.add(other)
                #         segment.isPart=True
                #         poly_refs[i].isPart=True
         
                            # st_segments.add(segment)
        other_refs.append([seg1,seg2,seg3])
        distances.append([d1,d2,d3])
    

    for i,segment in enumerate(poly_refs):
        d1,d2,d3=distances[i][0],distances[i][1],distances[i][2]
        s1,s2,s3=other_refs[i][0],other_refs[i][1],other_refs[i][2]
        if s3 is not None:
            if d1 is not None and abs(d1-d3)>1 :
                s1=None
            if d2 is not None and abs(d2-d3)>1:
                s2=None
        if s1 is not None and s2 is not None and s3 is not None:
            # print(d1,d2,d3)
            # print(s1,s2,s3)
            distance=d3
            l1=segment.length()
            s_set=set()
            s_set.add(s1)
            s_set.add(s2)
            s_set.add(s3)
            l2=0
            for other in list(s_set):
                l2+=other.length()
            # print(distance,l1,l2)
            other=s3
            q1,q2=other.start_point,other.end_point
            q3=other.mid_point()
            q4,q5=DSegment(q3,q1).mid_point(),DSegment(q3,q2).mid_point()
            count=0
            for q in [q1,q2,q3,q4,q5]:
                if point_is_inside(q,polygon) :
                    count+=1
            if count>2:
                is_inside=True
            else:
                is_inside=False
            if distance < segmentation_config.parallel_max_distance and is_inside==False and not isinstance(segment.ref,DArc) and l1>=segmentation_config.constraint_min_length and l1 > segmentation_config.toe_length:
                #contraint
                # print(1)
                others.add(other)
                segment.isPart=True
                poly_refs[i].isPart=True
                segment.isConstraint = True
                poly_refs[i].isConstraint = True
            elif l1<l2 *segmentation_config.contraint_factor and is_inside==False and not isinstance(segment.ref,DArc) and l1>=segmentation_config.constraint_min_length and l1 > segmentation_config.toe_length:
                poly_refs[i].isPart=True
                segment.isConstraint = True
                poly_refs[i].isConstraint = True
            elif l1<segmentation_config.free_edge_min_length and is_inside==False and not isinstance(segment.ref,DArc) and l1>=segmentation_config.constraint_min_length and l1 > segmentation_config.toe_length:
                # print(3)
                others.add(other)
                segment.isPart=True
                poly_refs[i].isPart=True
                segment.isConstraint = True
                poly_refs[i].isConstraint = True
            elif l1<l2 *segmentation_config.contraint_factor*0.5 and is_inside==False and not isinstance(segment.ref,DArc) and l1<segmentation_config.constraint_min_length and l1 > segmentation_config.toe_length:
                # print(l2)
                others.add(other)
                segment.isPart=True
                poly_refs[i].isPart=True
                segment.isConstraint = True
                poly_refs[i].isConstraint = True
            elif l1>=segmentation_config.free_edge_min_length:

                # print(4)
                if is_inside:
                    is_fb=True
                    fb_segments.add(other)
                    other.isFb=True
                else:
                    fl_segments.add(other)

                # others.add(other)
                segment.isPart=True
                poly_refs[i].isPart=True
         
     
     
     # 属于同一参考线的边只要有一个是固定边，则所有都是固定边
   
    for i,segment in enumerate(poly_refs):
        if not segment.isCornerhole and poly_refs[(i-1+len(poly_refs))%len(poly_refs)].isConstraint and poly_refs[(i+1+len(poly_refs))%len(poly_refs)].isConstraint:
            segment.isConstraint = True
            poly_refs[i].isConstraint = True

    # for a in range(len(poly_refs)):
    #     for b in range(len(poly_refs)):
    #         if isinstance(poly_refs[a].ref, DLwpolyline) and isinstance(poly_refs[b].ref, DLwpolyline):
    #             if poly_refs[a].ref.points[0] == poly_refs[b].ref.points[0] and poly_refs[a].ref.points[-1] == poly_refs[b].ref.points[-1]:
    #                 poly_refs[b].isConstraint = poly_refs[a].isConstraint or poly_refs[b].isConstraint
    #                 poly_refs[a].isConstraint = poly_refs[a].isConstraint or poly_refs[b].isConstraint
    #         if isinstance(poly_refs[a].ref, DLine) and isinstance(poly_refs[b].ref, DLine):
    #             if poly_refs[a].ref.start_point == poly_refs[b].ref.start_point and poly_refs[a].ref.end_point == poly_refs[b].ref.end_point:
    #                 poly_refs[b].isConstraint = poly_refs[a].isConstraint or poly_refs[b].isConstraint
    #                 poly_refs[a].isConstraint = poly_refs[a].isConstraint or poly_refs[b].isConstraint


    #相邻夹角小的边，如果有一个是固定边，则所有都是固定边
    # for i,s in enumerate(poly):
    #     current=i
    #     next=(i+1)%len(poly)
       
    #     if conpute_angle_of_two_segments(poly[current],poly[next])  <segmentation_config.constraint_split_angle:
            
    #         if poly[current].isConstraint:
             
    #             # print(poly[current],poly[next])
    #             poly[next].isConstraint=True
    #         if  poly[next].isConstraint:
    #             poly[current].isConstraint=True



    st_segments=list(st_segments)
    others=list(others)
    fb_segments=list(fb_segments)
    fl_segments=list(fl_segments)
    if len(sfs)>0 or len(fb_segments)>0:
        is_fb=True
    others.extend(fb_segments)
    others.extend(fl_segments)
    # step:5 确定自由边，合并相邻的固定边
    constraint_edges = []
    free_edges = []
    cornerhole_edges = []
    edges = []

    # 初始化合并边的列表和当前的合并状态
    current_edge = []
    current_type = None  # 使用三种类型：'constraint', 'free', 'cornerhole'

    for segment in poly_refs:
        # 处理角隅孔
        if segment.isCornerhole:
            # 如果当前是角隅孔边
            if current_type == 'cornerhole':
                current_edge.append(segment)  # 将当前段添加到当前的角隅孔边
            else:
                # 保存上一个合并边
                if current_edge:
                    if current_type == 'constraint':
                        constraint_edges.append(current_edge)
                    elif current_type == 'free':
                        free_edges.append(current_edge)
                    edges.append(current_edge)

                # 开始新的角隅孔边合并
                current_edge = [segment]
                current_type = 'cornerhole'

        # 处理固定边
        elif segment.isConstraint:
            if current_type == 'constraint':  # 如果当前在合并固定边
                if current_edge[-1].ref.color == segment.ref.color:  # 如果颜色相同
                    current_edge.append(segment)  # 添加到当前边
                else:
                    # 保存当前合并边，并开始新的合并边
                    constraint_edges.append(current_edge)
                    edges.append(current_edge)
                    current_edge = [segment]  # 开始新的合并边
            else:
                # 保存上一个合并边
                if current_edge:
                    if current_type == 'free':
                        free_edges.append(current_edge)
                    elif current_type == 'cornerhole':
                        cornerhole_edges.append(current_edge)
                    edges.append(current_edge)
                
                # 开始新的固定边合并
                current_edge = [segment]
                current_type = 'constraint'

        # 处理自由边
        else:
            if current_type == 'free':  # 如果当前在合并自由边
                
                current_edge.append(segment)  # 添加到当前边
                
            else:
                # 保存上一个合并边
                if current_edge:
                    if current_type == 'constraint':
                        constraint_edges.append(current_edge)
                    elif current_type == 'cornerhole':
                        cornerhole_edges.append(current_edge)
                    edges.append(current_edge)
                
                # 开始新的自由边合并
                current_edge = [segment]
                current_type = 'free'

    # 添加最后一个合并边，并检查与第一条边的属性
    if current_edge:
        if current_type == 'constraint':
            # 检查最后一个合并边的属性是否与第一条边相同
            if poly_refs[0].isConstraint and not poly_refs[0].isCornerhole and len(constraint_edges) > 0:
                constraint_edges[0] = current_edge + constraint_edges[0]  # 合并到第一条固定边
                edges[0] = current_edge + edges[0]
            else:
                constraint_edges.append(current_edge)
                edges.append(current_edge)
        elif current_type == 'free':
            if not poly_refs[0].isConstraint and not poly_refs[0].isCornerhole and len(free_edges) > 0:
                free_edges[0] = current_edge + free_edges[0]  # 合并到第一条自由边
                edges[0] = current_edge + edges[0]
            else:
                free_edges.append(current_edge)
                edges.append(current_edge)
        elif current_type == 'cornerhole':
            if poly_refs[0].isCornerhole and len(cornerhole_edges) > 0:
                cornerhole_edges[0] = current_edge + cornerhole_edges[0]
                edges[0] = current_edge + edges[0]
            else:
                cornerhole_edges.append(current_edge)
                edges.append(current_edge)




    # step 5.5：找到所有的标注
    
    # print(len(stiffeners))
    # print(len(sfs))
    
    # if len(sfs)>0:
    #     is_fb=True
    #     print(f"回路{index}有加强边！")
    # else:
    #     print(f"回路{index}没有加强边！")
    
    tis=textsInPoly(text_map,poly,segmentation_config,is_fb,polygon)
    ds=dimensionsInPoly(dimensions,poly,segmentation_config)




    # print(free_edges)
    # print(cornor_holes[0].segments)
    # for s in poly_refs:
    #     print(s.isCornerhole)
    # print(current_edge)
    # print(current_type)
    # # 附加：添加筛选条件
    # # 如果有多于两条的自由边则判定为不是肘板，不进行输出
    # print(constraint_edges)
    # print(free_edges)
    # print(cornerhole_edges)
    # for s in poly:
    #     print(s.isCornerhole)
 
  
        

    # print(constraint_edges)
    # print("===========")
    # print(free_edges)
    # print("=============")
    # print(cornerhole_edges)
    if segmentation_config.mode=="dev":
        plot_info_poly(polygon,poly_refs, os.path.join(segmentation_config.poly_info_dir, f'所有肘板图像(仅限开发模式)/infopoly{index}.png'),tis,ds,sfs,others)
    if len(free_edges) > 1:
        print(f"回路{index}超过两条自由边！")
        #return poly_refs
        return None
    if len(free_edges) == 0:
        print(f"回路{index}没有自由边！")
        #return poly_refs
        return  None
    if len(constraint_edges)==0:
        print(f"回路{index}没有约束边！")
        #return poly_refs
        return None
    
    # 依据自由边的起始，调整固定边和角隅孔的顺序
    cycle_edges = edges + edges
    constarint_cornerhole_edges = []
    constarint_edges = []
    cornerhole_edges = []
    start = False
    for edge_ in cycle_edges:
        # 如果遇到第一个非 Cornerhole 和非 Constraint 边，开始收集
        if not start and edge_[0].isCornerhole == False and edge_[0].isConstraint == False:
            start = True
        
        # 如果已经开始收集，且当前边是 Cornerhole 或 Constraint，加入 al_edges
        elif start and (edge_[0].isCornerhole or edge_[0].isConstraint):
            constarint_cornerhole_edges.append(edge_)
            if edge_[0].isCornerhole:
                cornerhole_edges.append(edge_)
            elif edge_[0].isConstraint:
                constarint_edges.append(edge_)
        
        # 如果已经开始收集，且当前边不再是 Cornerhole 或 Constraint，停止收集
        elif start and edge_[0].isCornerhole == False and edge_[0].isConstraint == False:
            start = False
    edges = free_edges + constarint_cornerhole_edges
    # print(len(cornerhole_edges))
    # print(len(cornerhole_edges))
    # print(len(cornerhole_edges))
    # print(len(cornerhole_edges))
    #分割固定边
    new_edges=[]
    for edge_ in edges:
        if edge_[0].isConstraint and not edge_[0].isCornerhole:

    
            angles=[0]
            constarint_edge=edge_
            for i in range(len(constarint_edge)-1):
                angles.append(conpute_angle_of_two_segments(constarint_edge[i],constarint_edge[i+1]))
            edge=[]
            for i,angle in enumerate(angles):
                if angle<=segmentation_config.constraint_split_angle:
                    edge.append(constarint_edge[i])
                else:
                    new_edges.append(edge)
                    edge=[]
                    edge.append(constarint_edge[i])
            new_edges.append(edge)
        else:
            new_edges.append(edge_)
    edges=new_edges
    # for edge in edges:
    #     print(edge[0].isConstraint,edge[0].isCornerhole)
    new_constraint_edges=[]
    for constarint_edge in constraint_edges:
        angles=[0]
        for i in range(len(constarint_edge)-1):
            angles.append(conpute_angle_of_two_segments(constarint_edge[i],constarint_edge[i+1]))
        edge=[]
        for i,angle in enumerate(angles):
            if angle<=segmentation_config.constraint_split_angle:
                edge.append(constarint_edge[i])
            else:
                new_constraint_edges.append(edge)
                edge=[]
                edge.append(constarint_edge[i])
        new_constraint_edges.append(edge)
    constraint_edges=new_constraint_edges

    # 如果回路轮廓线条数小于等于4条，则不进行输出
    if len(poly_refs) <= 4:
        print(f"回路{index}轮廓线条数小于等于4条！")
        return None
    
    # 如果除去圆弧外固定边多边形不是凸多边形则不进行输出
    constraint_edge_poly = []

    for constarint_edge in constraint_edges:
        for seg in constarint_edge:
            if not (isinstance(seg.ref, DArc) and seg.ref.radius<200):
                constraint_edge_poly.append(seg.start_point)
                constraint_edge_poly.append(seg.end_point)

    if not is_near_convex(constraint_edge_poly, index,segmentation_config.near_convex_tolerance):
        print(f"回路{index}去圆弧外固定边多边形不是凸多边形")
        return None
    
    # 如果整体轮廓不是凸多边形则不进行输出
    # all_edge_poly = []

    # for edge in edges:
    #     for seg in edge:
    #         if seg.isConstraint and not seg.isCornerhole and not (isinstance(seg.ref, DArc) and seg.ref.radius<200):
    #             all_edge_poly.append(seg.start_point)
    #             all_edge_poly.append(seg.end_point)
    #         elif not (seg.isConstraint or seg.isCornerhole):
    #             all_edge_poly.append(seg.start_point)
    #             all_edge_poly.append(seg.end_point)
    # if  is_near_convex(all_edge_poly, index,segmentation_config.near_convex_tolerance):
    #     print(f"回路{index}整体轮廓不是凸多边形")
    #     return None
    
    
    # 如果自由边轮廓中出现了夹角小于45°的折线则不进行输出
    for i in range(len(free_edges[0]) - 1):
        seg1 = free_edges[0][i]
        seg2 = free_edges[0][i + 1]
        if isinstance(seg1.ref, DArc) or isinstance(seg2.ref, DArc):
            continue
        angle = calculate_angle(seg1.start_point, seg1.end_point, seg2.end_point)
        if angle < segmentation_config.min_angle_in_free_edge:
            print(f"回路{index}自由边轮廓中出现了夹角小于45°的折线")
            return None

    # 如果自由边长度在总轮廓长度中占比不超过25%则不进行输出
    free_len = 0
    all_len = 0
    for seg in poly_refs:
        if isinstance(seg.ref, DArc):
            l = ((seg.ref.end_point.x - seg.ref.start_point.x) ** 2 +   
                (seg.ref.end_point.y - seg.ref.start_point.y) ** 2) ** 0.5
        else:
            l = seg.length()
        if not seg.isConstraint and not seg.isCornerhole:
            free_len += l
        all_len += l
    if free_len / all_len < segmentation_config.free_edge_ratio:
        print(f"回路{index}自由边长度在总轮廓长度中占比不超过{segmentation_config.free_edge_ratio*100}%")
        return None
    
    # 如果整体肘板轮廓接近于矩形，则不进行输出
    max_x = float('-inf')
    min_x = float('inf')
    max_y = float('-inf')
    min_y = float('inf')
    points = []
    for seg in poly_refs:
        if isinstance(seg.ref, DArc):
            continue
        points.append(seg.start_point)
        points.append(seg.end_point)
        x_coords = [seg.start_point[0], seg.end_point[0]]
        y_coords = [seg.start_point[1], seg.end_point[1]]

        max_x = max(max_x, *x_coords)
        min_x = min(min_x, *x_coords)
        max_y = max(max_y, *y_coords)
        min_y = min(min_y, *y_coords)

    area = (max_x - min_x) * (max_y - min_y)
    if is_near_rectangle(points, area, segmentation_config.near_rectangle_tolerance):
        print("肘板轮廓接近矩形！")
        return None



    new_edges=[]
    new_constraint_edges=[]
    new_poly_ref=[]
    for i,edge in enumerate(edges):
        if (not edge[0].isConstraint) and (not edge[0].isCornerhole):
            new_edges.append(edge)
            continue
        if edge[0].isCornerhole:
            new_edges.append(edge)
            continue
        else:
            if len(edge)==1 and isinstance(edge[0].ref,DLine):
                new_edges.append(edge)
                new_constraint_edges.append(edge)
                continue
            ori_segs=[]
            for seg in edge:
                if seg in ref_map:
                    ori_segs.extend(ref_map[seg])
                else:
                    ori_segs.append(seg)
            ori_map={}
            for s in ori_segs:
                if s.start_point not in ori_map:
                    ori_map[s.start_point]=1
                else:
                    ori_map[s.start_point]+=1
                if s.end_point not in ori_map:
                    ori_map[s.end_point]=1
                else:
                    ori_map[s.end_point]+=1
            ps=[]
            for p,cnt in ori_map.items():
                if cnt==1:
                    ps.append(p)
            if len(ps)!=2:
                new_edges.append(edge)
                new_constraint_edges.append(edge)
            else:
                new_ref=DLine(ps[0],ps[1])
                new_seg=DSegment(ps[0],ps[1],new_ref)
                new_seg.isConstraint=True
                new_seg.isCornerhole=False
                ref_map[new_seg]=ori_segs
                new_edges.append([new_seg])
                new_constraint_edges.append([new_seg])
    
    edges=new_edges
    constraint_edges=new_constraint_edges

    # step6: 绘制对边分类后的几何图像
    plot_info_poly(polygon,poly_refs, os.path.join(segmentation_config.poly_info_dir, f'所有有效回路图像/infopoly{index}.png'),tis,ds,sfs,others)

    # for i,t_t in enumerate(tis):
    #     print(t_t[0].content)
    #标注匹配
    r_anno=[]
    l_anno=[]
    a_anno=[]
    c_anno=[]
    for i,t_t in enumerate(tis):
        if t_t[2]["Type"]=="R":
            r_anno.append((t_t[1],t_t[0]))
        if t_t[2]["Type"]=="CornerHole":
            c_anno.append((t_t[1],t_t[0]))
    for i,d_t in enumerate(ds):
        d=d_t[0]
        pos=d_t[1]
        if  d.dimtype==32 or d.dimtype==33 or d.dimtype==161 or d.dimtype==160:
            l0=p_minus(d.defpoints[0],d.defpoints[2])
            l1=p_minus(d.defpoints[1],d.defpoints[2])
            d10=l0.x*l1.x+l0.y*l1.y
            d00=l0.x*l0.x+l0.y*l0.y
            if d00 <1e-4:
                x=d.defpoints[1]
            else:
                x=p_minus(p_add(d.defpoints[1],l0),p_mul(l0,d10/d00))
            d1,d2,d3,d4=d.defpoints[0], x,d.defpoints[1],d.defpoints[2]
            d1,d2,d3,d4=compute_accurate_position(d1,d2,d3,d4,constraint_edges)
            l_anno.append((d3,d4,d))
        elif d.dimtype==34 or d.dimtype==162:
            s1=DSegment(d.defpoints[3],d.defpoints[0])
            s2=DSegment(d.defpoints[2],d.defpoints[1])
            
            inter=segment_intersection_line(s1.start_point,s1.end_point,s2.start_point,s2.end_point)
            pos=d.defpoints[4]
            r=DSegment(pos,inter).length()
            v1=DPoint((s1.end_point.x-s1.start_point.x)/s1.length()*(r+5),(s1.end_point.y-s1.start_point.y)/s1.length()*(r+5))
            v2=DPoint((s2.end_point.x-s2.start_point.x)/s2.length()*(r+5),(s2.end_point.y-s2.start_point.y)/s2.length()*(r+5))
            d1=DPoint(v1.x+inter.x,v1.y+inter.y)
            d2=DPoint(-v1.x+inter.x,-v1.y+inter.y)
            d3=DPoint(v2.x+inter.x,v2.y+inter.y)
            d4=DPoint(-v2.x+inter.x,-v2.y+inter.y)
            p1=None
            p2=None
            if DSegment(pos,d1).length()<DSegment(pos,d2).length():
                p1=d1
            else:
                p1=d2
            if DSegment(pos,d3).length()<DSegment(pos,d4).length():
                p2=d3
            else:
                p2=d4
            a_anno.append((p1,p2,inter,d))

    all_the_edge=[]
    for edge in edges:
        for seg in edge:
            all_the_edge.append(seg)
    all_the_edges=[all_the_edge]
    r_map,radius_anno=match_r_anno(r_anno,all_the_edges)
    l_whole_map,l_half_map,l_cornor_map,l_para_map,l_para_single_map,l_n_para_map,l_n_para_single_map,l_ver_map,l_ver_single_map,d_map,whole_anno,half_anno,cornor_anno,parallel_anno,non_parallel_anno,vertical_anno,d_anno=match_l_anno(l_anno,poly_refs,constraint_edges,free_edges,segmentation_config)
    a_map,angle_anno,toe_angle_anno=match_a_anno(a_anno,free_edges,constraint_edges)
    all_anno=(c_anno,radius_anno,whole_anno,half_anno,cornor_anno,parallel_anno,non_parallel_anno,vertical_anno,d_anno,angle_anno,toe_angle_anno)
    all_map=(r_map,l_whole_map,l_half_map,l_cornor_map,l_para_map,l_para_single_map,l_n_para_map,l_n_para_single_map,l_ver_map,l_ver_single_map,d_map,a_map)


    all_edge_map,edge_types,features,feature_map,constraint_edge_no,free_edge_no=match_edge_anno(segments,constraint_edges,free_edges,edges,all_anno,all_map)

    

    is_standard_elbow=True
    bracket_parameter=None
    strengthen_parameter=None
    thickness=0
    has_B_anno=False
    for t_t in tis:
        content=t_t[0].content.strip()
        if t_t[2]["Type"]=="B_anno":
            has_B_anno=True
            break
    has_B_hint=False
    has_fb=False
    for t_t in tis:
        content=t_t[0].content.strip()
        if t_t[2]["Type"]=="BK":
            has_B_anno=True
        if t_t[2]["Type"]=="B":
            has_B_hint=True
            # print(len(content))
            if "B" in content and "B"==content[0] and len(content)>1 and content[1] in ["0","1","2","3","4","5","6","7","8","9"]:
                has_B_anno=True
            bracket_parameter=t_t[2]
            if bracket_parameter["Thickness"] is not None:
                thickness=bracket_parameter["Thickness"]
        if t_t[2]["Type"]=="FB" or t_t[2]["Type"]=="FL":
            strengthen_parameter=t_t[2]
            has_fb=True

    if has_B_hint==True and has_B_anno==False:
        is_standard_elbow=False
    if has_B_hint==False and has_fb==True:
        is_standard_elbow=False
    for t_t in tis:
        content=t_t[0].content.strip()
        # print(content,t_t[2]["Type"])
    has_hint= bracket_parameter is not None or strengthen_parameter is not None
    for t_t in tis:
        if has_hint:
            break
        content=t_t[0].content.strip()
        if t_t[2]["Type"]=="R" or t_t[2]["Type"]=="BK":
            has_hint=True
            break
    for seg,dic in all_edge_map.items():
        if has_hint:
            break
        for ty,d_s in dic.items():
            if isinstance(d_s,list):
                if len(d_s)>0:
                    d_t=d_s[0]
                    if isinstance(d_t[0],DDimension) and d_t[0].dimtype==0:
                        continue
                    has_hint=True
                    break
        if has_hint:
            break
        

    
    is_diff=False
    meta_info=(is_standard_elbow,bracket_parameter,strengthen_parameter,has_hint,is_fb,is_diff)
    edges_info=(free_edges,constraint_edges,edges,poly_refs,ref_map)
    hint_info=(all_edge_map,edge_types,features,feature_map,constraint_edge_no,free_edge_no,all_anno,tis,ds)

    return edges_info,poly_centroid,hint_info,meta_info
def is_similar(edges_info,other_edges_info,edge_types,other_edge_types):


    edges1=edges_info[2]
    non_free_edges1=[]
    non_free_edges_types1=[]
    free_edges1=[]
    free_edges_types1=[]
    for edge in edges_info[0][0]:
        free_edges1.append(edge)
        free_edges_types1.append(edge_types[edge])
    for i,edge in enumerate(edges1):
        if (not edge[0].isConstraint) and (not edge[0].isCornerhole):
            continue
        if edge[0].isCornerhole:
            non_free_edges_types1.append('cornerhole')
            segs=[]
            for s in edge:
                segs.append(s)
            non_free_edges1.append(segs)
        else:
            non_free_edges_types1.append('constraint')
            segs=[]
            for s in edge:
                segs.append(s)
            non_free_edges1.append(segs)
    r_non_free_edges1=non_free_edges1[::-1]
    r_non_free_edges_types1=non_free_edges_types1[::-1]
    r_free_edges1=free_edges1[::-1]
    r_free_edges_types1=free_edges_types1[::-1]
    new_edges=[]
    for edge in r_non_free_edges1:
        new_edges.append(edge[::-1])
    r_non_free_edges1=new_edges
    



    edges2=other_edges_info[2]
    non_free_edges2=[]
    non_free_edges_types2=[]
    free_edges2=[]
    free_edges_types2=[]
    for edge in other_edges_info[0][0]:
        free_edges2.append(edge)
        free_edges_types2.append(other_edge_types[edge])
    for i,edge in enumerate(edges2):
        if (not edge[0].isConstraint) and (not edge[0].isCornerhole):
            continue
        if edge[0].isCornerhole:
            non_free_edges_types2.append('cornerhole')
            segs=[]
            for s in edge:
                segs.append(s)
            non_free_edges2.append(segs)
        else:
            non_free_edges_types2.append('constraint')
            segs=[]
            for s in edge:
                segs.append(s)
            non_free_edges2.append(segs)
    r_non_free_edges2=non_free_edges2[::-1]
    r_non_free_edges_types2=non_free_edges_types2[::-1]
    r_free_edges2=free_edges2[::-1]
    r_free_edges_types2=free_edges_types2[::-1]
    new_edges=[]
    for edge in r_non_free_edges2:
        new_edges.append(edge[::-1])
    r_non_free_edges2=new_edges



    des1=""
    for edge in free_edges_types1:
        des1+=edge
    for i in range(len(non_free_edges_types1)):
        des1+=non_free_edges_types1[i]
        for edge in non_free_edges1[i]:
            if isinstance(edge.ref,DLine):
                des1+="line"
            else:
                des1+="arc"
    
    des2=""
    for edge in free_edges_types2:
        des2+=edge
    for i in range(len(non_free_edges_types2)):
        des2+=non_free_edges_types2[i]
        for edge in non_free_edges2[i]:
            if isinstance(edge.ref,DLine):
                des2+="line"
            else:
                des2+="arc"
    # print("111111111111111111")
    # print(des1,des2)
    if des1==des2:
        point_map={}
        order_list1=[]
        cons_edges1=[]
        for i in range(len(non_free_edges1)):
            for j in range(len(non_free_edges1[i])):
                cons_edges1.append(non_free_edges1[i][j])

        order_list2=[]
        cons_edges2=[]
        for i in range(len(non_free_edges2)):
            for j in range(len(non_free_edges2[i])):
                cons_edges2.append(non_free_edges2[i][j])



        for i in range(len(cons_edges1)-1):
            current_edge=cons_edges1[i]
            next_edge=cons_edges1[i+1]
            pairs=[(current_edge.start_point,next_edge.start_point),(current_edge.start_point,next_edge.end_point),(current_edge.end_point,next_edge.start_point),(current_edge.end_point,next_edge.end_point)]
            distance=float('inf')
            min_pair=None
            for pair in pairs:
                p1,p2=pair
                dis=DSegment(p1,p2).length()
                if dis< distance:
                    distance=dis
                    min_pair=pair
            p2,p=min_pair
            if  p2==current_edge.start_point:
                p1=current_edge.end_point
            else:
                p1=current_edge.start_point
            order_list1.append(p1)
            order_list1.append(p2)
            if i==len(cons_edges1)-2:
                order_list1.append(p)
                if p==next_edge.start_point:
                    order_list1.append(next_edge.end_point)
                else:
                    order_list1.append(next_edge.start_point)



        for i in range(len(cons_edges2)-1):
            current_edge=cons_edges2[i]
            next_edge=cons_edges2[i+1]
            pairs=[(current_edge.start_point,next_edge.start_point),(current_edge.start_point,next_edge.end_point),(current_edge.end_point,next_edge.start_point),(current_edge.end_point,next_edge.end_point)]
            distance=float('inf')
            min_pair=None
            for pair in pairs:
                p1,p2=pair
                dis=DSegment(p1,p2).length()
                if dis< distance:
                    distance=dis
                    min_pair=pair
            p2,p=min_pair
            if  p2==current_edge.start_point:
                p1=current_edge.end_point
            else:
                p1=current_edge.start_point
            order_list2.append(p1)
            order_list2.append(p2)
            if i==len(cons_edges2)-2:
                order_list2.append(p)
                if p==next_edge.start_point:
                    order_list2.append(next_edge.end_point)
                else:
                    order_list2.append(next_edge.start_point)
        for i in range(min(len(order_list1),len(order_list2))):
            point_map[order_list2[i]]=order_list1[i]

        edge_map={}
        for i,edge in enumerate(free_edges1):
            edge_map[free_edges2[i]]=edge
        for i in range(len(non_free_edges1)):
            for j in range(len(non_free_edges1[i])):
                edge_map[non_free_edges2[i][j]]=non_free_edges1[i][j]
        return True,edge_map,point_map
    
    

    des2=""
    for edge in r_free_edges_types2:
        des2+=edge
    for i in range(len(r_non_free_edges_types2)):
        des2+=r_non_free_edges_types2[i]
        for edge in r_non_free_edges2[i]:
            if isinstance(edge.ref,DLine):
                des2+="line"
            else:
                des2+="arc"
    # print("222222222222222222222222")
    # print(des1,des2)
    if des1==des2:
        point_map={}
        order_list1=[]
        cons_edges1=[]
        for i in range(len(non_free_edges1)):
            for j in range(len(non_free_edges1[i])):
                cons_edges1.append(non_free_edges1[i][j])

        order_list2=[]
        cons_edges2=[]
        for i in range(len(r_non_free_edges2)):
            for j in range(len(r_non_free_edges2[i])):
                cons_edges2.append(r_non_free_edges2[i][j])



        for i in range(len(cons_edges1)-1):
            current_edge=cons_edges1[i]
            next_edge=cons_edges1[i+1]
            pairs=[(current_edge.start_point,next_edge.start_point),(current_edge.start_point,next_edge.end_point),(current_edge.end_point,next_edge.start_point),(current_edge.end_point,next_edge.end_point)]
            distance=float('inf')
            min_pair=None
            for pair in pairs:
                p1,p2=pair
                dis=DSegment(p1,p2).length()
                if dis< distance:
                    distance=dis
                    min_pair=pair
            p2,p=min_pair
            if  p2==current_edge.start_point:
                p1=current_edge.end_point
            else:
                p1=current_edge.start_point
            order_list1.append(p1)
            order_list1.append(p2)
            if i==len(cons_edges1)-2:
                order_list1.append(p)
                if p==next_edge.start_point:
                    order_list1.append(next_edge.end_point)
                else:
                    order_list1.append(next_edge.start_point)



        for i in range(len(cons_edges2)-1):
            current_edge=cons_edges2[i]
            next_edge=cons_edges2[i+1]
            pairs=[(current_edge.start_point,next_edge.start_point),(current_edge.start_point,next_edge.end_point),(current_edge.end_point,next_edge.start_point),(current_edge.end_point,next_edge.end_point)]
            distance=float('inf')
            min_pair=None
            for pair in pairs:
                p1,p2=pair
                dis=DSegment(p1,p2).length()
                if dis< distance:
                    distance=dis
                    min_pair=pair
            p2,p=min_pair
            if  p2==current_edge.start_point:
                p1=current_edge.end_point
            else:
                p1=current_edge.start_point
            order_list2.append(p1)
            order_list2.append(p2)
            if i==len(cons_edges2)-2:
                order_list2.append(p)
                if p==next_edge.start_point:
                    order_list2.append(next_edge.end_point)
                else:
                    order_list2.append(next_edge.start_point)
        for i in range(min(len(order_list1),len(order_list2))):
            point_map[order_list2[i]]=order_list1[i]

        edge_map={}
        for i,edge in enumerate(free_edges1):
            edge_map[r_free_edges2[i]]=edge
        for i in range(len(non_free_edges1)):
            for j in range(len(non_free_edges1[i])):
                edge_map[r_non_free_edges2[i][j]]=non_free_edges1[i][j]
        return True,edge_map,point_map

    return False,None,None
def calculate_new_hint_info(edges_info,other_edges_info,hint_info,other_hint_info,meta_info,other_meta_info,edge_map,point_map):

    new_all_edge_map={}
    other_all_edge_map=other_hint_info[0]
 
    for seg,dic in other_all_edge_map.items():
        s=edge_map[seg]
        new_dic={}
        for ty,d_s in dic.items():
            if isinstance(d_s,list):
                new_d_s=[]
                for d_t in d_s:
                    if len(d_t)==4:
                        if isinstance(d_t[0],DDimension) and d_t[0].dimtype!=0:
                            content=f"值：{d_t[0].text}"
                        elif isinstance(d_t[0],DDimension):
                            content=f"{d_t[1]}"
                        else:
                            content=f"值：{d_t[0].content}"
                        sub_ty=d_t[1].split("：")[-1]
                        if isinstance(d_t[0],DDimension) and d_t[0].dimtype!=0:
                            new_d_s.append([d_t[0],f"值：{content}，参考边：{sub_ty}",edge_map[d_t[2]],(edge_map[d_t[2]],point_map[d_t[3][1]],point_map[d_t[3][1]])])
                        elif isinstance(d_t[0],DDimension):
                            new_d_s.append([d_t[0],f"{d_t[1]}",edge_map[d_t[2]]])
                        else:
                            new_d_s.append([d_t[0],f"值：{content}，参考边：{sub_ty}",edge_map[d_t[2]]])
                    elif len(d_t)==3:
                        if isinstance(d_t[0],DDimension) and d_t[0].dimtype!=0:
                            content=f"值：{d_t[0].text}"
                        elif isinstance(d_t[0],DDimension):
                            content=f"{d_t[1]}"
                        else:
                            content=f"值：{d_t[0].content}"
                        sub_ty=d_t[1].split("：")[-1]
                        if isinstance(d_t[0],DDimension) and d_t[0].dimtype!=0:
                            new_d_s.append([d_t[0],f"值：{content}，参考边：{sub_ty}",edge_map[d_t[2]]])
                        elif isinstance(d_t[0],DDimension):
                            new_d_s.append([d_t[0],f"{d_t[1]}",edge_map[d_t[2]]])
                        else:
                            new_d_s.append([d_t[0],f"值：{content}，参考边：{sub_ty}",edge_map[d_t[2]]])
                        
                    else:
                        if isinstance(d_t[0],DDimension) and d_t[0].dimtype!=0:
                            content=f"值：{d_t[0].text}"
                        elif  isinstance(d_t[0],DDimension):
                            content=f"{d_t[1]}"
                        else:
                            content=f"值：{d_t[0].content}"
                        new_d_s.append([d_t[0],content])

                new_dic[ty]=new_d_s
            else:
                new_dic[ty]=d_s
        new_all_edge_map[s]=new_dic
    new_features=other_hint_info[2]
    new_feature_map={}
    for seg,feats in other_hint_info[3].items():
        new_feature_map[edge_map[seg]]=feats
    new_hint_info=(new_all_edge_map,hint_info[1],new_features,new_feature_map,hint_info[4],hint_info[5],hint_info[6],hint_info[7],hint_info[8])

    new_meta_info=(meta_info[0],other_meta_info[1],other_meta_info[2],other_meta_info[3],other_meta_info[4],True)
    return new_hint_info,new_meta_info
def calculate_new_hint_info_bk(edges_info,other_edges_info,hint_info,other_hint_info,meta_info,other_meta_info,edge_map,point_map):

    new_all_edge_map={}
    other_all_edge_map=other_hint_info[0]
 
    for seg,dic in other_all_edge_map.items():
        s=edge_map[seg]
        new_dic={}
        for ty,d_s in dic.items():
            if isinstance(d_s,list):
                new_d_s=[]
                for d_t in d_s:
                    if len(d_t)==4:
                        if isinstance(d_t[0],DDimension) and d_t[0].dimtype!=0:
                            content=f"值：{d_t[0].text}"
                        elif isinstance(d_t[0],DDimension):
                            content=f"{d_t[1]}"
                        else:
                            content=f"值：{d_t[0].content}"
                        sub_ty=d_t[1].split("：")[-1]
                        if isinstance(d_t[0],DDimension) and d_t[0].dimtype!=0:
                            new_d_s.append([d_t[0],f"值：{content}，参考边：{sub_ty}",edge_map[d_t[2]],(edge_map[d_t[2]],point_map[d_t[3][1]],point_map[d_t[3][2]])])
                        elif isinstance(d_t[0],DDimension):
                            new_d_s.append([d_t[0],f"{d_t[1]}",edge_map[d_t[2]]])
                        else:
                            new_d_s.append([d_t[0],f"值：{content}，参考边：{sub_ty}",edge_map[d_t[2]]])
                    elif len(d_t)==3:
                        if isinstance(d_t[0],DDimension) and d_t[0].dimtype!=0:
                            content=f"值：{d_t[0].text}"
                        elif isinstance(d_t[0],DDimension):
                            content=f"{d_t[1]}"
                        else:
                            content=f"值：{d_t[0].content}"
                        sub_ty=d_t[1].split("：")[-1]
                        if isinstance(d_t[0],DDimension) and d_t[0].dimtype!=0:
                            new_d_s.append([d_t[0],f"值：{content}，参考边：{sub_ty}",edge_map[d_t[2]]])
                        elif isinstance(d_t[0],DDimension):
                            new_d_s.append([d_t[0],f"{d_t[1]}",edge_map[d_t[2]]])
                        else:
                            new_d_s.append([d_t[0],f"值：{content}，参考边：{sub_ty}",edge_map[d_t[2]]])
                    else:
                        if isinstance(d_t[0],DDimension) and d_t[0].dimtype!=0:
                            content=f"值：{d_t[0].text}"
                        elif  isinstance(d_t[0],DDimension):
                            content=f"{d_t[1]}"
                        else:
                            content=f"值：{d_t[0].content}"
                        new_d_s.append([d_t[0],content])

                new_dic[ty]=new_d_s
            else:
                new_dic[ty]=d_s
        new_all_edge_map[s]=new_dic
    new_features=other_hint_info[2]
    new_feature_map={}
    for seg,feats in other_hint_info[3].items():
        new_feature_map[edge_map[seg]]=feats
    new_hint_info=(new_all_edge_map,hint_info[1],new_features,new_feature_map,hint_info[4],hint_info[5],hint_info[6],hint_info[7],hint_info[8])

    new_meta_info=(meta_info[0],meta_info[1],meta_info[2],other_meta_info[3],meta_info[4],True)
    return new_hint_info,new_meta_info
def find_bkcode(texts):
    bk_code_pos={}
    for t in texts:
        content=t.content.strip()
        pattern_bk = r"BK(?P<bk_code>\d+)"
        if re.fullmatch(pattern_bk, content) and t.color==3:
            bk_code_pos[content]=t.insert
    return bk_code_pos
def calculate_codemap(edges_infos,poly_centroids,hint_infos,meta_infos,bk_code_pos):
    code_map={}
    for bk_code,pos in bk_code_pos.items():
        distance=float("inf")
        idx=None
        for i in range(len(meta_infos)):
            edges_info,poly_centroid,hint_info,meta_info=edges_infos[i],poly_centroids[i],hint_infos[i],meta_infos[i]
            is_standard_elbow,bracket_parameter,strengthen_parameter,has_hint,is_fb,is_diff=meta_info
        
            if has_hint==False or is_standard_elbow==False:
                continue
                
            dis=DSegment(DPoint(poly_centroid[0],poly_centroid[1]),pos).length()
            if distance>dis:
                distance=dis
                idx=i
        if idx is not None and distance<4000:
            code_map[bk_code]=(edges_infos[idx],poly_centroids[idx],hint_infos[idx],meta_infos[idx])
    return code_map
def hint_search_step(edges_infos,poly_centroids,hint_infos,meta_infos,code_map):
    new_edges_infos,new_poly_centroids,new_hint_infos,new_meta_infos=[],[],[],[]
    for i in range(len(meta_infos)):
        edges_info,poly_centroid,hint_info,meta_info=edges_infos[i],poly_centroids[i],hint_infos[i],meta_infos[i]
        is_standard_elbow,bracket_parameter,strengthen_parameter,has_hint,is_fb,is_diff=meta_info
  

        if  is_standard_elbow:
            tis=hint_info[7]
            has_code=False
            code=None
            for t_t in tis:
                if has_code:
                    break
                content=t_t[0].content.strip()
                if t_t[2]["Type"]=="BK":
                    has_code=True
                    code=t_t[2]["Typical Section Code"]
                    break   
            if has_code and code is not None:
                code="BK"+code
                if code in code_map:
                    info=code_map[code]
                    other_edges_info,other_poly_centroid,other_hint_info,other_meta_info=info
                    flag,edge_map,point_map=is_similar(edges_info,other_edges_info,hint_info[1],other_hint_info[1])
                    if flag:
                        new_hint_info,new_meta_info=calculate_new_hint_info_bk(edges_info,other_edges_info,hint_info,other_hint_info,meta_info,other_meta_info,edge_map,point_map)
                        
                        new_edges_infos.append(edges_info)
                        new_poly_centroids.append(poly_centroid)
                        new_hint_infos.append(new_hint_info)
                        new_meta_infos.append(new_meta_info)
                        continue     
            

            new_edges_infos.append(edges_info)
            new_poly_centroids.append(poly_centroid)
            new_hint_infos.append(hint_info)
            new_meta_infos.append(meta_info)
        else:
            new_edges_infos.append(edges_info)
            new_poly_centroids.append(poly_centroid)
            new_hint_infos.append(hint_info)
            new_meta_infos.append(meta_info)
    return new_edges_infos,new_poly_centroids,new_hint_infos,new_meta_infos

def diffusion_step(edges_infos,poly_centroids,hint_infos,meta_infos):
    new_edges_infos,new_poly_centroids,new_hint_infos,new_meta_infos=[],[],[],[]
    for i in range(len(meta_infos)):
        edges_info,poly_centroid,hint_info,meta_info=edges_infos[i],poly_centroids[i],hint_infos[i],meta_infos[i]
        is_standard_elbow,bracket_parameter,strengthen_parameter,has_hint,is_fb,is_diff=meta_info
        
        if has_hint==False and is_standard_elbow:
            distance=float('inf')
            info_=None
            for j in range(len(meta_infos)):
                if j!=i:
                    other_poly_centroid=poly_centroids[j]
                    other_hint_info=hint_infos[j]
                    other_is_standard_elbow,other_bracket_parameter,other_strengthen_parameter,other_has_hint,other_is_fb,other_is_diff=meta_infos[j]
                    if other_is_standard_elbow  and other_has_hint==True and DSegment(DPoint(poly_centroid[0],poly_centroid[1]),DPoint(other_poly_centroid[0],other_poly_centroid[1])).length()<100000 :


                        flag,edge_map,point_map=is_similar(edges_info,edges_infos[j],hint_info[1],other_hint_info[1])
                        if flag:
                            dis=DSegment(DPoint(poly_centroid[0],poly_centroid[1]),DPoint(other_poly_centroid[0],other_poly_centroid[1])).length()
                            if dis<distance:
                                distance=dis
                                info_=[edges_info,edges_infos[j],hint_info,hint_infos[j],meta_info,meta_infos[j],edge_map,point_map] 

            
            if info_ is not None:

                            
                new_hint_info,new_meta_info=calculate_new_hint_info(info_[0],info_[1],info_[2],info_[3],info_[4],info_[5],info_[6],info_[7])
                
                new_edges_infos.append(edges_info)
                new_poly_centroids.append(poly_centroid)
                new_hint_infos.append(new_hint_info)
                new_meta_infos.append(new_meta_info)
        
            else:
                new_edges_infos.append(edges_info)
                new_poly_centroids.append(poly_centroid)
                new_hint_infos.append(hint_info)
                new_meta_infos.append(meta_info)
        else:
            new_edges_infos.append(edges_info)
            new_poly_centroids.append(poly_centroid)
            new_hint_infos.append(hint_info)
            new_meta_infos.append(meta_info)
    return new_edges_infos,new_poly_centroids,new_hint_infos,new_meta_infos
def find_intersection(p1, p2, p3, p4):
    """
    计算两条线段的交点，包括延长线上的交点
    
    参数:
    p1, p2: 第一条线段的两个端点 (x, y)
    p3, p4: 第二条线段的两个端点 (x, y)
    
    返回:
    (x, y): 交点坐标
    None: 如果两条线段平行或重合
    """
    # 解线性方程组求参数
    # 第一条线段参数方程: p1 + t(p2-p1)
    # 第二条线段参数方程: p3 + u(p4-p3)
    
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    # 计算分母
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    
    # 如果分母为0，说明线段平行或重合
    if denom == 0:
        return None
    
    # 计算参数t和u
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    
    # 计算交点坐标
    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)
    
    return DPoint(x, y)

def calculate_dimesion_pos(d,constraint_edges):
    if  d.dimtype==32 or d.dimtype==33 or d.dimtype==161 or d.dimtype==160:
        l0=p_minus(d.defpoints[0],d.defpoints[2])
        l1=p_minus(d.defpoints[1],d.defpoints[2])
        d10=l0.x*l1.x+l0.y*l1.y
        d00=l0.x*l0.x+l0.y*l0.y
        if d00 <1e-4:
            x=d.defpoints[1]
        else:
            x=p_minus(p_add(d.defpoints[1],l0),p_mul(l0,d10/d00))
        d1,d2,d3,d4=d.defpoints[0], x,d.defpoints[1],d.defpoints[2]
        d1,d2,d3,d4=compute_accurate_position(d1,d2,d3,d4,constraint_edges)
        return [d3,d4]
    elif d.dimtype==34 or d.dimtype==162:
        s1=DSegment(d.defpoints[3],d.defpoints[0])
        s2=DSegment(d.defpoints[2],d.defpoints[1])
        
        inter=segment_intersection_line(s1.start_point,s1.end_point,s2.start_point,s2.end_point)
        pos=d.defpoints[4]
        r=DSegment(pos,inter).length()
        v1=DPoint((s1.end_point.x-s1.start_point.x)/s1.length()*(r+5),(s1.end_point.y-s1.start_point.y)/s1.length()*(r+5))
        v2=DPoint((s2.end_point.x-s2.start_point.x)/s2.length()*(r+5),(s2.end_point.y-s2.start_point.y)/s2.length()*(r+5))
        d1=DPoint(v1.x+inter.x,v1.y+inter.y)
        d2=DPoint(-v1.x+inter.x,-v1.y+inter.y)
        d3=DPoint(v2.x+inter.x,v2.y+inter.y)
        d4=DPoint(-v2.x+inter.x,-v2.y+inter.y)
        p1=None
        p2=None
        if DSegment(pos,d1).length()<DSegment(pos,d2).length():
            p1=d1
        else:
            p1=d2
        if DSegment(pos,d3).length()<DSegment(pos,d4).length():
            p2=d3
        else:
            p2=d4
        return [p1,p2,inter]
def draw_segment(segment,ax):
    if segment.isCornerhole:
        color = "#FF0000"
    elif segment.isConstraint:
        color = "#00FF00"
        if segment.isPart:
            color="#00AA00"
    else:
        color = "#0000FF"
        if segment.isPart:
            color="#000000"
    if isinstance(segment.ref, DArc):
        # if segment.ref.end_angle < segment.ref.start_angle:
        #     end_angle = segment.ref.end_angle + 360
        # else:
        #     end_angle = segment.ref.end_angle
        # start_angle=segment.ref.start_angle
        # if end_angle-start_angle>180.5:
        end_angle,start_angle=segment.ref.end_angle ,segment.ref.start_angle
        delta=end_angle-start_angle
        while delta<0:
            end_angle+=360
            delta+=360
        while delta>360:
            end_angle-=360
            delta-=360

        if delta>180:
            start_angle,end_angle=end_angle,start_angle
            end_angle+=360

        theta = np.linspace(np.radians(start_angle), np.radians(end_angle), 100)
        x_arc = segment.ref.center.x + segment.ref.radius * np.cos(theta)
        y_arc = segment.ref.center.y + segment.ref.radius * np.sin(theta)
        # p=transform_point_(DPoint(x_arc,y_arc),segment.ref.meta)
        ax.plot(x_arc, y_arc, color, lw=2)
    else:
        x_values = [segment.start_point.x, segment.end_point.x]
        y_values = [segment.start_point.y, segment.end_point.y]
        ax.plot(x_values, y_values, color, lw=2)
    return []
def plot_info_poly_std(constraint_edges,ori_edge_map,template_map,path):
    fig, ax = plt.subplots()
    free_idx=1
   
    while f'free{free_idx}' in template_map:
        if len(template_map[f'free{free_idx}'])==0:
            free_idx+=1
            continue
        seg=template_map[f'free{free_idx}'][0]
        draw_segment(seg,ax)
        mid=seg.mid_point()
        ax.text(mid.x, mid.y, free_idx,rotation=0,color="#0000FF", fontsize=25)
        free_idx+=1
        for ty,ds in ori_edge_map[seg].items():
            if isinstance(ds,list):
                for d_t in ds:
                    d=d_t[0]
                    if isinstance(d,DDimension):
                        if len(d_t)==2:
                            ps=calculate_dimesion_pos(d,constraint_edges)
                            if ps is None or len(ps)==0:
                                continue
                            elif len(ps)==2:
                                v1,v2=ps
                                ax.plot(v1.x,v1.y,'g.')
                                ax.plot(v2.x,v2.y,'g.')
                                q=DSegment(v1,v2).mid_point()
                                s=DSegment(v1,v2)
                                ax.text(q.x, q.y, f"{d.text}({free_idx-1})",rotation=0,color="#0000FF", fontsize=15)
                                ax.plot([s.start_point.x,s.end_point.x], [s.start_point.y,s.end_point.y], color="#FF0000", lw=2,linestyle='--')
                            elif len(ps)==3:
                                p1,p2,inter=ps
                                ax.plot(p1.x,p1.y,'g.')
                                ax.plot(p2.x,p2.y,'g.')
                                ax.plot(inter.x,inter.y,'g.')
                                ax.text(inter.x, inter.y, f"{d.text}({free_idx-1})",rotation=0,color="#0000FF", fontsize=15)
                                s=DSegment(p1,inter)
                                ax.plot([s.start_point.x,s.end_point.x], [s.start_point.y,s.end_point.y], color="#FF0000", lw=2,linestyle='--')
                                s=DSegment(p2,inter)
                                ax.plot([s.start_point.x,s.end_point.x], [s.start_point.y,s.end_point.y], color="#FF0000", lw=2,linestyle='--')

                        elif len(d_t)==3:
                            seg_=d_t[-1]
                            s_p=seg_.mid_point()
                            e_p=seg.mid_point()
                            # ax.plot([seg_.start_point.x,seg_.end_point.x], [seg_.start_point.y,seg_.end_point.y], color="#000000", lw=2,linestyle='--')
                            ax.plot([s_p.x,e_p.x], [s_p.y,e_p.y], color="#000000", lw=2,linestyle='--')
                            ps=calculate_dimesion_pos(d,constraint_edges)
                            if ps is None or  len(ps)==0:
                                continue
                            elif len(ps)==2:
                                v1,v2=ps
                                q=DSegment(v1,v2).mid_point()
                                s=DSegment(v1,v2)
                                ax.plot(v1.x,v1.y,'g.')
                                ax.plot(v2.x,v2.y,'g.')
                                ax.text(q.x, q.y, f"{d.text}({free_idx-1})",rotation=0,color="#0000FF", fontsize=15)
                                ax.plot([s.start_point.x,s.end_point.x], [s.start_point.y,s.end_point.y], color="#FF0000", lw=2,linestyle='--')
                            elif len(ps)==3:
                                p1,p2,inter=ps
                                ax.plot(p1.x,p1.y,'g.')
                                ax.plot(p2.x,p2.y,'g.')
                                ax.plot(inter.x,inter.y,'g.')
                                ax.text(inter.x, inter.y, f"{d.text}({free_idx-1})",rotation=0,color="#0000FF", fontsize=15)
                                s=DSegment(p1,inter)
                                ax.plot([s.start_point.x,s.end_point.x], [s.start_point.y,s.end_point.y], color="#FF0000", lw=2,linestyle='--')
                                s=DSegment(p2,inter)
                                ax.plot([s.start_point.x,s.end_point.x], [s.start_point.y,s.end_point.y], color="#FF0000", lw=2,linestyle='--')
                    elif isinstance(d,DText):
                        ax.text(d.insert.x, d.insert.y,f"{d.content}({free_idx-1})",rotation=0,color="#0000FF", fontsize=15)
    
    constarint_idx=1
    cornerhole_idx=1
    template_edges=[]
    # print("============")
    while True:
        # print(constarint_idx,cornerhole_idx)
        if f'constraint{constarint_idx}' in template_map:
            template_edges.append(template_map[f'constraint{constarint_idx}'])
            constarint_idx+=1
        else:
            break
        if f'cornerhole{cornerhole_idx}' in template_map:  
            template_edges.append(template_map[f'cornerhole{cornerhole_idx}'])
            cornerhole_idx+=1
    k=1
    constarint_idx=0
    cornerhole_idx=0

    for i,edge in enumerate(template_edges):


        if len(edge)==0:
            k+=1
            cornerhole_idx+=1
        elif edge[0].isCornerhole:
            k+=1
            corner_hole_start_edge=edge[0]
            cornerhole_idx+=1
            mid=edge[0].mid_point()
            ax.text(mid.x, mid.y, cornerhole_idx,rotation=0,color="#FF0000", fontsize=25)
            for seg in edge:
                draw_segment(seg,ax)
        else:
          
            k+=1
            constarint_idx+=1
            mid=edge[0].mid_point()
            ax.text(mid.x, mid.y, constarint_idx,rotation=0,color="#00FF00", fontsize=25)
            next_cons_edge=[]
            last_cons_edge=[]
            index=i+1
            while index<len(template_edges):
                next_e=template_edges[index]
                if not(len(next_e)==0 or next_e[0].isCornerhole):
                    next_cons_edge=next_e
                    break
                index+=1
            index=i-1
            while index>=0:
                last_e=template_edges[index]
                if not(len(last_e)==0 or last_e[0].isCornerhole):
                    last_cons_edge=last_e
                    break
                index-=1
            next_seg=next_cons_edge[0] if len(next_cons_edge)>0 else None
            last_seg=last_cons_edge[0] if len(last_cons_edge)>0 else None
            if last_seg is not None:
                inter1=find_intersection(last_seg.start_point,last_seg.end_point,seg.start_point,seg.end_point)
            else:
                inter1=None
            if next_seg is not None:
                inter2=find_intersection(next_seg.start_point,next_seg.end_point,seg.start_point,seg.end_point)
            else:
                inter2=None
            if inter1 is not None and inter2 is not None:
                new_seg=DSegment(inter1,inter2)
            elif inter1 is not None:
                if DSegment(inter1,seg.start_point).length()<DSegment(inter1,seg.end_point).length():
                    new_seg=DSegment(inter1,seg.end_point)
                else:
                    new_seg=DSegment(inter1,seg.start_point)
            elif inter2 is not None:
                if DSegment(inter2,seg.start_point).length()<DSegment(inter2,seg.end_point).length():
                    new_seg=DSegment(inter2,seg.end_point)
                else:
                    new_seg=DSegment(inter2,seg.start_point)
            else:
                new_seg=seg
            correct_edge=[new_seg]
            for seg in edge:
                draw_segment(seg,ax)
                for ty,ds in ori_edge_map[seg].items():
                    if isinstance(ds,list):
                        for d_t in ds:
                            d=d_t[0]
                            if isinstance(d,DDimension):
                                if len(d_t)==2:
                                    ps=calculate_dimesion_pos(d,constraint_edges)
                                    if len(ps)==0:
                                        continue
                                    elif len(ps)==2:
                                        v1,v2=ps
                                        ax.plot(v1.x,v1.y,'g.')
                                        ax.plot(v2.x,v2.y,'g.')
                                        q=DSegment(v1,v2).mid_point()
                                        s=DSegment(v1,v2)
                                        ax.text(q.x, q.y, f"{d.text}({constarint_idx})",rotation=0,color="#00FF00", fontsize=15)
                                        ax.plot([s.start_point.x,s.end_point.x], [s.start_point.y,s.end_point.y], color="#FF0000", lw=2,linestyle='--')
                                    elif len(ps)==3:
                                        p1,p2,inter=ps
                                        ax.plot(p1.x,p1.y,'g.')
                                        ax.plot(p2.x,p2.y,'g.')
                                        ax.plot(inter.x,inter.y,'g.')
                                        ax.text(inter.x, inter.y, f"{d.text}({constarint_idx})",rotation=0,color="#00FF00", fontsize=15)
                                        s=DSegment(p1,inter)
                                        ax.plot([s.start_point.x,s.end_point.x], [s.start_point.y,s.end_point.y], color="#FF0000", lw=2,linestyle='--')
                                        s=DSegment(p2,inter)
                                        ax.plot([s.start_point.x,s.end_point.x], [s.start_point.y,s.end_point.y], color="#FF0000", lw=2,linestyle='--')

                                elif len(d_t)==3:
                                    seg_=d_t[-1]
                                    s_p=seg_.mid_point()
                                    e_p=seg.mid_point()
                                    # ax.plot([seg_.start_point.x,seg_.end_point.x], [seg_.start_point.y,seg_.end_point.y], color="#000000", lw=2,linestyle='--')
                                    ax.plot([s_p.x,e_p.x], [s_p.y,e_p.y], color="#000000", lw=2,linestyle='--')
                                    ps=calculate_dimesion_pos(d,constraint_edges)
                                    if len(ps)==0:
                                        continue
                                    elif len(ps)==2:
                                        v1,v2=ps
                                        q=DSegment(v1,v2).mid_point()
                                        s=DSegment(v1,v2)
                                        ax.plot(v1.x,v1.y,'g.')
                                        ax.plot(v2.x,v2.y,'g.')
                                        ax.text(q.x, q.y, f"{d.text}({constarint_idx})",rotation=0,color="#00FF00", fontsize=15)
                                        ax.plot([s.start_point.x,s.end_point.x], [s.start_point.y,s.end_point.y], color="#FF0000", lw=2,linestyle='--')
                                    elif len(ps)==3:
                                        p1,p2,inter=ps
                                        ax.plot(p1.x,p1.y,'g.')
                                        ax.plot(p2.x,p2.y,'g.')
                                        ax.plot(inter.x,inter.y,'g.')
                                        ax.text(inter.x, inter.y, f"{d.text}({constarint_idx})",rotation=0,color="#00FF00", fontsize=15)
                                        s=DSegment(p1,inter)
                                        ax.plot([s.start_point.x,s.end_point.x], [s.start_point.y,s.end_point.y], color="#FF0000", lw=2,linestyle='--')
                                        s=DSegment(p2,inter)
                                        ax.plot([s.start_point.x,s.end_point.x], [s.start_point.y,s.end_point.y], color="#FF0000", lw=2,linestyle='--')


                            elif isinstance(d,DText):
                                ax.text(d.insert.x, d.insert.y, f"{d.content}({constarint_idx})",rotation=0,color="#00FF00", fontsize=15)



   








    # t_map={}
    # for t_t in texts:
    #     t=t_t[0]
    #     pos=t_t[1]
    #     content=t.content
    #     if t_t[2]["Type"] in ["B","FB","BK","FL"] and not point_is_inside(pos,polygon):
    #         continue
    #     if t_t[2]["Type"]=="None":
    #         continue
    #     if pos not in t_map:
    #         t_map[pos]=[]
    #         t_map[pos].append(content)
    #     else:
    #         t_map[pos].append(content)
    # for pos,cs in t_map.items():
    #     ax.text(pos.x, pos.y, cs, fontsize=12, color='blue', rotation=0)
    # for d_t in dimensions:
    #     d=d_t[0]
    #     pos=d_t[1]
    #     content=d.text
    #     # for i,p in enumerate(d.defpoints):
    #     #     if p!=DPoint(0,0):
    #     #         ax.text(p.x, p.y, str(i),color="#EEEE00", fontsize=15)
    #     #         ax.plot(p.x, p.y, 'b.')
    #     if  d.dimtype==32 or d.dimtype==33 or d.dimtype==161 or d.dimtype==160:
    #         l0=p_minus(d.defpoints[0],d.defpoints[2])
    #         l1=p_minus(d.defpoints[1],d.defpoints[2])
    #         d10=l0.x*l1.x+l0.y*l1.y
    #         d00=l0.x*l0.x+l0.y*l0.y
    #         if d00 <1e-4:
    #             x=d.defpoints[1]
    #         else:
    #             x=p_minus(p_add(d.defpoints[1],l0),p_mul(l0,d10/d00))
    #         d1,d2,d3,d4=d.defpoints[0], x,d.defpoints[1],d.defpoints[2]
          
    #         ss=[DSegment(d1,d4),DSegment(d4,d3),DSegment(d3,d2)]
    #         sss=[DSegment(d2,d1)]
    #         ss=expandFixedLengthGeo(ss,25,True)
    #         sss=expandFixedLengthGeo(sss,100,True)
    #         q=sss[0].end_point
    #         sss=expandFixedLengthGeo(sss,100,False)
    #         for s in ss:
    #             ax.plot([s.start_point.x,s.end_point.x], [s.start_point.y,s.end_point.y], color="#FF0000", lw=2,linestyle='--')
    #         for s in sss:
    #             ax.plot([s.start_point.x,s.end_point.x], [s.start_point.y,s.end_point.y], color="#FF0000", lw=2,linestyle='--')
    #         ax.arrow(sss[0].end_point.x, sss[0].end_point.y, d1.x-sss[0].end_point.x, d1.y-sss[0].end_point.y, head_width=20, head_length=20, fc='red', ec='red')
    #         ax.arrow(sss[0].start_point.x, sss[0].start_point.y, d2.x-sss[0].start_point.x, d2.y-sss[0].start_point.y, head_width=20, head_length=20, fc='red', ec='red')
    #         perp_vx, perp_vy = sss[0].start_point.x - sss[0].end_point.x, sss[0].start_point.y-sss[0].end_point.y
    #         rotation_angle = np.arctan2(-perp_vy, -perp_vx) * (180 / np.pi)
    #         ax.text(q.x, q.y, d.text,rotation=rotation_angle,color="#EEC933", fontsize=15)
            


        
    #     elif d.dimtype==37:
    #         a,b,o=d.defpoints[1],d.defpoints[2],d.defpoints[3]
    #         r=DSegment(d.defpoints[0],o).length()
    #         ra=DSegment(a,o).length()
    #         rb=DSegment(b,o).length()
    #         oa_=p_mul(p_minus(a,o),r/ra)
    #         ob_=p_mul(p_minus(b,o),r/rb)
    #         ao_=p_mul(oa_,-1)
    #         bo_=p_mul(ob_,-1)
    #         a_= p_add(o, oa_)
    #         b_ = p_add(o, ob_)
    #         ia_=p_add(o,ao_)
    #         ib_=p_add(o,bo_)
    #         delta=p_mul(DPoint(oa_.y,-oa_.x),3)
    #         sp=p_add(delta,a_)
    #         ax.arrow(ia_.x, ia_.y, a_.x-ia_.x,a_.y-ia_.y, head_width=20, head_length=20, fc='red', ec='red',linestyle='--')
    #         ax.arrow(ib_.x, ib_.y, b_.x-ib_.x,b_.y-ib_.y, head_width=20, head_length=20, fc='red', ec='red',linestyle='--')
    #         ax.plot([sp.x,a_.x], [sp.y,a_.y], color="#FF0000", lw=2,linestyle='--')
    #         q=p_mul(p_add(a_,sp),0.5)
    #         rotation_angle = np.arctan2(-delta.y, delta.x) * (180 / np.pi)
    #         ax.text(q.x, q.y, d.text,rotation=rotation_angle,color="#EEC933", fontsize=15)
    #     elif d.dimtype==34 or d.dimtype==162:
                
    #             s1=DSegment(d.defpoints[3],d.defpoints[0])
    #             s2=DSegment(d.defpoints[2],d.defpoints[1])
                
    #             inter=segment_intersection_line_(s1.start_point,s1.end_point,s2.start_point,s2.end_point)
    #             pos=d.defpoints[4]
    #             r=DSegment(pos,inter).length()
    #             v1=DPoint((s1.end_point.x-s1.start_point.x)/s1.length()*(r+5),(s1.end_point.y-s1.start_point.y)/s1.length()*(r+5))
    #             v2=DPoint((s2.end_point.x-s2.start_point.x)/s2.length()*(r+5),(s2.end_point.y-s2.start_point.y)/s2.length()*(r+5))
    #             d1=DPoint(v1.x+inter.x,v1.y+inter.y)
    #             d2=DPoint(-v1.x+inter.x,-v1.y+inter.y)
    #             d3=DPoint(v2.x+inter.x,v2.y+inter.y)
    #             d4=DPoint(-v2.x+inter.x,-v2.y+inter.y)
    #             p1=None
    #             p2=None
    #             if DSegment(pos,d1).length()<DSegment(pos,d2).length():
    #                 p1=d1
    #             else:
    #                 p1=d2
    #             if DSegment(pos,d3).length()<DSegment(pos,d4).length():
    #                 p2=d3
    #             else:
    #                 p2=d4

                
    #             # if d.dimtype==34:
    #             #     for i,p in enumerate([d1,d2,d3,d4]):
    #             #         if p!=DPoint(0,0):
    #             #             plt.text(p.x, p.y, str(i+1),color="#EE0000", fontsize=15)
    #             #             plt.plot(p.x, p.y, 'b.')
    #             ss1=DSegment(inter,p1)
    #             ss2=DSegment(inter,p2)

    #             ax.arrow(ss1.start_point.x, ss1.start_point.y, ss1.end_point.x-ss1.start_point.x,ss1.end_point.y-ss1.start_point.y, head_width=20, head_length=20, fc='red', ec='red')
    #             ax.arrow(ss2.start_point.x, ss2.start_point.y, ss2.end_point.x-ss2.start_point.x,ss2.end_point.y-ss2.start_point.y, head_width=20, head_length=20, fc='red', ec='red')
    #             ax.text(pos.x, pos.y, d.text,rotation=0,color="#EEC933", fontsize=15)
    #             ax.plot(inter.x,inter.y,'g.')
    #             ax.plot([p1.x,p2.x], [p1.y,p2.y], color="#FF0000", lw=2,linestyle='--')
        
    #     elif d.dimtype==163:
    #         a,b=d.defpoints[0],d.defpoints[3]
    #         o=p_mul(p_add(a,b),0.5)
    #         ss=[DSegment(a,b)]
    #         ss=expandFixedLengthGeo(ss,100)
    #         ss=expandFixedLengthGeo(ss,100,False)
    #         a_,b_=ss[0].start_point,ss[0].end_point
    #         ax.arrow(a_.x, a_.y, a.x-a_.x,a.y-a_.y, head_width=20, head_length=20, fc='red', ec='red')
    #         ax.arrow(b_.x, b_.y, b.x-b_.x,b.y-b_.y, head_width=20, head_length=20, fc='red', ec='red')
    #         q=p_add(p_mul(b_,0.7),p_mul(b,0.3))
    #         ab=p_minus(b,a)
    #         rotation_angle = np.arctan2(ab.y, -ab.x) * (180 / np.pi)
    #         ax.text(q.x, q.y, d.text,rotation=rotation_angle,color="#EEC933", fontsize=15)
            
            

    ax.set_aspect('equal', 'box')
    plt.savefig(path)
    plt.close('all')
    plt.close()

def check_one_class(classification_res,template_map,free_edges,constraint_edges,edges,poly_refs,poly,all_edge_map, ref_map):
    polygon=computePolygon(poly)
    if "DPK(" in classification_res or "DPV(" in classification_res  or "DPKN(" in classification_res:
        if "free2" not in template_map or len(template_map["free2"])==0 or (not isinstance(template_map["free2"][0].ref,DArc)):
            return False
        center=template_map["free2"][0].ref.center 
        center=Point(center.x,center.y)
        if polygon.contains(center):
            return False

    if "BR(" in classification_res:
        for c_edge in constraint_edges:
            for seg in c_edge:
                actual_segs = []
                if seg in ref_map:
                    actual_segs.extend(ref_map[seg])
                else:
                    actual_segs.append(seg)
                for se in actual_segs:
                    if not se.isCornerhole and isinstance(se.ref, DArc):
                        return False





    if "DPKN-2(" in classification_res:
        for edge in edges:
            for seg in edge:
                if seg in all_edge_map and "与约束边及其平行线夹角标注" in all_edge_map[seg]:
                    ds=all_edge_map[seg]["与约束边及其平行线夹角标注"]
                    if len(ds)>0:
                        return False
    
    if "BR-2(" in classification_res:
        if "free2" not in template_map or len(template_map["free2"])==0 or (not isinstance(template_map["free2"][0].ref,DArc)):
            return False
        mid=DSegment(template_map["free2"][0].ref.start_point,template_map["free2"][0].ref.end_point).mid_point() 
        mid=Point(mid.x,mid.y)
        if polygon.contains(mid):
            return False
    if "DMA(" in classification_res:
        if "free3" not in template_map or len(template_map["free3"])==0 or (not isinstance(template_map["free3"][0].ref,DArc)):
            return False
        center=template_map["free3"][0].ref.center 
        center=Point(center.x,center.y)
        if polygon.contains(center):
            return False




    return True


def check_one_class_ustd(polyline_handles,classification_res,free_edges,constraint_edges,edges,poly_refs,poly,all_edge_map):
    #二维多段线 
    for s in free_edges[0]:
        if s.ref is not None and s.ref.handle is not None and s.ref.handle in polyline_handles:
            return False
    return True 
def outputInfo(index,edges_info,poly_centroid,hint_info,meta_info,segmentation_config,poly,polyline_handles):
    is_standard_elbow,bracket_parameter,strengthen_parameter,has_hint,is_fb,is_diff=meta_info
    free_edges,constraint_edges,edges,poly_refs,ref_map=edges_info
    all_edge_map,edge_types,features,feature_map,constraint_edge_no,free_edge_no,all_anno,tis,ds=hint_info
    json_data={}
    json_name = os.path.join(segmentation_config.poly_info_dir, f'标准肘板/info{index}.json')
    # 获取剖图信息

    try:
        with open(segmentation_config.multi_json_path, "r", encoding="utf-8") as f:
            multi_data = json.load(f)
    except FileNotFoundError:
        print(f"文件未找到：{segmentation_config.multi_json_path}")
        multi_data = None
    except json.JSONDecodeError:
        print(f"文件格式错误，非合法JSON：{segmentation_config.multi_json_path}")
        multi_data = None
    except Exception as e:
        print(f"加载过程中出现其他错误：{e}")
        multi_data = None

    poumian_name = None
    poumian_copy_time = 1
    if multi_data is not None:
        multi_poly_dic = multi_data[-1]["子图调用次数"]
        poumian_name, poumian_copy_time = get_copy_info(poly_centroid, multi_poly_dic)
        

    #handle center classification type edge  value AorB  is_diff
    size_hints=[]
    #handle center classification info is_diff
    meta_hints=[]
    if is_standard_elbow==False:
        thickness=0
        if meta_info[1] is not None and meta_info[1]["Thickness"] is not None:
            thickness=meta_info[1]["Thickness"]
        classification_res=poly_classifier_ustd(free_edges, edges, segmentation_config.unstandard_type_path, edge_types, thickness, feature_map)
        if classification_res is None or classification_res=="Unclassified":
            return poly_refs,"Unclassified",[],[]
        classification_res=f"ustd:{classification_res}"
        if check_one_class_ustd(polyline_handles,classification_res,free_edges,constraint_edges,edges,poly_refs,poly,all_edge_map)==False:
            return poly_refs,"Unclassified",[],[]
        template_cons_edges=[]
        cornorhole_edge_no={}
        i_=1
        for i,edge in enumerate(edges):
    

            if len(edge)==0:
                continue

            elif edge[0].isCornerhole:
                for s in edge[0]:
                    cornorhole_edge_no[s]=i_
                i_+=1
                continue
            else:
                for seg in edge:
                    template_cons_edges.append(seg)

        order_set=set()
       
        for i in range(len(template_cons_edges)-1):
            current_edge=template_cons_edges[i]
            next_edge=template_cons_edges[i+1]
            pairs=[(current_edge.start_point,next_edge.start_point),(current_edge.start_point,next_edge.end_point),(current_edge.end_point,next_edge.start_point),(current_edge.end_point,next_edge.end_point)]
            distance=float('inf')
            min_pair=None
            for pair in pairs:
                p1,p2=pair
                dis=DSegment(p1,p2).length()
                if dis< distance:
                    distance=dis
                    min_pair=pair
            p2,p=min_pair
            if  p2==current_edge.start_point:
                p1=current_edge.end_point
            else:
                p1=current_edge.start_point
            order_set.add((current_edge,p1))
            if i==len(template_cons_edges)-2:
                order_set.add((next_edge,p))
        new_all_edge_map={}
        for s,dic in all_edge_map.items():
            new_dic={}
            for ty,ds in dic.items():
                if isinstance(ds,list):
                    new_ds=[]
                    for d_t in ds:
                        if len(d_t)==4:
                            d,des,seg=d_t[0],d_t[1],d_t[2] 
                            _,p1,p2=d_t[3]
                            if seg in constraint_edge_no:
                                des+=str(constraint_edge_no[seg])
                            else:
                                des+=str(free_edge_no[seg])
                            order_ty=""
                            if (seg,p1) in order_set:
                                des+="，1号边"
                                order_ty="A"
                            else:
                                des+="，2号边"
                                order_ty="B"
                            new_ds.append([d,des])
                            edge_no=""
                            if s in free_edge_no:
                                edge_no=f"自由边{free_edge_no[s]}"
                            elif s in constraint_edge_no:
                                edge_no=f"约束边{constraint_edge_no[s]}"
                            ref_no="无"
                            if seg in free_edge_no:
                                ref_no=f"自由边{free_edge_no[seg]}"
                            elif seg in constraint_edge_no:
                                ref_no=f"约束边{constraint_edge_no[seg]}"
                            size_hints.append([index,d.handle,poly_centroid,classification_res,ty,edge_no,s.ref.handle,ref_no,seg.ref.handle,d.text,order_ty,is_diff])
                        elif len(d_t)==3:
                            d,des,seg=d_t 
                            if seg in constraint_edge_no:
                                des+=str(constraint_edge_no[seg])
                            else:
                                des+=str(free_edge_no[seg])
                            new_ds.append([d,des])
                            if d.dimtype!=0:
                                edge_no=""
                                if s in free_edge_no:
                                    edge_no=f"自由边{free_edge_no[s]}"
                                elif s in constraint_edge_no:
                                    edge_no=f"约束边{constraint_edge_no[s]}"
                                ref_no="无"
                                if seg in free_edge_no:
                                    ref_no=f"自由边{free_edge_no[seg]}"
                                elif seg in constraint_edge_no:
                                    ref_no=f"约束边{constraint_edge_no[seg]}"
                                size_hints.append([index,d.handle,poly_centroid,classification_res,ty,edge_no,s.ref.handle,ref_no,seg.ref.handle,d.text,"无",is_diff])
                        else:
                            new_ds.append(d_t)
                            if len(d_t)==2:
                                d=d_t[0]
                                if isinstance(d_t[0],DDimension):
                                    if d.dimtype!=0:
                                        edge_no=""
                                        if s in free_edge_no:
                                            edge_no=f"自由边{free_edge_no[s]}"
                                        elif s in constraint_edge_no:
                                            edge_no=f"约束边{constraint_edge_no[s]}"
                                        size_hints.append([index,d.handle,poly_centroid,classification_res,ty,edge_no,s.ref.handle,"无","无",d.text,"无",is_diff])
                                else:
                                    edge_no=""
                                    if s in free_edge_no:
                                        edge_no=f"自由边{free_edge_no[s]}"
                                    elif s in constraint_edge_no:
                                        edge_no=f"约束边{constraint_edge_no[s]}"
                                    else:
                                        edge_no=f"角隅孔{cornorhole_edge_no[s]}"
                                    size_hints.append([index,d.handle,poly_centroid,classification_res,ty,edge_no,s.ref.handle,"无","无",d.content,"无",is_diff])
                    new_dic[ty]=new_ds
                else:
                    new_dic[ty]=ds
            new_all_edge_map[s]=new_dic
        all_edge_map=new_all_edge_map


        for t_t in tis:
            content=t_t[0].content.strip()
            if t_t[2]["Type"]=="B" or t_t[2]["Type"]=="FB" or t_t[2]["Type"]=="FL":
                meta_hints.append([index,t_t[0].handle,poly_centroid,classification_res,content,False])
        file_path = os.path.join(segmentation_config.poly_info_dir, f'非标准肘板/info{index}.txt')
        clear_file(file_path)
        log_to_file(file_path, f"几何中心坐标：{poly_centroid}")

        json_data["几何中心坐标"]=[poly_centroid[0],poly_centroid[1]]
        json_data["主图标题"] = poumian_name
        json_data["调用总次数"] = poumian_copy_time
        log_to_file(file_path, f"边界数量（含自由边）：{len(constraint_edges) + len(free_edges)}")
        log_to_file(file_path, "边界信息（非自由边）：")
        json_data["非自由边轮廓"]=[]
        json_data["非自由边"]=[]
        constarint_idx=0
        cornerhole_idx=0
        k=1
        for i,edge in enumerate(edges):
            if (not edge[0].isConstraint) and (not edge[0].isCornerhole):
                continue
            log_to_file(file_path, f"边界颜色{k}: {edge[0].ref.color}")
            log_to_file(file_path, f"边界轮廓{k}({get_edge_des(edge)}): ")
            k+=1
            if edge[0].isCornerhole:
                json_data["非自由边轮廓"].append({"类型":"角隅孔","轮廓":["line" if isinstance(s.ref,DLine) else "arc" for s in edge]})
                json_data["非自由边"].append({"缺省":False,"颜色":edge[0].ref.color,"边":[]})
                corner_hole_start_edge=edge[0]
                log_to_file(file_path,f"    角隅孔{cornerhole_idx+1}({get_edge_des(edge)})")
                cornerhole_idx+=1
                if len(edge)>1 and is_vu(edge):
                    anno_des=""
                    anno_des2=""
                    for anno in all_edge_map[corner_hole_start_edge]["短边尺寸标注"]:
                        anno_des=f"{anno_des},{anno[1]}"
                    for anno in all_edge_map[corner_hole_start_edge]["尺寸参数"]:
                        anno_des2=f"{anno_des2},{anno[1]}"
                    des=f"短边是否平行于相邻边:{all_edge_map[corner_hole_start_edge]["短边是否平行于相邻边"]};\n\t\t\t短边尺寸标注:{anno_des}\n\t\t\t尺寸参数:{anno_des2}"
                    json_data["非自由边"][-1]["标注"]=all_edge_map[corner_hole_start_edge]
                    log_to_file(file_path,f"        VU孔标注:\n\t\t\t{des}")
                    for seg in edge:
                        if isinstance(seg.ref, DArc):
                            log_to_file(file_path, f"       起点：{seg.ref.start_point}、终点：{seg.ref.end_point}、圆心：{seg.ref.center}、半径：{seg.ref.radius}（圆弧）、句柄: {seg.ref.handle}")
                            anno_des=""
                            for anno in all_edge_map[seg]["半径尺寸标注"]:
                                anno_des=f"{anno_des},{anno[1]}"
                            des=f"半径尺寸标注:{anno_des}"
                            log_to_file(file_path, f"           标注:\n\t\t\t{des}")        
                            json_data["非自由边"][-1]["边"].append({
                                "几何":{
                                    "起点":[seg.ref.start_point.x,seg.ref.start_point.y],
                                    "终点":[seg.ref.end_point.x,seg.ref.end_point.y],
                                    "圆心":[seg.ref.center.x,seg.ref.center.y],
                                    "半径":seg.ref.radius
                                },
                                "句柄":seg.ref.handle,
                                "标注":all_edge_map[seg],
                                "类型":"arc"

                            })          
                        else:
                            
                            log_to_file(file_path, f"       起点：{seg.start_point}、终点{seg.end_point}（直线）、句柄: {seg.ref.handle}")
                            json_data["非自由边"][-1]["边"].append({
                                "几何":{
                                    "起点":[seg.ref.start_point.x,seg.ref.start_point.y],
                                    "终点":[seg.ref.end_point.x,seg.ref.end_point.y],
                                    "长度":seg.length()
                                },
                                "句柄":seg.ref.handle,
                                "标注":{},
                                "类型":"line"

                            })   
                else:

                    for seg in edge:
                        if isinstance(seg.ref, DArc):
                            json_data["非自由边"][-1]["边"].append({
                                "几何":{
                                    "起点":[seg.ref.start_point.x,seg.ref.start_point.y],
                                    "终点":[seg.ref.end_point.x,seg.ref.end_point.y],
                                    "圆心":[seg.ref.center.x,seg.ref.center.y],
                                    "半径":seg.ref.radius
                                },
                                "句柄":seg.ref.handle,
                                "标注":all_edge_map[seg],
                                "类型":"arc"

                            })     
                            log_to_file(file_path, f"       起点：{seg.ref.start_point}、终点：{seg.ref.end_point}、圆心：{seg.ref.center}、半径：{seg.ref.radius}（圆弧）、句柄: {seg.ref.handle}")
                            anno_des=""
                            for anno in all_edge_map[seg]["半径尺寸标注"]:
                                anno_des=f"{anno_des},{anno[1]}"
                            des=f"半径尺寸标注:{anno_des}"
                            log_to_file(file_path, f"           标注:\n\t\t\t{des}")                  
                        else:
                            json_data["非自由边"][-1]["边"].append({
                                "几何":{
                                    "起点":[seg.ref.start_point.x,seg.ref.start_point.y],
                                    "终点":[seg.ref.end_point.x,seg.ref.end_point.y],
                                    "长度":seg.length()
                                },
                                "句柄":seg.ref.handle,
                                "标注":{},
                                "类型":"line"

                            })   
                            log_to_file(file_path, f"       起点：{seg.start_point}、终点{seg.end_point}（直线）、句柄: {seg.ref.handle}")
            else:
                json_data["非自由边轮廓"].append({"类型":"约束边","轮廓":["line" if isinstance(s.ref,DLine) else "arc" for s in edge]})
                json_data["非自由边"].append({"缺省":False,"颜色":edge[0].ref.color,"边":[]})
                log_to_file(file_path,f"    约束边{constarint_idx+1}({get_edge_des(edge)})")
                constarint_idx+=1
                for seg in edge:
                    if isinstance(seg.ref, DArc):
                        json_data["非自由边"][-1]["边"].append({
                            "几何":{
                                "起点":[seg.ref.start_point.x,seg.ref.start_point.y],
                                "终点":[seg.ref.end_point.x,seg.ref.end_point.y],
                                "圆心":[seg.ref.center.x,seg.ref.center.y],
                                "半径":seg.ref.radius
                            },
                            "句柄":seg.ref.handle,
                            "标注":all_edge_map[seg],
                            "类型":"arc"

                        })  
                        log_to_file(file_path, f"       起点：{seg.ref.start_point}、终点：{seg.ref.end_point}、圆心：{seg.ref.center}、半径：{seg.ref.radius}（圆弧）、句柄: {seg.ref.handle}")
                    else:
                        json_data["非自由边"][-1]["边"].append({
                            "几何":{
                                "起点":[seg.ref.start_point.x,seg.ref.start_point.y],
                                "终点":[seg.ref.end_point.x,seg.ref.end_point.y],
                                "长度":seg.length()
                            },
                            "句柄":seg.ref.handle,
                            "标注":{},
                            "类型":"line"

                        })   
                        log_to_file(file_path, f"       起点：{seg.start_point}、终点{seg.end_point}（直线）、句柄: {seg.ref.handle}")
                        des=""
                        for ty,annos in all_edge_map[seg].items():
                            anno_des=""
                            for anno in annos:
                                anno_des=f"{anno_des},{anno[1]}"
                            des=f"{des}\n\t\t\t{ty}:{anno_des}"
                        log_to_file(file_path, f"           标注:{des}")

        # step8: 输出自由边信息
        log_to_file(file_path, f"自由边轮廓({get_free_edge_des(free_edges[0],edge_types)})：")
        free_idx=0
        json_data["自由边轮廓"]=[edge_types[s] for s in free_edges[0]]
        json_data["自由边"]=[]
        for seg in free_edges[0]:
            log_to_file(file_path, f"   自由边{free_idx+1}")
            log_to_file(file_path, f"        类型：{edge_types[seg]}")

            json_data["自由边"].append({"序号":free_idx+1,"类型":edge_types[seg],"缺省":False})
            free_idx+=1
            if edge_types[seg]=="arc":

                json_data["自由边"][-1]["几何"]={
                    "起点":[seg.ref.start_point.x,seg.ref.start_point.y],
                    "终点":[seg.ref.end_point.x,seg.ref.end_point.y],
                    "圆心":[seg.ref.center.x,seg.ref.center.y],
                    "半径":seg.ref.radius
                }
                json_data["自由边"][-1]["句柄"]=seg.ref.handle
                json_data["自由边"][-1]["标注"]=all_edge_map[seg]
                log_to_file(file_path, f"       起点：{seg.ref.start_point}、终点：{seg.ref.end_point}、圆心：{seg.ref.center}、半径：{seg.ref.radius}（圆弧）、句柄: {seg.ref.handle}")
                anno_des=""
                for anno in all_edge_map[seg]["半径尺寸标注"]:
                    anno_des=f"{anno_des},{anno[1]}"
                des=f"是否相切:{all_edge_map[seg]["是否相切"]}\n\t\t\t半径尺寸标注:{anno_des}\n\t\t\t圆心是否在趾端延长线上:{all_edge_map[seg]["圆心是否在趾端延长线上"]}"
            
                log_to_file(file_path, f"           标注:\n\t\t\t{des}")        
            elif edge_types[seg]=="line":
                json_data["自由边"][-1]["几何"]={
                    "起点":[seg.ref.start_point.x,seg.ref.start_point.y],
                    "终点":[seg.ref.end_point.x,seg.ref.end_point.y],
                    "长度":seg.length()
                }
                json_data["自由边"][-1]["句柄"]=seg.ref.handle
                json_data["自由边"][-1]["标注"]=all_edge_map[seg]
                log_to_file(file_path, f"       起点：{seg.start_point}、终点{seg.end_point}、长度：{seg.length()}（直线）、句柄: {seg.ref.handle}")
                des=f"是否与约束边平行:{all_edge_map[seg]["是否与约束边平行"]}\n\t\t\t是否与相邻约束边夹角为90度:{all_edge_map[seg]["是否与相邻约束边夹角为90度"]}"
                for ty,annos in all_edge_map[seg].items():
                    if ty=="是否与约束边平行" or ty=="是否与相邻约束边夹角为90度" :
                        continue
                    anno_des=""
                    for anno in annos:
                        anno_des=f"{anno_des},{anno[1]}"
                    des=f"{des};{ty}:{anno_des}"
                log_to_file(file_path, f"           标注:\n\t\t\t{des}")
            elif edge_types[seg]=="toe": 
                json_data["自由边"][-1]["几何"]={
                    "起点":[seg.ref.start_point.x,seg.ref.start_point.y],
                    "终点":[seg.ref.end_point.x,seg.ref.end_point.y],
                    "长度":seg.length()
                }
                json_data["自由边"][-1]["句柄"]=seg.ref.handle
                json_data["自由边"][-1]["标注"]=all_edge_map[seg]
                log_to_file(file_path, f"       起点：{seg.start_point}、终点{seg.end_point}、长度：{seg.length()}（直线）、句柄: {seg.ref.handle}")
                des=f""
                for ty,annos in all_edge_map[seg].items():
                    anno_des=""
                    for anno in annos:
                        anno_des=f"{anno_des},{anno[1]}"
                    des=f"{des};{ty}:{anno_des}"
                log_to_file(file_path, f"           标注:{des}")
            elif edge_types[seg]=="KS_corner":
                json_data["自由边"][-1]["几何"]={
                    "起点":[seg.ref.start_point.x,seg.ref.start_point.y],
                    "终点":[seg.ref.end_point.x,seg.ref.end_point.y],
                    "长度":seg.length()
                }
                json_data["自由边"][-1]["句柄"]=seg.ref.handle
                json_data["自由边"][-1]["标注"]=all_edge_map[seg]
                log_to_file(file_path, f"       起点：{seg.start_point}、终点{seg.end_point}、长度：{seg.length()}（直线）、句柄: {seg.ref.handle}")
                des=f""
                for ty,annos in all_edge_map[seg].items():
                    anno_des=""
                    for anno in annos:
                        anno_des=f"{anno_des},{anno[1]}"
                    des=f"{des}\n\t\t\t{ty}:{anno_des}"
                log_to_file(file_path, f"           标注:{des}")   
        

    
        #输出加强信息
        s_info="没有加强提示"
        has_fb_text=0
        for t_t in tis:
            result=t_t[2]
            if result["Type"]=="FB":
                has_fb_text=1
                break
            if result["Type"]=="FL":
                has_fb_text=2
                break
        if has_fb_text==1:
            s_info=="FB"
        elif has_fb_text ==2:
            s_info=="折边肘板"
        else:
            if is_fb:
                s_info="带筋肘板"
            else:
                for s in free_edges[0]:
                    if s.isPart:
                        s_info="折边肘板"
                        break
        

        log_to_file(file_path, f"肘板加强类别为:{s_info}")
        json_data["肘板加强类别"]=s_info
        if strengthen_parameter is  None:
            log_to_file(file_path,f"肘板加强参数为：缺省")
            json_data["肘板加强参数"]={
                "缺省":True
            }
        else:
            section_height=strengthen_parameter["Section Height"]
            ty=strengthen_parameter["Type"]
            thick=strengthen_parameter["Thickness"]
            material=strengthen_parameter["Material"]
            special=strengthen_parameter["Special"]
            if section_height is None:
                section_height="缺省"
            if ty is None:
                ty="缺省"
            if thick is None:
                thick="缺省"
            if material is None:
                material="缺省"
            if special is None:
                special="缺省"
            log_to_file(file_path,f"肘板加强参数为：")
            log_to_file(file_path,f"    加强类型：{ty}")
            log_to_file(file_path,f"    截面高度：{section_height}")

            log_to_file(file_path,f"    厚度：{thick}")
            log_to_file(file_path,f"    材质：{material}")
            log_to_file(file_path,f"    特殊符号：{special}")
            json_data["肘板加强参数"]={
                "缺省":False,
                "加强类型":ty,
                "截面高度":section_height,
                "厚度":thick,
                "材质":material,
                "特殊符号":special
            }
       
        if  bracket_parameter is None:
            log_to_file(file_path,f"肘板参数为：缺省")
            json_data["肘板加强参数"]={
                "缺省":True
            }
        else:
            log_to_file(file_path,f"肘板参数为：")
            al1=bracket_parameter["Arm Length1"]
            al2=bracket_parameter["Arm Length2"]
            thick=bracket_parameter["Thickness"]
            material=bracket_parameter["Material"]
            special=bracket_parameter["Special"]

            cons_edges=[]
            for i,edge in enumerate(edges):
                if (not edge[0].isConstraint) and (not edge[0].isCornerhole):
                    continue
                if edge[0].isCornerhole:
                    continue
                cons_edges.append(edge)
            edge1,edge2=cons_edges[0],cons_edges[-1]
            if edge1[0].length()<edge2[0].length():
                edge1,edge2=edge2,edge1
            if al1 is None:
                anno1=""
                anno2=""
            elif al2 is None:
                if (edge1[0].length()-edge2[0].length())<=25:
                    anno1=f" 参考边：约束边{str(constraint_edge_no[edge1[0]])}，约束边{str(constraint_edge_no[edge2[0]])}"
                else:
                    anno1=f" 参考边：约束边{str(constraint_edge_no[edge1[0]])}"
                anno2=""
            else:
                if al1>al2:
                    anno1=f" 参考边：约束边{str(constraint_edge_no[edge1[0]])}"
                    anno2=f" 参考边：约束边{str(constraint_edge_no[edge2[0]])}"
                else:
                    anno1=f" 参考边：约束边{str(constraint_edge_no[edge2[0]])}"
                    anno2=f" 参考边：约束边{str(constraint_edge_no[edge1[0]])}"
            
            if al1 is None:
                al1="缺省"
            if al2 is None:
                al2="缺省"
            if thick is None:
                thick="缺省"
            if material is None:
                material="缺省"
            if special is None:
                special="缺省"
            
            log_to_file(file_path,f"    臂长1：{al1}{anno1}")
            # str(constranit_edge_template_no[seg])

            log_to_file(file_path,f"    臂长2：{al2}{anno2}")

            log_to_file(file_path,f"    厚度：{thick}")
            log_to_file(file_path,f"    材质：{material}")
            log_to_file(file_path,f"    特殊符号：{special}")
            json_data["肘板参数"]={
                "缺省":False,
                "臂长1":{"长度":al1,"位置":anno1},
                "臂长2":{"长度":al2,"位置":anno2},
                "厚度":thick,
                "材质":material,
                "特殊符号":special
            }
        log_to_file(file_path, f"肘板类别为{classification_res}")
        json_data["肘板类别"]=classification_res
        log_to_file(file_path, f"肘板混淆类分类特征为：{str(features)}")
        json_data["肘板混淆类分类特征"]=features
        free_idx=1
        free_codes=[]
        non_free_codes=[]
        for i,edge in enumerate(free_edges[0]):

            free_codes.append(feature_map[edge])
            free_idx+=1
        for i,edge in enumerate(edges):
            if (not edge[0].isConstraint) and (not edge[0].isCornerhole):
                continue
            
            non_free_codes.append(feature_map[edge[0]])
        log_to_file(file_path, f"肘板混淆类分类逐边特征为：")
        log_to_file(file_path, f"   free_codes：{str(free_codes)}")
        log_to_file(file_path, f"   non_free_codes：{str(non_free_codes)}")
        json_data["肘板混淆类逐边分类特征"]={
            "free_codes":free_codes,
            "non_free_codes":non_free_codes
        }
        log_to_file(file_path, f"非标准肘板")
        json_data["标准肘板"]=False
        json_str=json.dumps(json_data,indent=4,ensure_ascii=False,default=str)
        with open(json_name,'w',encoding='utf-8') as f:
            f.write(json_str)
        
        return poly_refs, classification_res,meta_hints,size_hints


    cornerhole_num=0
    for i,edge in enumerate(edges):
        if (not edge[0].isConstraint) and (not edge[0].isCornerhole):
            continue
        if edge[0].isCornerhole:
            cornerhole_num+=1

    thickness=0
    if meta_info[1] is not None and meta_info[1]["Thickness"] is not None:
        thickness=meta_info[1]["Thickness"]
    classification_res,output_template = poly_classifier(features,all_anno,poly_refs, tis,ds,cornerhole_num, free_edges, edges, thickness,feature_map,edge_types,
                                         segmentation_config.standard_type_path, segmentation_config.json_output_path, 
                                         f"{os.path.splitext(os.path.basename(segmentation_config.json_path))[0]}_infopoly{index}",
                                         is_output_json=True)
    if classification_res is None:
        classification_res="Unclassified"
        output_template=None
    free_order=True
    # cons_order=True
    if output_template is not None:

        if thickness>25:
            ignore_types=["arc"]
        else:
            ignore_types=["line"]
        template_map=match_template(edges,free_edges,output_template,edge_types,thickness)
        if check_one_class(classification_res,template_map,free_edges,constraint_edges,edges,poly_refs,poly,all_edge_map, ref_map)==False:
            return poly_refs,"Unclassified",[],[]
        # print(output_template)
        free_edge_template_no={}
        free_idx=1
        while f'free{free_idx}' in template_map:
            if len(template_map[f'free{free_idx}'])==0:
                free_idx+=1
                continue
            seg=template_map[f'free{free_idx}'][0]
            free_edge_template_no[seg]=free_idx
            free_idx+=1
        constranit_edge_template_no={}
        cornerhole_edge_template_no={}
        constarint_idx=1
        cornerhole_idx=1
        while True:
            # print(constarint_idx,cornerhole_idx)
            if f'constraint{constarint_idx}' in template_map:
                for edge in template_map[f'constraint{constarint_idx}']:

                    constranit_edge_template_no[edge]=constarint_idx
                
                constarint_idx+=1
            else:
                break
            if f'cornerhole{cornerhole_idx}' in template_map:  
                for edge in template_map[f'cornerhole{cornerhole_idx}']:
    
                    cornerhole_edge_template_no[edge]=cornerhole_idx
                cornerhole_idx+=1



        constarint_idx=1
        cornerhole_idx=1
        template_edges=[]
        # print("============")
        while True:
            # print(constarint_idx,cornerhole_idx)
            if f'constraint{constarint_idx}' in template_map:
                template_edges.append(template_map[f'constraint{constarint_idx}'])
                constarint_idx+=1
            else:
                break
            if f'cornerhole{cornerhole_idx}' in template_map:  
                template_edges.append(template_map[f'cornerhole{cornerhole_idx}'])
                cornerhole_idx+=1

        template_cons_edges=[]
        for i,edge in enumerate(template_edges):
    

            if len(edge)==0:
                continue

            elif edge[0].isCornerhole:
                continue
            else:
                for seg in edge:
                    template_cons_edges.append(seg)

        order_set=set()
       
        for i in range(len(template_cons_edges)-1):
            current_edge=template_cons_edges[i]
            next_edge=template_cons_edges[i+1]
            pairs=[(current_edge.start_point,next_edge.start_point),(current_edge.start_point,next_edge.end_point),(current_edge.end_point,next_edge.start_point),(current_edge.end_point,next_edge.end_point)]
            distance=float('inf')
            min_pair=None
            for pair in pairs:
                p1,p2=pair
                dis=DSegment(p1,p2).length()
                if dis< distance:
                    distance=dis
                    min_pair=pair
            p2,p=min_pair
            if  p2==current_edge.start_point:
                p1=current_edge.end_point
            else:
                p1=current_edge.start_point
            order_set.add((current_edge,p1))
            if i==len(template_cons_edges)-2:
                order_set.add((next_edge,p))

        


        new_all_edge_map={}
        for s,dic in all_edge_map.items():
            new_dic={}
            for ty,ds in dic.items():
                if isinstance(ds,list):
                    new_ds=[]
                    for d_t in ds:
                        if len(d_t)==4:
                            d,des,seg=d_t[0],d_t[1],d_t[2] 
                            _,p1,p2=d_t[3]
                            if seg in constranit_edge_template_no:
                                des+=str(constranit_edge_template_no[seg])
                            else:
                                des+=str(free_edge_template_no[seg])
                            order_ty=""
                            if (seg,p1) in order_set:
                                des+="，1号边"
                                order_ty="A"
                            else:
                                des+="，2号边"
                                order_ty="B"
                            new_ds.append([d,des])
                            edge_no=""
                            if s in free_edge_template_no:
                                edge_no=f"自由边{free_edge_template_no[s]}"
                            elif s in constranit_edge_template_no:
                                edge_no=f"约束边{constranit_edge_template_no[s]}"
                            ref_no="无"
                            if seg in free_edge_template_no:
                                ref_no=f"自由边{free_edge_template_no[seg]}"
                            elif seg in constranit_edge_template_no:
                                ref_no=f"约束边{constranit_edge_template_no[seg]}"
                            size_hints.append([index,d.handle,poly_centroid,classification_res,ty,edge_no,s.ref.handle,ref_no,seg.ref.handle,d.text,order_ty,is_diff])
                        elif len(d_t)==3:
                            d,des,seg=d_t 
                            if seg in constranit_edge_template_no:
                                des+=str(constranit_edge_template_no[seg])
                            else:
                                des+=str(free_edge_template_no[seg])
                            new_ds.append([d,des])
                            if d.dimtype!=0:
                                edge_no=""
                                if s in free_edge_template_no:
                                    edge_no=f"自由边{free_edge_template_no[s]}"
                                elif s in constranit_edge_template_no:
                                    edge_no=f"约束边{constranit_edge_template_no[s]}"
                                ref_no="无"
                                if seg in free_edge_template_no:
                                    ref_no=f"自由边{free_edge_template_no[seg]}"
                                elif seg in constranit_edge_template_no:
                                    ref_no=f"约束边{constranit_edge_template_no[seg]}"
                                size_hints.append([index,d.handle,poly_centroid,classification_res,ty,edge_no,s.ref.handle,ref_no,seg.ref.handle,d.text,"无",is_diff])
                        else:
                            new_ds.append(d_t)
                            if len(d_t)==2:
                                d=d_t[0]
                                if isinstance(d_t[0],DDimension):
                                    if d.dimtype!=0:
                                        edge_no=""
                                        if s in free_edge_template_no:
                                            edge_no=f"自由边{free_edge_template_no[s]}"
                                        elif s in constranit_edge_template_no:
                                            edge_no=f"约束边{constranit_edge_template_no[s]}"
                                        size_hints.append([index,d.handle,poly_centroid,classification_res,ty,edge_no,s.ref.handle,"无","无",d.text,"无",is_diff])
                                else:
                                    edge_no=""
                                    if s in free_edge_template_no:
                                        edge_no=f"自由边{free_edge_template_no[s]}"
                                    elif s in constranit_edge_template_no:
                                        edge_no=f"约束边{constranit_edge_template_no[s]}"
                                    else:
                                        edge_no=f"角隅孔{cornerhole_edge_template_no[s]}"
                                    size_hints.append([index,d.handle,poly_centroid,classification_res,ty,edge_no,s.ref.handle,"无","无",d.content,"无",is_diff])
                    new_dic[ty]=new_ds
                else:
                    new_dic[ty]=ds
            new_all_edge_map[s]=new_dic
        ori_edge_map=all_edge_map
        all_edge_map=new_all_edge_map


        for t_t in tis:
            content=t_t[0].content.strip()
            if t_t[2]["Type"]=="B" or t_t[2]["Type"]=="FB" or t_t[2]["Type"]=="FL":
                meta_hints.append([index,t_t[0].handle,poly_centroid,classification_res,content,False])






        file_path = os.path.join(segmentation_config.poly_info_dir, f'标准肘板/info{index}.txt')
        clear_file(file_path)
        log_to_file(file_path, f"几何中心坐标：{poly_centroid}")
        json_data["几何中心坐标"]=[poly_centroid[0],poly_centroid[1]]
        json_data["主图标题"] = poumian_name
        json_data["调用总次数"] = poumian_copy_time

        free_idx=1
        log_to_file(file_path, f"自由边轮廓({get_free_edge_des(free_edges[0],edge_types)})：")
        json_data["自由边轮廓"]=[edge_types[s] for s in free_edges[0]]
        json_data["自由边"]=[]
        while f'free{free_idx}' in template_map:
            if len(template_map[f'free{free_idx}'])==0:
                log_to_file(file_path, f"   自由边{free_idx}")
                log_to_file(file_path, f"        类型：KS_corner")
                log_to_file(file_path,f"         缺省")
                json_data["自由边"].append({"序号":free_idx,"类型":"KS_corner","缺省":True})
                free_idx+=1
                continue
            seg=template_map[f'free{free_idx}'][0]

            
            log_to_file(file_path, f"   自由边{free_idx}")
            log_to_file(file_path, f"        类型：{edge_types[seg]}")

            json_data["自由边"].append({"序号":free_idx,"类型":edge_types[seg],"缺省":False})
            if edge_types[seg]=="arc":
                log_to_file(file_path, f"       起点：{seg.ref.start_point}、终点：{seg.ref.end_point}、圆心：{seg.ref.center}、半径：{seg.ref.radius}（圆弧）、句柄: {seg.ref.handle}")
                json_data["自由边"][-1]["几何"]={
                    "起点":[seg.ref.start_point.x,seg.ref.start_point.y],
                    "终点":[seg.ref.end_point.x,seg.ref.end_point.y],
                    "圆心":[seg.ref.center.x,seg.ref.center.y],
                    "半径":seg.ref.radius
                }
                json_data["自由边"][-1]["句柄"]=seg.ref.handle
                json_data["自由边"][-1]["标注"]=all_edge_map[seg]
                anno_des=""
                for anno in all_edge_map[seg]["半径尺寸标注"]:
                    anno_des=f"{anno_des},{anno[1]}"
                des=f"是否相切:{all_edge_map[seg]["是否相切"]}\n\t\t\t半径尺寸标注:{anno_des}\n\t\t\t圆心是否在趾端延长线上:{all_edge_map[seg]["圆心是否在趾端延长线上"]}"
            
                log_to_file(file_path, f"           标注:\n\t\t\t{des}")        
            elif edge_types[seg]=="line":
                
                log_to_file(file_path, f"       起点：{seg.start_point}、终点{seg.end_point}、长度：{seg.length()}（直线）、句柄: {seg.ref.handle}")
                json_data["自由边"][-1]["几何"]={
                    "起点":[seg.ref.start_point.x,seg.ref.start_point.y],
                    "终点":[seg.ref.end_point.x,seg.ref.end_point.y],
                    "长度":seg.length()
                }
                json_data["自由边"][-1]["句柄"]=seg.ref.handle
                json_data["自由边"][-1]["标注"]=all_edge_map[seg]
                
                des=f"是否与约束边平行:{all_edge_map[seg]["是否与约束边平行"]}\n\t\t\t是否与相邻约束边夹角为90度:{all_edge_map[seg]["是否与相邻约束边夹角为90度"]}"
                for ty,annos in all_edge_map[seg].items():
                    if ty=="是否与约束边平行" or ty=="是否与相邻约束边夹角为90度" :
                        continue
                
                    anno_des=""
                    if ty=="":
                        for anno in annos:
                            anno_des=f"{anno_des},{anno[1]}"
                    else:

                        for anno in annos:
                            anno_des=f"{anno_des},{anno[1]}"
                    des=f"{des}\n\t\t\t{ty}:{anno_des}"
                log_to_file(file_path, f"           标注:\n\t\t\t{des}")
            elif edge_types[seg]=="toe":  
                log_to_file(file_path, f"       起点：{seg.start_point}、终点{seg.end_point}、长度：{seg.length()}（直线）、句柄: {seg.ref.handle}")
                
                json_data["自由边"][-1]["几何"]={
                    "起点":[seg.ref.start_point.x,seg.ref.start_point.y],
                    "终点":[seg.ref.end_point.x,seg.ref.end_point.y],
                    "长度":seg.length()
                }
                json_data["自由边"][-1]["句柄"]=seg.ref.handle
                json_data["自由边"][-1]["标注"]=all_edge_map[seg]
                des=f""
                for ty,annos in all_edge_map[seg].items():
                    anno_des=""
                    for anno in annos:
                        anno_des=f"{anno_des},{anno[1]}"
                    des=f"{des}\n\t\t\t{ty}:{anno_des}"
                log_to_file(file_path, f"           标注:{des}")
            elif edge_types[seg]=="KS_corner":
                log_to_file(file_path, f"       起点：{seg.start_point}、终点{seg.end_point}、长度：{seg.length()}（直线）、句柄: {seg.ref.handle}")
                json_data["自由边"][-1]["几何"]={
                    "起点":[seg.start_point.x,seg.start_point.y],
                    "终点":[seg.end_point.x,seg.end_point.y],
                    "长度":seg.length()
                }
                json_data["自由边"][-1]["句柄"]=seg.ref.handle
                json_data["自由边"][-1]["标注"]=all_edge_map[seg]
                
                des=f""
                for ty,annos in all_edge_map[seg].items():
                    anno_des=""
                    for anno in annos:
                        anno_des=f"{anno_des},{anno[1]}"
                    des=f"{des}\n\t\t\t{ty}:{anno_des}"
                log_to_file(file_path, f"           标注:{des}")   
            
            free_idx+=1
        





        log_to_file(file_path, "边界信息（非自由边）：")
        k=1
        constarint_idx=0
        cornerhole_idx=0
        json_data["非自由边轮廓"]=[]
        json_data["非自由边"]=[]
        for i,edge in enumerate(template_edges):


            if len(edge)==0:
                if thickness>25:
                    content="R20"
                elif thickness<10:
                    content="KS10"
                else:
                    content="KS15"
                log_to_file(file_path, f"边界颜色{k}: 缺省")
                log_to_file(file_path, f"边界轮廓{k}({ignore_types[0]}): ")
                log_to_file(file_path, f"边界缺省轮廓{k}({content}): ")
                k+=1
                log_to_file(file_path,f"    角隅孔{cornerhole_idx+1}({get_edge_des(edge)})")
                log_to_file(file_path, f"       缺省")
                cornerhole_idx+=1
                json_data["非自由边轮廓"].append({"类型":"角隅孔","轮廓":[ignore_types[0]]})
                json_data["非自由边"].append({"缺省内容":content,"缺省":True})
            elif edge[0].isCornerhole:
                log_to_file(file_path, f"边界颜色{k}: {edge[0].ref.color}")
                log_to_file(file_path, f"边界轮廓{k}({get_edge_des(edge)}): ")
                k+=1
                corner_hole_start_edge=edge[0]
                log_to_file(file_path,f"    角隅孔{cornerhole_idx+1}({get_edge_des(edge)})")
                cornerhole_idx+=1
                json_data["非自由边轮廓"].append({"类型":"角隅孔","轮廓":["line" if isinstance(s.ref,DLine) else "arc" for s in edge]})
                json_data["非自由边"].append({"缺省":False,"颜色":edge[0].ref.color,"边":[]})
                if len(edge)>1 and is_vu(edge):
                    anno_des=""
                    anno_des2=""
                    for anno in all_edge_map[corner_hole_start_edge]["短边尺寸标注"]:
                        anno_des=f"{anno_des},{anno[1]}"
                    for anno in all_edge_map[corner_hole_start_edge]["尺寸参数"]:
                        anno_des2=f"{anno_des2},{anno[1]}"
                    des=f"短边是否平行于相邻边:{all_edge_map[corner_hole_start_edge]["短边是否平行于相邻边"]};\n\t\t\t短边尺寸标注:{anno_des}\n\t\t\t尺寸参数:{anno_des2}"
                    log_to_file(file_path,f"        VU孔标注:\n\t\t\t{des}")
                    json_data["非自由边"][-1]["标注"]=all_edge_map[corner_hole_start_edge]
                    for seg in edge:
                        if isinstance(seg.ref, DArc):
                            # actual_radius= seg.ref.radius if seg not in r_map else r_map[seg].content.lstrip("R")
                            log_to_file(file_path, f"       起点：{seg.ref.start_point}、终点：{seg.ref.end_point}、圆心：{seg.ref.center}、半径：{seg.ref.radius}（圆弧）、句柄: {seg.ref.handle}")
                            anno_des=""
                            for anno in all_edge_map[seg]["半径尺寸标注"]:
                                anno_des=f"{anno_des},{anno[1]}"
                            des=f"半径尺寸标注:{anno_des}"
                            log_to_file(file_path, f"           标注:\n\t\t\t{des}")   
                            json_data["非自由边"][-1]["边"].append({
                                "几何":{
                                    "起点":[seg.ref.start_point.x,seg.ref.start_point.y],
                                    "终点":[seg.ref.end_point.x,seg.ref.end_point.y],
                                    "圆心":[seg.ref.center.x,seg.ref.center.y],
                                    "半径":seg.ref.radius
                                },
                                "句柄":seg.ref.handle,
                                "标注":all_edge_map[seg],
                                "类型":"arc"

                            })
                        else:
                            json_data["非自由边"][-1]["边"].append({
                                "几何":{
                                    "起点":[seg.ref.start_point.x,seg.ref.start_point.y],
                                    "终点":[seg.ref.end_point.x,seg.ref.end_point.y],
                                    "长度":seg.length()
                                },
                                "句柄":seg.ref.handle,
                                "标注":{},
                                "类型":"line"

                            })
                            log_to_file(file_path, f"       起点：{seg.start_point}、终点{seg.end_point}（直线）、句柄: {seg.ref.handle}")
                else:

                    for seg in edge:
                        if isinstance(seg.ref, DArc):
                            # actual_radius= seg.ref.radius if seg not in r_map else r_map[seg].content.lstrip("R")
                            log_to_file(file_path, f"       起点：{seg.ref.start_point}、终点：{seg.ref.end_point}、圆心：{seg.ref.center}、半径：{seg.ref.radius}（圆弧）、句柄: {seg.ref.handle}")
                            anno_des=""
                            for anno in all_edge_map[seg]["半径尺寸标注"]:
                                anno_des=f"{anno_des},{anno[1]}"
                            des=f"半径尺寸标注:{anno_des}"
                            log_to_file(file_path, f"           标注:\n\t\t\t{des}")                  
                        else:
                            
                            log_to_file(file_path, f"       起点：{seg.start_point}、终点{seg.end_point}（直线）、句柄: {seg.ref.handle}")
            else:
                log_to_file(file_path, f"边界颜色{k}: {edge[0].ref.color}")
                log_to_file(file_path, f"边界轮廓{k}({get_edge_des(edge)}): ")
                k+=1
                log_to_file(file_path,f"    约束边{constarint_idx+1}({get_edge_des(edge)})")
                json_data["非自由边轮廓"].append({"类型":"约束边","轮廓":["line" if isinstance(s.ref,DLine) else "arc" for s in edge]})
                json_data["非自由边"].append({"缺省":False,"颜色":edge[0].ref.color,"边":[],"实际轮廓":[],"延长后直线":[]})
                constarint_idx+=1
                next_cons_edge=[]
                last_cons_edge=[]
                index_=i+1
                while index_<len(template_edges):
                    next_e=template_edges[index_]
                    if not(len(next_e)==0 or next_e[0].isCornerhole):
                        next_cons_edge=next_e
                        break
                    index_+=1
                index_=i-1
                while index_>=0:
                    last_e=template_edges[index_]
                    if not(len(last_e)==0 or last_e[0].isCornerhole):
                        last_cons_edge=last_e
                        break
                    index_-=1
                next_seg=next_cons_edge[0] if len(next_cons_edge)>0 else None
                last_seg=last_cons_edge[0] if len(last_cons_edge)>0 else None
                if last_seg is not None:
                    inter1=find_intersection(last_seg.start_point,last_seg.end_point,seg.start_point,seg.end_point)
                else:
                    inter1=None
                if next_seg is not None:
                    inter2=find_intersection(next_seg.start_point,next_seg.end_point,seg.start_point,seg.end_point)
                else:
                    inter2=None
                if inter1 is not None and inter2 is not None:
                    new_seg=DSegment(inter1,inter2)
                elif inter1 is not None:
                    if DSegment(inter1,seg.start_point).length()<DSegment(inter1,seg.end_point).length():
                        new_seg=DSegment(inter1,seg.end_point)
                    else:
                        new_seg=DSegment(inter1,seg.start_point)
                elif inter2 is not None:
                    if DSegment(inter2,seg.start_point).length()<DSegment(inter2,seg.end_point).length():
                        new_seg=DSegment(inter2,seg.end_point)
                    else:
                        new_seg=DSegment(inter2,seg.start_point)
                else:
                    new_seg=seg
                correct_edge=[new_seg]
                log_to_file(file_path,f"        约束边实际轮廓：")
                actual_segs=[]
                for seg in edge:
                    if seg in ref_map:
                        actual_segs.extend(ref_map[seg])
                    else:
                        actual_segs.append(seg)
                for seg in actual_segs:
                    if isinstance(seg.ref, DArc):
                        # actual_radius= seg.ref.radius if seg not in r_map else r_map[seg].content.lstrip("R")
                        log_to_file(file_path, f"           起点：{seg.ref.start_point}、终点：{seg.ref.end_point}、圆心：{seg.ref.center}、半径：{seg.ref.radius}（圆弧）、句柄: {seg.ref.handle}")
                        json_data["非自由边"][-1]["实际轮廓"].append({
                            "几何":{
                                "起点":[seg.ref.start_point.x,seg.ref.start_point.y],
                                "终点":[seg.ref.end_point.x,seg.ref.end_point.y],
                                "圆心":[seg.ref.center.x,seg.ref.center.y],
                                "半径":seg.ref.radius
                            },
                            "句柄":seg.ref.handle,
                            "类型":"arc"

                        })
                    else:
                        log_to_file(file_path, f"           起点：{seg.start_point}、终点{seg.end_point}（直线）、句柄: {seg.ref.handle}")
                        json_data["非自由边"][-1]["实际轮廓"].append({
                            "几何":{
                                "起点":[seg.start_point.x,seg.start_point.y],
                                "终点":[seg.end_point.x,seg.end_point.y]
                            },
                            "句柄":seg.ref.handle,
                            "类型":"line"

                        })


                log_to_file(file_path,f"        经过延长后直线：")
                for seg in correct_edge:
                    log_to_file(file_path, f"           起点：{seg.start_point}、终点{seg.end_point}（直线）、长度: {seg.length()}")
                    json_data["非自由边"][-1]["延长后直线"].append({
                        "几何":{
                            "起点":[seg.start_point.x,seg.start_point.y],
                            "终点":[seg.end_point.x,seg.end_point.y],
                            "长度":seg.length()
                        },
                        "类型":"line"

                    })
                log_to_file(file_path,f"        将约束边估计为直线：")
                for seg in edge:

                    if isinstance(seg.ref, DArc):
                        # actual_radius= seg.ref.radius if seg not in r_map else r_map[seg].content.lstrip("R")
                        log_to_file(file_path, f"           起点：{seg.ref.start_point}、终点：{seg.ref.end_point}、圆心：{seg.ref.center}、半径：{seg.ref.radius}（圆弧）、句柄: {seg.ref.handle}")
                        json_data["非自由边"][-1]["边"].append({
                            "几何":{
                                "起点":[seg.ref.start_point.x,seg.ref.start_point.y],
                                "终点":[seg.ref.end_point.x,seg.ref.end_point.y],
                                "圆心":[seg.ref.center.x,seg.ref.center.y],
                                "半径":seg.ref.radius
                            },
                            "句柄":seg.ref.handle,
                            "标注":{},
                            "类型":"arc"

                        })
                    else:
                        
                        log_to_file(file_path, f"           起点：{seg.start_point}、终点{seg.end_point}（直线）、句柄: {seg.ref.handle}")
                        des=""
                        for ty,annos in all_edge_map[seg].items():
                            anno_des=""
                            for anno in annos:
                                anno_des=f"{anno_des},{anno[1]}"
                            des=f"{des}\n\t\t\t\t{ty}:{anno_des}"
                        log_to_file(file_path, f"               标注:{des}")
                        json_data["非自由边"][-1]["边"].append({
                            "几何":{
                                "起点":[seg.ref.start_point.x,seg.ref.start_point.y],
                                "终点":[seg.ref.end_point.x,seg.ref.end_point.y],
                                "长度":seg.length()
                            },
                            "句柄":seg.ref.handle,
                            "标注":all_edge_map[seg],
                            "类型":"line"

                        })

    

   
        #输出加强信息
        s_info="没有加强提示"
        has_fb_text=0
        for t_t in tis:
            result=t_t[2]
            if result["Type"]=="FB":
                has_fb_text=1
                break
            if result["Type"]=="FL":
                has_fb_text=2
                break
        if has_fb_text==1:
            s_info=="FB"
        elif has_fb_text ==2:
            s_info=="折边肘板"
        else:
            if is_fb:
                s_info="带筋肘板"
            else:
                for s in free_edges[0]:
                    if s.isPart:
                        s_info="折边肘板"
                        break
        

        log_to_file(file_path, f"肘板加强类别为：{s_info}")
        json_data["肘板加强类别"]=s_info
        if strengthen_parameter is  None:
            log_to_file(file_path,f"肘板加强参数为：缺省")
            json_data["肘板加强参数"]={
                "缺省":True
            }
        else:
            section_height=strengthen_parameter["Section Height"]
            ty=strengthen_parameter["Type"]
            thick=strengthen_parameter["Thickness"]
            material=strengthen_parameter["Material"]
            special=strengthen_parameter["Special"]
            if section_height is None:
                section_height="缺省"
            if ty is None:
                ty="缺省"
            if thick is None:
                thick="缺省"
            if material is None:
                material="缺省"
            if special is None:
                special="缺省"
            log_to_file(file_path,f"肘板加强参数为：")
            log_to_file(file_path,f"    加强类型：{ty}")
            log_to_file(file_path,f"    截面高度：{section_height}")

            log_to_file(file_path,f"    厚度：{thick}")
            log_to_file(file_path,f"    材质：{material}")
            log_to_file(file_path,f"    特殊符号：{special}")
            json_data["肘板加强参数"]={
                "缺省":False,
                "加强类型":ty,
                "截面高度":section_height,
                "厚度":thick,
                "材质":material,
                "特殊符号":special
            }
       
        if  bracket_parameter is None:
            log_to_file(file_path,f"肘板参数为：缺省")
            json_data["肘板加强参数"]={
                "缺省":True
            }
        else:
            log_to_file(file_path,f"肘板参数为：")
            al1=bracket_parameter["Arm Length1"]
            al2=bracket_parameter["Arm Length2"]
            thick=bracket_parameter["Thickness"]
            material=bracket_parameter["Material"]
            special=bracket_parameter["Special"]

            cons_edges=[]
            for i,edge in enumerate(edges):
                if (not edge[0].isConstraint) and (not edge[0].isCornerhole):
                    continue
                if edge[0].isCornerhole:
                    continue
                cons_edges.append(edge)
            edge1,edge2=cons_edges[0],cons_edges[-1]
            if edge1[0].length()<edge2[0].length():
                edge1,edge2=edge2,edge1
            if al1 is None:
                anno1=""
                anno2=""
            elif al2 is None:
                if (edge1[0].length()-edge2[0].length())<=25:
                    anno1=f" 参考边：约束边{str(constranit_edge_template_no[edge1[0]])}，约束边{str(constranit_edge_template_no[edge2[0]])}"
                else:
                    anno1=f" 参考边：约束边{str(constranit_edge_template_no[edge1[0]])}"
                anno2=""
            else:
                if al1>al2:
                    anno1=f" 参考边：约束边{str(constranit_edge_template_no[edge1[0]])}"
                    anno2=f" 参考边：约束边{str(constranit_edge_template_no[edge2[0]])}"
                else:
                    anno1=f" 参考边：约束边{str(constranit_edge_template_no[edge2[0]])}"
                    anno2=f" 参考边：约束边{str(constranit_edge_template_no[edge1[0]])}"
            
            if al1 is None:
                al1="缺省"
            if al2 is None:
                al2="缺省"
            if thick is None:
                thick="缺省"
            if material is None:
                material="缺省"
            if special is None:
                special="缺省"
            
            log_to_file(file_path,f"    臂长1：{al1}{anno1}")
            # str(constranit_edge_template_no[seg])

            log_to_file(file_path,f"    臂长2：{al2}{anno2}")

            log_to_file(file_path,f"    厚度：{thick}")
            log_to_file(file_path,f"    材质：{material}")
            log_to_file(file_path,f"    特殊符号：{special}")
            json_data["肘板参数"]={
                "缺省":False,
                "臂长1":{"长度":al1,"位置":anno1},
                "臂长2":{"长度":al2,"位置":anno2},
                "厚度":thick,
                "材质":material,
                "特殊符号":special
            }
        log_to_file(file_path, f"肘板类别为{classification_res}")
        json_data["肘板类别"]=classification_res
        log_to_file(file_path, f"肘板混淆类分类特征为：{str(features)}")
        json_data["肘板混淆类分类特征"]=features
        free_idx=1
        free_codes=[]
        non_free_codes=[]
        while f'free{free_idx}' in template_map:
            if len(template_map[f'free{free_idx}'])==0:
                free_idx+=1
                free_codes.append([])
                continue

            free_codes.append(feature_map[template_map[f'free{free_idx}'][0]])
            free_idx+=1
        for i,edge in enumerate(template_edges):
            if len(edge)==0:
                non_free_codes.append([])
            elif edge[0].isCornerhole:
                non_free_codes.append(feature_map[edge[0]])
            else:
                non_free_codes.append(feature_map[edge[0]])
        log_to_file(file_path, f"肘板混淆类分类逐边特征为：")
        log_to_file(file_path, f"   free_codes：{str(free_codes)}")
        log_to_file(file_path, f"   non_free_codes：{str(non_free_codes)}")
        json_data["肘板混淆类逐边分类特征"]={
            "free_codes":free_codes,
            "non_free_codes":non_free_codes
        }
        log_to_file(file_path, f"标准肘板")
        json_data["标准肘板"]=True
        json_str=json.dumps(json_data,indent=4,ensure_ascii=False,default=str)
        with open(json_name,'w',encoding='utf-8') as f:
            f.write(json_str)
        if is_diff==False:
            plot_info_poly_std(constraint_edges,ori_edge_map,template_map,os.path.join(segmentation_config.poly_info_dir, f'标准肘板详细信息参考图/std_infopoly{index}.png'))
        if classification_res == "Unclassified":
            # log_to_file("./output/Unclassified.txt", f"{os.path.splitext(os.path.basename(segmentation_config.json_path))[0]}_infopoly{index}")
            return poly_refs, classification_res,[],[]
        # else:
        #     if len(classification_res.split(","))>1:
        #         log_to_file("./output/duplicate_class.txt",f"{os.path.splitext(os.path.basename(segmentation_config.json_path))[0]}_infopoly{index}")
        
        
        matched_types=[]
        for t in classification_res.split(','):
            if t.strip()=="":
                continue
            matched_types.append(t)
        if len(matched_types)>1 :
            log_to_file("./output/duplicate_class.txt",f"{os.path.splitext(os.path.basename(segmentation_config.json_path))[0]}_infopoly{index}   {str(matched_types)}")
    else:
        template_cons_edges=[]
        for i,edge in enumerate(edges):
    

            if len(edge)==0:
                continue

            elif edge[0].isCornerhole:
                continue
            else:
                for seg in edge:
                    template_cons_edges.append(seg)

        order_set=set()
       
        for i in range(len(template_cons_edges)-1):
            current_edge=template_cons_edges[i]
            next_edge=template_cons_edges[i+1]
            pairs=[(current_edge.start_point,next_edge.start_point),(current_edge.start_point,next_edge.end_point),(current_edge.end_point,next_edge.start_point),(current_edge.end_point,next_edge.end_point)]
            distance=float('inf')
            min_pair=None
            for pair in pairs:
                p1,p2=pair
                dis=DSegment(p1,p2).length()
                if dis< distance:
                    distance=dis
                    min_pair=pair
            p2,p=min_pair
            if  p2==current_edge.start_point:
                p1=current_edge.end_point
            else:
                p1=current_edge.start_point
            order_set.add((current_edge,p1))
            if i==len(template_cons_edges)-2:
                order_set.add((next_edge,p))
        new_all_edge_map={}
        for s,dic in all_edge_map.items():
            new_dic={}
            for ty,ds in dic.items():
                if isinstance(ds,list):
                    new_ds=[]
                    for d_t in ds:
                        if len(d_t)==4:
                            d,des,seg=d_t[0],d_t[1],d_t[2] 
                            _,p1,p2=d_t[3]
                            if seg in constraint_edge_no:
                                des+=str(constraint_edge_no[seg])
                            else:
                                des+=str(free_edge_no[seg])
                            if (seg,p1) in order_set:
                                des+="，1号边" 
                            else:
                                des+="，2号边"
                            new_ds.append([d,des])
                        elif len(d_t)==3:
                            d,des,seg=d_t 
                            if seg in constraint_edge_no:
                                des+=str(constraint_edge_no[seg])
                            else:
                                des+=str(free_edge_no[seg])
                            new_ds.append([d,des])
                        else:
                            new_ds.append(d_t)
                    new_dic[ty]=new_ds
                else:
                    new_dic[ty]=ds
            new_all_edge_map[s]=new_dic
        all_edge_map=new_all_edge_map
        file_path = os.path.join(segmentation_config.poly_info_dir, f'标准肘板(无分类)/info{index}.txt')

        clear_file(file_path)
        log_to_file(file_path, f"几何中心坐标：{poly_centroid}")
        log_to_file(file_path, f"边界数量（含自由边）：{len(constraint_edges) + len(free_edges)}")
        log_to_file(file_path, "边界信息（非自由边）：")
        constarint_idx=0
        cornerhole_idx=0
        k=1
        for i,edge in enumerate(edges):
            if (not edge[0].isConstraint) and (not edge[0].isCornerhole):
                continue
            log_to_file(file_path, f"边界颜色{k}: {edge[0].ref.color}")
            log_to_file(file_path, f"边界轮廓{k}({get_edge_des(edge)}): ")
            k+=1
            if edge[0].isCornerhole:
                corner_hole_start_edge=edge[0]
                log_to_file(file_path,f"    角隅孔{cornerhole_idx+1}({get_edge_des(edge)})")
                cornerhole_idx+=1
                if len(edge)>1 and is_vu(edge):
                    anno_des=""
                    anno_des2=""
                    for anno in all_edge_map[corner_hole_start_edge]["短边尺寸标注"]:
                        anno_des=f"{anno_des},{anno[1]}"
                    for anno in all_edge_map[corner_hole_start_edge]["尺寸参数"]:
                        anno_des2=f"{anno_des2},{anno[1]}"
                    des=f"短边是否平行于相邻边:{all_edge_map[corner_hole_start_edge]["短边是否平行于相邻边"]};\n\t\t\t短边尺寸标注:{anno_des}\n\t\t\r尺寸参数:{anno_des2}"
                    log_to_file(file_path,f"        VU孔标注:\n\t\t\t{des}")
                    for seg in edge:
                        if isinstance(seg.ref, DArc):
                            log_to_file(file_path, f"       起点：{seg.ref.start_point}、终点：{seg.ref.end_point}、圆心：{seg.ref.center}、半径：{seg.ref.radius}（圆弧）、句柄: {seg.ref.handle}")
                            anno_des=""
                            for anno in all_edge_map[seg]["半径尺寸标注"]:
                                anno_des=f"{anno_des},{anno[1]}"
                            des=f"半径尺寸标注:{anno_des}"
                            log_to_file(file_path, f"           标注:\n\t\t\t{des}")                  
                        else:
                            
                            log_to_file(file_path, f"       起点：{seg.start_point}、终点{seg.end_point}（直线）、句柄: {seg.ref.handle}")
                else:

                    for seg in edge:
                        if isinstance(seg.ref, DArc):
                            log_to_file(file_path, f"       起点：{seg.ref.start_point}、终点：{seg.ref.end_point}、圆心：{seg.ref.center}、半径：{seg.ref.radius}（圆弧）、句柄: {seg.ref.handle}")
                            anno_des=""
                            for anno in all_edge_map[seg]["半径尺寸标注"]:
                                anno_des=f"{anno_des},{anno[1]}"
                            des=f"半径尺寸标注:{anno_des}"
                            log_to_file(file_path, f"           标注:\n\t\t\t{des}")                  
                        else:
                            
                            log_to_file(file_path, f"       起点：{seg.start_point}、终点{seg.end_point}（直线）、句柄: {seg.ref.handle}")
            else:
                log_to_file(file_path,f"    约束边{constarint_idx+1}({get_edge_des(edge)})")
                constarint_idx+=1
                for seg in edge:
                    if isinstance(seg.ref, DArc):
                        log_to_file(file_path, f"       起点：{seg.ref.start_point}、终点：{seg.ref.end_point}、圆心：{seg.ref.center}、半径：{seg.ref.radius}（圆弧）、句柄: {seg.ref.handle}")
                    else:
                        
                        log_to_file(file_path, f"       起点：{seg.start_point}、终点{seg.end_point}（直线）、句柄: {seg.ref.handle}")
                        des=""
                        for ty,annos in all_edge_map[seg].items():
                            anno_des=""
                            for anno in annos:
                                anno_des=f"{anno_des},{anno[1]}"
                            des=f"{des}\n\t\t\t{ty}:{anno_des}"
                        log_to_file(file_path, f"           标注:{des}")

        # step8: 输出自由边信息
        log_to_file(file_path, f"自由边轮廓({get_free_edge_des(free_edges[0],edge_types)})：")
        free_idx=0
        for seg in free_edges[0]:
            log_to_file(file_path, f"   自由边{free_idx+1}")
            log_to_file(file_path, f"        类型：{edge_types[seg]}")

       
            free_idx+=1
            if edge_types[seg]=="arc":
                log_to_file(file_path, f"       起点：{seg.ref.start_point}、终点：{seg.ref.end_point}、圆心：{seg.ref.center}、半径：{seg.ref.radius}（圆弧）、句柄: {seg.ref.handle}")
                anno_des=""
                for anno in all_edge_map[seg]["半径尺寸标注"]:
                    anno_des=f"{anno_des},{anno[1]}"
                des=f"是否相切:{all_edge_map[seg]["是否相切"]}\n\t\t\t半径尺寸标注:{anno_des}\n\t\t\t圆心是否在趾端延长线上:{all_edge_map[seg]["圆心是否在趾端延长线上"]}"
            
                log_to_file(file_path, f"           标注:\n\t\t\t{des}")        
            elif edge_types[seg]=="line":
                
                log_to_file(file_path, f"       起点：{seg.start_point}、终点{seg.end_point}、长度：{seg.length()}（直线）、句柄: {seg.ref.handle}")
                des=f"是否与约束边平行:{all_edge_map[seg]["是否与约束边平行"]}\n\t\t\t是否与相邻约束边夹角为90度:{all_edge_map[seg]["是否与相邻约束边夹角为90度"]}"
                for ty,annos in all_edge_map[seg].items():
                    if ty=="是否与约束边平行" or ty=="是否与相邻约束边夹角为90度" :
                        continue
                    anno_des=""
                    for anno in annos:
                        anno_des=f"{anno_des},{anno[1]}"
                    des=f"{des}\n\t\t\t{ty}:{anno_des}"
                log_to_file(file_path, f"           标注:\n\t\t\t{des}")
            elif edge_types[seg]=="toe":  
                log_to_file(file_path, f"       起点：{seg.start_point}、终点{seg.end_point}、长度：{seg.length()}（直线）、句柄: {seg.ref.handle}")
                des=f""
                for ty,annos in all_edge_map[seg].items():
                    anno_des=""
                    for anno in annos:
                        anno_des=f"{anno_des},{anno[1]}"
                    des=f"{des}\n\t\t\t{ty}:{anno_des}"
                log_to_file(file_path, f"           标注:{des}")
            elif edge_types[seg]=="KS_corner":
                log_to_file(file_path, f"       起点：{seg.start_point}、终点{seg.end_point}、长度：{seg.length()}（直线）、句柄: {seg.ref.handle}")
                des=f""
                for ty,annos in all_edge_map[seg].items():
                    anno_des=""
                    for anno in annos:
                        anno_des=f"{anno_des},{anno[1]}"
                    des=f"{des}\n\t\t\t{ty}:{anno_des}"
                log_to_file(file_path, f"           标注:{des}")   
        

    
        #输出加强信息
        s_info="没有加强提示"
        has_fb_text=0
        for t_t in tis:
            result=t_t[2]
            if result["Type"]=="FB":
                has_fb_text=1
                break
            if result["Type"]=="FL":
                has_fb_text=2
                break
        if has_fb_text==1:
            s_info=="FB"
        elif has_fb_text ==2:
            s_info=="折边肘板"
        else:
            if is_fb:
                s_info="带筋肘板"
            else:
                for s in free_edges[0]:
                    if s.isPart:
                        s_info="折边肘板"
                        break
        

        log_to_file(file_path, f"肘板加强类别为:{s_info}")

        log_to_file(file_path, f"肘板类别为{classification_res}")
        log_to_file(file_path, f"肘板混淆类分类特征为:{str(features)}")
        free_codes=[]
        non_free_codes=[]
        for edge in free_edges[0]:
            free_codes.append(feature_map[edge])
        for i,edge in enumerate(edges):
            if len(edge)==0:
                non_free_codes.append([])
            elif edge[0].isCornerhole:
                non_free_codes.append(feature_map[edge[0]])
            else:
                non_free_codes.append(feature_map[edge[0]])
        log_to_file(file_path, f"肘板混淆类分类逐边特征为：")
        log_to_file(file_path, f"   free_codes：{str(free_codes)}")
        log_to_file(file_path, f"   non_free_codes：{str(non_free_codes)}")
        log_to_file(file_path, f"标准肘板")
        return poly_refs, classification_res,[],[]

    return poly_refs, classification_res,meta_hints,size_hints


def   outputHints(meta_hints,size_hints,path="./标注.csv"):
    # for meta_hint in meta_hints:
    #     print(meta_hint)
    # for size_hint in size_hints:
    #     print(size_hint)
    # pass
    columns1=["poly_id","标注句柄","肘板几何中心点坐标","肘板类别","板厚材质信息","是否为扩散值"]
    columns2=["poly_id","标注句柄","肘板几何中心点坐标","肘板类别","标注属性","标注边","标注边句柄","参考边","参考边句柄","标注值","顺时针A或B","是否为扩散值"]
    with open(os.path.join(path,"板厚材质信息.csv"),"w",newline="",encoding="utf-8-sig") as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow(columns1)
        writer.writerows(meta_hints)
    with open(os.path.join(path,"尺寸标注信息.csv"),"w",newline="",encoding="utf-8-sig") as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow(columns2)
        writer.writerows(size_hints)

def classificationAndOutputStep(indices,edges_infos,poly_centroids,hint_infos,meta_infos,segmentation_config,polys,polyline_handles):
    classification_table = load_classification_table(segmentation_config.standard_type_path)
    poly_infos=[]
    types=[]
    flags=[]
    meta_hints=[]
    size_hints=[]
    class_count={}
    for i in range(len(indices)):
        index,edges_info,poly_centroid,hint_info,meta_info=indices[i],edges_infos[i],poly_centroids[i],hint_infos[i],meta_infos[i]
        poly_refs, classification_res,meta_hint,size_hint=outputInfo(index,edges_info,poly_centroid,hint_info,meta_info,segmentation_config,polys[index],polyline_handles)
        poly_infos.append(poly_refs)
        types.append(classification_res)
        meta_hints.extend(meta_hint)
        size_hints.extend(size_hint)
        if classification_res=="Unstandard" or "ustd" in classification_res or classification_res=="Unclassified" or len(classification_res.split(","))>1:
            flags.append(False)
        else:
            if classification_res not in class_count:
                class_count[classification_res]=0
            class_count[classification_res]+=1
            thickness=0
            if meta_info[1] is not None and meta_info[1]["Thickness"] is not None:
                thickness=meta_info[1]["Thickness"]
            free_edges,constraint_edges,edges,poly_refs,ref_map=edges_info
            template_map=match_template(edges,free_edges,classification_table[classification_res],hint_info[1],thickness)
            feature_map=hint_info[3]
            flag=True
            for key,eds in template_map.items():
                if len(eds)==0:
                    flag=False
            free_code = classification_table[classification_res]["free_code"]
            no_free_code = classification_table[classification_res]["no_free_code"]
            best_match_flag=True
            # 自由边特征比对
            free_idx = 1
            f_score=0
            while f'free{free_idx}' in template_map:
                if len(template_map[f'free{free_idx}'])==0:
                    free_idx += 1
                    continue
                seg=template_map[f'free{free_idx}'][0]
                f = feature_map[seg]
                c = free_code[free_idx - 1]
                if eva_c_f(c, f):
                    f_score += 1
                else:
                    best_match_flag = False
                free_idx += 1
            
            # 非自由边特征对比
            constarint_idx=1
            cornerhole_idx=1
            while True:
                if f'constraint{constarint_idx}' in template_map:
                    seg = template_map[f'constraint{constarint_idx}'][0]
                    f = feature_map[seg]
                    c = no_free_code[constarint_idx + cornerhole_idx - 2]
                    if eva_c_f(c, f):
                        f_score += 1
                    else:
                        best_match_flag = False
                    constarint_idx+=1
                else:
                    break
                if f'cornerhole{cornerhole_idx}' in template_map:
                    if len(template_map[f'cornerhole{cornerhole_idx}'])==0:
                        cornerhole_idx+=1
                        break
                    seg = template_map[f'cornerhole{cornerhole_idx}'][0]
                    f = feature_map[seg]
                    c = no_free_code[constarint_idx + cornerhole_idx - 2]
                    if eva_c_f(c, f):
                        f_score += 1
                    else:
                        best_match_flag = False
                    cornerhole_idx+=1   
            if best_match_flag==False:
                flag=False
            flags.append(flag)

    names=[]
    for name,value in classification_table.items():
        names.append(name)
    class_unincluded=[]
    for name in names:
        if name not in class_count:
            class_unincluded.append(name)
    for name in class_unincluded:
        log_to_file("./output/class_not_included.txt",name)
    for name ,count in class_count.items():
        log_to_file("./output/class_included.txt",name+"  "+str(count))
    outputHints(meta_hints,size_hints,segmentation_config.dxf_output_folder)
    return poly_infos,types,flags


def get_copy_info(poly_c, copy_poly_dic):
    poumian_name = None
    poumian_copy_time = 1
    for poly_co in copy_poly_dic:
        poly_co_dic = ast.literal_eval(poly_co)
        x1 = poly_co_dic["x1"]
        x2 = poly_co_dic["x2"]
        y1 = poly_co_dic["y1"]
        y2 = poly_co_dic["y2"]
        poutu_poly = [[x1, y1], [x2, y1], [x2, y2],[x1, y2]]
        center = Point(poly_c[0], poly_c[1])
        poutu_polygon = Polygon(poutu_poly)
        if center.within(poutu_polygon):
            poumian_name = copy_poly_dic[poly_co]["主图标题"]
            poumian_copy_time = copy_poly_dic[poly_co]["剖切符号调用次数"] + copy_poly_dic[poly_co]["副标题调用次数"] + 1
            return poumian_name, poumian_copy_time
    
    return poumian_name, poumian_copy_time