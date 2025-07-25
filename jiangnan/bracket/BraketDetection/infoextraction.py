from  element import *
from plot_geo import plot_geometry,plot_polys, plot_info_poly,p_minus,p_add,p_mul
import os
from utils import segment_intersection_line,segment_intersection,computeBoundingBox,is_parallel,conpute_angle_of_two_segments,point_segment_position,shrinkFixedLength,check_points_against_segments,check_points_against_free_segments,check_parallel_anno,check_vertical_anno,check_non_parallel_anno
from classifier import poly_classifier
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from bracket_parameter_extraction import parse_elbow_plate
import json
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
    convex_hull = ConvexHull(points)
    hull_area = convex_hull.volume  # ConvexHull 的面积
    # 检查面积差
    #print(poly_area,hull_area,abs(poly_area - hull_area) / hull_area)
    if abs(poly_area - hull_area) / hull_area > tolerance:
        return False

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
def combine_the_same_line(poly,segmentation_config):
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
                        for k in range(j+1,n):
                            new_poly.append(poly[k])
                        return new_poly,True
                    else:
                        ref=calculate_combined_ref([s1,s2],segmentation_config)
                        new_segment=DSegment(s2.start_point,s1.end_point,ref)
                        for k in range(i):
                            new_poly.append(poly[k])
                        new_poly.append(new_segment)
                        for k in range(j+1,n):
                            new_poly.append(poly[k])
                        return new_poly,True
                if i==0 and j==n-1:
                    if s1.end_point ==s2.start_point:
                        ref=calculate_combined_ref([s1,s2],segmentation_config)
                        new_segment=DSegment(s1.start_point,s2.end_point,ref)
                        new_poly.append(new_segment)
                        for k in range(1,n-1):
                            new_poly.append(poly[k])
                        return new_poly,True
                    else:
                        ref=calculate_combined_ref([s1,s2],segmentation_config)
                        new_segment=DSegment(s2.start_point,s1.end_point,ref)
                        new_poly.append(new_segment)
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
    #combine the same line in the poly
    # print("===================")
    # print(len(poly))
    while True:

        
        new_poly,flag=combine_the_same_line(old_poly,segmentation_config)
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
                    continue
            refs.append(segment)
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
        # elif not isinstance(first_segment.ref, DArc) and not isinstance(last_segment.ref, DArc):
        #     if is_parallel(first_segment, last_segment,segmentation_config.is_parallel_tolerance):
        #         new_segment = DSegment(
        #             start_point=last_segment.start_point,
        #             end_point=first_segment.end_point,
        #             ref=last_segment.ref
        #         )
        #         refs[0] = new_segment
        #         refs.pop()

    return refs


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
def textsInPoly(text_map,poly,segmentation_config,is_fb):
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
            if x_min <= pos.x and pos.x <=x_max and y_min <=pos.y and y_max>=pos.y:
                if t[1]["Type"]=="FB" or t[1]["Type"]=="FL":
                    result=parse_elbow_plate(t[0].content, "bottom",is_fb)
                else:
                    result=t[1]
                ts.append([t[0],pos,result,t[2]])#element,position,result,anotation
    return ts

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
    radius_anno=[]
    s_free_edges=free_edges[0]
    for r_t in r_anno:
        pos,content=r_t
        min_distance=float("inf")
        target=None
        for s in s_free_edges:
            if isinstance(s.ref,DArc):
                center=s.ref.center
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
            r_map[target]=content
            radius_anno.append(content)
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
    v1=DPoint(point1.x-point2.x,point1.y-point2.y)
    v2=DPoint(segment.start_point.x-segment.end_point.x,segment.start_point.y-segment.end_point.y)
    cross_product=(v1.x*v2.x+v1.y+v2.y)/(DSegment(point1,point2).length()*segment.length())
    if  abs(cross_product) <epsilon:
        return True
    return False 

def is_toe(free_edge,cons_edge,max_free_edge_length):
    if (free_edge.length()<56 or free_edge.length()<=0.105*max_free_edge_length) and is_vertical_(free_edge.start_point,free_edge.end_point,cons_edge,epsilon=0.35):
        return True
    return False
def is_ks_corner(free_edge,last_free_edge,cons_edge,max_free_edge_length):
    if (not is_toe(free_edge,cons_edge,max_free_edge_length)) and (not is_vertical_(free_edge.start_point,free_edge.end_point,cons_edge,epsilon=0.35)) and isinstance(last_free_edge.ref,DLine) and free_edge.length() <= 100:
        return True
    return False

def find_cons_edge(poly_refs,seg):
    for s in poly_refs:
        if (not s.isCornerhole) and (not s.isConstraint):
            continue
        if s.start_point==seg.start_point or s.start_point == seg.end_point or s.end_point ==seg.start_point or s.end_point==seg.end_point:
            return s
        
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
            
                if edge not in l_whole_map:
                    l_whole_map[edge]=[]    
                l_whole_map[edge].append((v1,v2,l))
                whole_anno.append(l)
            elif ty=="half":
                
                if edge not in l_half_map:
                    l_half_map[edge]=[]    
                l_half_map[edge].append((v1,v2,l))
                half_anno.append(l)
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
            if  is_parallel_(constraint_edge,free_edge,0.15):
                
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
                
            if is_toe(target,cons_edge,max_free_edge_length):
            
                toe_angle_anno.append(d)
            else:
                angle_anno.append(d)
    return a_map,angle_anno,toe_angle_anno


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
                return d1,d2,d3,d4
            if pos1!= "not_on_line":
                return d1,d2,d3,p_minus(p_add(d1,d3),d2)
            if pos2 !="not_on_line":
                return d1,d2,p_minus(p_add(d2,d4),d1),d4
    l1,l2=DSegment(d1,d4).length(),DSegment(d2,d3).length()
    if l1<l2:
        return d1,d2,d3,p_minus(p_add(d1,d3),d2)
    else:
        return d1,d2,p_minus(p_add(d2,d4),d1),d4


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
        return is_vertical_(v1,v2,s1,0.35)
    if v1==arc.ref.end_point or v2==arc.ref.end_point:
        return is_vertical_(v1,v2,s2,0.35)
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
def match_edge_anno(constraint_edges,free_edges,edges,all_anno,all_map):
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
    radius_anno,whole_anno,half_anno,cornor_anno,parallel_anno,non_parallel_anno,vertical_anno,d_anno,angle_anno,toe_angle_anno=all_anno
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
                if is_toe(seg,cons_edge,max_free_edge_length):
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

    #TODO
    for seg, ds in l_whole_map.items():
        for d_t in ds:
            p1,p2,d=d_t
            all_edge_map[seg]["边长标注"].append([d,f"句柄：{d.handle}，值：{d.text}，箭头起点：{p1}，箭头终点：{p2}"])
    k=0
    for key, ds in l_ver_map.items():
        start_edge,base_edge=key
        for d in ds:
            k+=1
            p1, p2 = get_anno_position(d, constraint_edges)
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
        elif edge_type[edge]=="arc":
            all_edge_map[edge]["是否相切"]=False
            all_edge_map[edge]["圆心是否在趾端延长线上"]=False
            all_edge_map[edge]["半径标注"]=[]
        elif edge_type[edge]=="toe":
            all_edge_map[edge]["边长标注"]=[]
        elif edge_type[edge]=="KS_corner":
            all_edge_map[edge]["与自由边夹角标注"]=[]
            all_edge_map[edge]["与自由边长度标注"]=[]
    
    for corner_hole in corner_holes:
        for edge in corner_hole:
            all_edge_map[edge]={}
            all_edge_map[edge]["短边尺寸标注"]=[]
            all_edge_map[edge]["半径尺寸标注"]=[]
            all_edge_map[edge]["短边是否平行于相邻边"]=False
    
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
  

    #自由边直线延长线与约束边交点标注
    for seg,ds in l_half_map.items():
        for d_t in ds:
            v1,v2,d=d_t
            for idx,free_edge in enumerate(fr_edges):
                if edge_type[free_edge]!="line":
                    continue
                if isinstance(free_edge.ref,DArc):
                    continue
                if point_segment_position(v1,free_edge,epsilon=0.2,anno=False)!="not_on_line" or point_segment_position(v2,free_edge,epsilon=0.2,anno=False)!="not_on_line":
                    all_edge_map[free_edge]["延长线与约束边交点标注"].append([d,f"句柄：{d.handle}，值：{d.text}，箭头起点：{v1}，箭头终点：{v2}，参考边：约束边",seg])
                    features.add('short_anno')
                    feature_map[free_edge].add('short_anno')
                    break
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
                #(TODO:寻找参考边)
                features.add('angl')
                all_edge_map[seg]["与约束边及其平行线夹角标注"].append([d,f"句柄：{d.handle}，值：{d.text}，参考点1：{p1}，参考点2：{p2}，参考中心： {inter}"])
                feature_map[seg].add('angl')
            elif edge_type[seg]=="KS_corner":
                #(TODO:寻找参考边)
                features.add('angl')
                feature_map[seg].add('angl')
                all_edge_map[seg]["与自由边夹角标注"].append([d,f"句柄：{d.handle}，值：{d.text}，参考点1：{p1}，参考点2：{p2}，参考中心： {inter}"])
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
            for cons_edge in cons_edges:
                if flag:
                    break
                flag=is_parallel_(seg,cons_edge,0.15)
      
            all_edge_map[seg]["是否与约束边平行"]=flag
            if flag:
                features.add('is_para')
                feature_map[seg].add('is_para')
            if free_edge_no[seg]==1 or free_edge_no[seg]==len(fr_edges):
                cons_edge=find_cons_edge(cons_edges,seg)
                flag=is_vertical_(seg.start_point,seg.end_point,cons_edge,0.005)
                all_edge_map[seg]["是否与相邻约束边夹角为90度"]=flag
                if flag:
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
                if is_near(cons_edge,short_edge) and is_parallel_(short_edge,cons_edge,0.15):
                    flag=True
                    break
            all_edge_map[edge[0]]["短边是否平行于相邻边"]=flag
            all_edge_map[edge[1]]["短边是否平行于相邻边"]=flag
    #vu角隅孔短边尺寸标注
    for seg,ds in l_cornor_map.items():
        for d_t in ds:
            v1,v2,d=d_t
            for i,start_edge in enumerate(corner_hole_start_edge):
                edge=corner_holes[i]
                if is_vu(edge):
                    short_edge=edge[0] if isinstance(edge[0].ref,DLine) else edge[1] 
                    if is_near(DSegment(v1,v2),short_edge):
                        all_edge_map[edge[0]]["短边尺寸标注"].append([d,f"句柄：{d.handle}，值：{d.text}，箭头起点：{v1}，箭头终点：{v2}"])
                        all_edge_map[edge[1]]["短边尺寸标注"].append([d,f"句柄：{d.handle}，值：{d.text}，箭头起点：{v1}，箭头终点：{v2}"])
                        # features.add('vuf')
                        break
    #自由边直线平行标注
    for key,ds in l_para_map.items():
        cons_edge,free_edge=key
        for d in ds:
            v1,v2=get_anno_position(d,constraint_edges)
            if edge_type[free_edge]=="line":
                features.add('short_anno_para')
                feature_map[free_edge].add('short_anno_para')
                all_edge_map[free_edge]["平行标注"].append([d,f"句柄：{d.handle}，值：{d.text}，箭头起点：{v1}，箭头终点：{v2}，参考边：约束边",cons_edge])
    
    #半径标注
    for seg,t in r_map.items():

        if seg in edge_type and edge_type[seg]=="arc":
            all_edge_map[seg]["半径标注"].append([t,f"句柄：{t.handle}，值：{t.content}"])
        elif seg in corner_hole_arc:
            all_edge_map[seg]["半径标注"].append([t,f"句柄：{t.handle}，值：{t.content}"])
    
    for s,fs in feature_map.items():
        feature_map[s]=list(fs)
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
def get_score(free_edges_types,non_free_edges,non_free_edges_types,free_edge_seq,non_free_edges_seq,non_free_edges_types_seq):
    new_free_edges_types=[]
    for t in free_edges_types:
        if t!="KS_corner":
            new_free_edges_types.append(t)
    new_free_edges_types_seq=[]
    for t in free_edge_seq:
        if t!="KS_corner":
            new_free_edges_types_seq.append(t)
    total=0
    for i in range(len(new_free_edges_types)):
        if new_free_edges_types[i]== new_free_edges_types_seq[i]:
            total+=1
    new_non_free_edges=[]
    new_non_free_edges_types=[]
    for i,edge in enumerate(non_free_edges):
        if non_free_edges_types[i]=="cornerhole" and len(edge)==1 and isinstance(edge[0].ref,DLine):
            continue
        new_non_free_edges.append(edge)
        new_non_free_edges_types.append(non_free_edges_types[i])
    new_non_free_edges_seq=[]
    new_non_free_edges_types_seq=[]
    for i,edge in enumerate(non_free_edges_seq):
        if non_free_edges_types_seq[i]=="cornerhole" and len(edge)==1 and edge[0]=="line":
            continue
        new_non_free_edges_seq.append(edge)
        new_non_free_edges_types_seq.append(non_free_edges_types_seq[i])
    for i in range(len(new_non_free_edges)):
        edges1=new_non_free_edges[i]
        edges2=new_non_free_edges_seq[i]
        if new_non_free_edges_types[i]==new_non_free_edges_types_seq[i] and len(edges1)==len(edges2):
            edges3=[]
            for edge in edges1:
                if isinstance(edge.ref,DArc):
                    edges3.append('arc')
                else:
                    edges3.append('line')
            if edges3==edges2:
                total+=1
    return total
def match_template(edges,detected_free_edges,template,edge_types):

    non_free_edges=[]
    non_free_edges_types=[]
    free_edges=[]
    free_edges_types=[]
    for edge in detected_free_edges[0]:
        free_edges.append(edge)
        free_edges_types.append(edge_types[edge])
    for i,edge in enumerate(edges):
        if (not edge[0].isConstraint) and (not edge[0].isCornerhole):
            continue
        if edge[0].isCornerhole:
            non_free_edges_types.append('cornerhole')
            segs=[]
            for s in edge:
                segs.append(s)
            non_free_edges.append(segs)
        else:
            non_free_edges_types.append('constraint')
            segs=[]
            for s in edge:
                segs.append(s)
            non_free_edges.append(segs)
    r_non_free_edges=non_free_edges[::-1]
    r_non_free_edges_types=non_free_edges_types[::-1]
    r_free_edges=free_edges[::-1]
    r_free_edges_types=free_edges_types[::-1]
    new_edges=[]
    for edge in r_non_free_edges:
        new_edges.append(edge[::-1])
    r_non_free_edges=new_edges
    free_edge_seq=template["free_edges"]
    non_free_seq=template["non_free_edges"]
    non_free_edges_seq=[]
    non_free_edges_types_seq=[]
    for edge in non_free_seq:
        non_free_edges_types_seq.append(edge["type"])
        non_free_edges_seq.append(edge["edges"])

    total,r_total=0,0
    
    
    total=get_score(free_edges_types,non_free_edges,non_free_edges_types,free_edge_seq,non_free_edges_seq,non_free_edges_types_seq)
    r_total=get_score(r_free_edges_types,r_non_free_edges,r_non_free_edges_types,free_edge_seq,non_free_edges_seq,non_free_edges_types_seq)
    

    if total<r_total:
        free_edges=r_free_edges
        free_edges_types=r_free_edges_types
        non_free_edges=r_non_free_edges
        non_free_edges_types=r_non_free_edges_types
    template_map={}
    j=0
    for i in range(len(free_edge_seq)):
        key=f"free{i+1}"
        if free_edge_seq[i]=='KS_corner'and j>=len(free_edges_types):
            template_map[key]=[]
        elif free_edge_seq[i]=='KS_corner'and free_edges_types[j]!='KS_corner':
            template_map[key]=[]
        else:
            template_map[key]=[free_edges[j]]
            j+=1
    cornerhole_idx=0
    constraint_idx=0
    j=0
    for i in range(len(non_free_edges_seq)):
        if non_free_edges_types_seq[i]=="cornerhole":
            cornerhole_idx+=1
            key=f"cornerhole{cornerhole_idx}"
        else:
            constraint_idx+=1
            key=f"constraint{constraint_idx}"
        if non_free_edges_types_seq[i]=="cornerhole" and len(non_free_edges_seq[i])==1 and non_free_edges_seq[i][0]=="line" and j>=len(non_free_edges_types):
            template_map[key]=[]
        elif non_free_edges_types_seq[i]=="cornerhole" and len(non_free_edges_seq[i])==1 and non_free_edges_seq[i][0]=="line" and not(non_free_edges_types[j]=="cornerhole" and len(non_free_edges[j])==1 and isinstance(non_free_edges[j][0].ref,DLine)):
            template_map[key]=[]
        else:
            template_map[key]=non_free_edges[j]
            j+=1
    return template_map
def outputPolyInfo(poly, segments, segmentation_config, point_map, index,star_pos_map,cornor_holes,texts,dimensions,text_map,stiffeners):
    # step1: 计算几何中心坐标
    poly_centroid = calculate_poly_centroid(poly)

    # step2: 合并边界线
    poly_refs = calculate_poly_refs(poly,segmentation_config)
    
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
    star_set=set()
    for s in poly:
        if s in star_pos_map:
            for ss in star_pos_map[s]:
                star_set.add(ss)
    stars_pos=list(star_set)
    # for p in stars_pos:
    #     x,y=p.x,p.y
    #     lines=[]
    #     lines.append(DSegment(DPoint(x,y),DPoint(x-5000,y)))
    #     lines.append(DSegment(DPoint(x,y),DPoint(x+5000,y)))
    #     lines.append(DSegment(DPoint(x,y),DPoint(x,y+5000)))
    #     lines.append(DSegment(DPoint(x,y),DPoint(x,y-5000)))
    #     cornor=[]
    #     for i, seg1 in enumerate(lines):
    #         dist=None
    #         s=None
    #         for j, seg2 in enumerate(poly_refs):
    #             p1, p2 = seg1.start_point, seg1.end_point
    #             q1, q2 = seg2.start_point, seg2.end_point
    #             intersection = segment_intersection(p1, p2, q1, q2)
    #             if intersection:
    #                 if dist is None:
    #                     dist=(intersection[0]-p1.x)*(intersection[0]-p1.x)+(intersection[1]-p1.y)*(intersection[1]-p1.y)
    #                     s=seg2
    #                 else:
    #                     if dist>(intersection[0]-p1.x)*(intersection[0]-p1.x)+(intersection[1]-p1.y)*(intersection[1]-p1.y):
    #                         dist=(intersection[0]-p1.x)*(intersection[0]-p1.x)+(intersection[1]-p1.y)*(intersection[1]-p1.y)
    #                         s=seg2
    #         if s is not None:
    #            cornor.append((dist,s,p))
    #     cornor=sorted(cornor,key=lambda p:p[0])
    #     if len(cornor)>=2:
    #         cornor[0][1].isConstraint=True
    #         cornor[0][1].isCornerhole=False
    #         cornor[0][1].StarCornerhole=cornor[0][2]
    #         cornor[1][1].isConstraint=True
    #         cornor[1][1].isCornerhole=False
    #         cornor[1][1].StarCornerhole=cornor[1][2]
    

   
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
                if DSegment(pair[0],pair[1]).length()<20:
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
    polygon=computePolygon(poly)
    

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
    
    tis=textsInPoly(text_map,poly,segmentation_config,is_fb)
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
        plot_info_poly(polygon,poly_refs, os.path.join(segmentation_config.poly_info_dir, f'infopoly{index}.png'),tis,ds,sfs,others)
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


    for i,edge in enumerate(edges):
        if (not edge[0].isConstraint) and (not edge[0].isCornerhole):
            #自由边
            continue
        if edge[0].isCornerhole:
            #角隅孔
            continue
        else:
            #约束边
            continue
    


    is_standard_elbow=True
    bracket_parameter={}
    thickness=0
    for t_t in tis:
        content=t_t[0].content.strip()
        if t_t[2]["Type"]=="B" and '~' in content:
            is_standard_elbow=False
    for t_t in tis:
        content=t_t[0].content.strip()
        if t_t[2]["Type"]=="B" and '~' not in content:
            bracket_parameter=t_t[2]
            if bracket_parameter["Thickness"] is not None:
                thickness=bracket_parameter["Thickness"]
    if is_standard_elbow:

        #TODO:标准肘板缺省角隅孔的添加
        # [-1] line 
        # [0] line 
        #edges poly_refs constraint_edges
        pass
    # step6: 绘制对边分类后的几何图像
    plot_info_poly(polygon,poly_refs, os.path.join(segmentation_config.poly_info_dir, f'infopoly{index}.png'),tis,ds,sfs,others)

    # for i,t_t in enumerate(tis):
    #     print(t_t[0].content)
    #标注匹配
    r_anno=[]
    l_anno=[]
    a_anno=[]
    for i,t_t in enumerate(tis):
        if t_t[2]["Type"]=="R":
            r_anno.append((t_t[1],t_t[0]))
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

    r_map,radius_anno=match_r_anno(r_anno,free_edges)
    l_whole_map,l_half_map,l_cornor_map,l_para_map,l_para_single_map,l_n_para_map,l_n_para_single_map,l_ver_map,l_ver_single_map,d_map,whole_anno,half_anno,cornor_anno,parallel_anno,non_parallel_anno,vertical_anno,d_anno=match_l_anno(l_anno,poly_refs,constraint_edges,free_edges,segmentation_config)
    a_map,angle_anno,toe_angle_anno=match_a_anno(a_anno,free_edges,constraint_edges)
    all_anno=(radius_anno,whole_anno,half_anno,cornor_anno,parallel_anno,non_parallel_anno,vertical_anno,d_anno,angle_anno,toe_angle_anno)
    all_map=(r_map,l_whole_map,l_half_map,l_cornor_map,l_para_map,l_para_single_map,l_n_para_map,l_n_para_single_map,l_ver_map,l_ver_single_map,d_map,a_map)


    all_edge_map,edge_types,features,feature_map,constraint_edge_no,free_edge_no=match_edge_anno(constraint_edges,free_edges,edges,all_anno,all_map)

    # for s,fs in feature_map.items():
    #     print(s,fs)

    if is_standard_elbow==False:
        new_all_edge_map={}
        for s,dic in all_edge_map.items():
            new_dic={}
            for ty,ds in dic.items():
                if isinstance(ds,list):
                    new_ds=[]
                    for d_t in ds:
                        if len(d_t)==3:
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
        file_path = os.path.join(segmentation_config.poly_info_dir, f'info{index}.txt')
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
                    for anno in all_edge_map[corner_hole_start_edge]["短边尺寸标注"]:
                        anno_des=f"{anno_des},{anno[1]}"
                    des=f"短边是否平行于相邻边:{all_edge_map[corner_hole_start_edge]["短边是否平行于相邻边"]};短边尺寸标注:{anno_des}"
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
                for anno in all_edge_map[seg]["半径标注"]:
                    anno_des=f"{anno_des},{anno[1]}"
                des=f"是否相切:{all_edge_map[seg]["是否相切"]}\n\t\t\t半径标注:{anno_des}\n\t\t\t圆心是否在趾端延长线上:{all_edge_map[seg]["圆心是否在趾端延长线上"]}"
            
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
                    des=f"{des};{ty}:{anno_des}"
                log_to_file(file_path, f"           标注:\n\t\t\t{des}")
            elif edge_types[seg]=="toe":  
                log_to_file(file_path, f"       起点：{seg.start_point}、终点{seg.end_point}、长度：{seg.length()}（直线）、句柄: {seg.ref.handle}")
                des=f""
                for ty,annos in all_edge_map[seg].items():
                    anno_des=""
                    for anno in annos:
                        anno_des=f"{anno_des},{anno[1]}"
                    des=f"{des};{ty}:{anno_des}"
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
        log_to_file(file_path, f"非标准肘板")
        
        return poly_refs, "Unstandard"


    cornerhole_num=0
    for i,edge in enumerate(edges):
        if (not edge[0].isConstraint) and (not edge[0].isCornerhole):
            continue
        if edge[0].isCornerhole:
            cornerhole_num+=1

    classification_res,output_template = poly_classifier(features,all_anno,poly_refs, tis,ds,cornerhole_num, free_edges, edges, thickness,feature_map,edge_types,
                                         segmentation_config.standard_type_path, segmentation_config.json_output_path, 
                                         f"{os.path.splitext(os.path.basename(segmentation_config.json_path))[0]}_infopoly{index}",
                                         is_output_json=True)
    free_order=True
    # cons_order=True
    if output_template is not None:
        template_map=match_template(edges,free_edges,output_template,edge_types)
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
                cornerhole_idx+=1

        new_all_edge_map={}
        for s,dic in all_edge_map.items():
            new_dic={}
            for ty,ds in dic.items():
                if isinstance(ds,list):
                    new_ds=[]
                    for d_t in ds:
                        if len(d_t)==3:
                            d,des,seg=d_t 
                            if seg in constranit_edge_template_no:
                                des+=str(constranit_edge_template_no[seg])
                            else:
                                des+=str(free_edge_template_no[seg])
                            new_ds.append([d,des])
                        else:
                            new_ds.append(d_t)
                    new_dic[ty]=new_ds
                else:
                    new_dic[ty]=ds
            new_all_edge_map[s]=new_dic
        all_edge_map=new_all_edge_map





        file_path = os.path.join(segmentation_config.poly_info_dir, f'info{index}.txt')
        clear_file(file_path)
        log_to_file(file_path, f"几何中心坐标：{poly_centroid}")

        free_idx=1
        log_to_file(file_path, f"自由边轮廓({get_free_edge_des(free_edges[0],edge_types)})：")
        while f'free{free_idx}' in template_map:
            if len(template_map[f'free{free_idx}'])==0:
                log_to_file(file_path, f"   自由边{free_idx}")
                log_to_file(file_path, f"        类型：KS_corner")
                log_to_file(file_path,f"         缺省")
                free_idx+=1
                continue
            seg=template_map[f'free{free_idx}'][0]

            
            log_to_file(file_path, f"   自由边{free_idx}")
            log_to_file(file_path, f"        类型：{edge_types[seg]}")

            
            if edge_types[seg]=="arc":
                log_to_file(file_path, f"       起点：{seg.ref.start_point}、终点：{seg.ref.end_point}、圆心：{seg.ref.center}、半径：{seg.ref.radius}（圆弧）、句柄: {seg.ref.handle}")
                anno_des=""
                for anno in all_edge_map[seg]["半径标注"]:
                    anno_des=f"{anno_des},{anno[1]}"
                des=f"是否相切:{all_edge_map[seg]["是否相切"]}\n\t\t\t半径标注:{anno_des}\n\t\t\t圆心是否在趾端延长线上:{all_edge_map[seg]["圆心是否在趾端延长线上"]}"
            
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
            
            free_idx+=1
        
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
        log_to_file(file_path, "边界信息（非自由边）：")
        k=1
        constarint_idx=0
        cornerhole_idx=0
        for i,edge in enumerate(template_edges):


            if len(edge)==0:
                log_to_file(file_path, f"边界颜色{k}: 缺省")
                log_to_file(file_path, f"边界轮廓{k}(line): ")
                k+=1
                log_to_file(file_path,f"    角隅孔{cornerhole_idx+1}({get_edge_des(edge)})")
                log_to_file(file_path, f"       缺省")
                cornerhole_idx+=1

            elif edge[0].isCornerhole:
                log_to_file(file_path, f"边界颜色{k}: {edge[0].ref.color}")
                log_to_file(file_path, f"边界轮廓{k}({get_edge_des(edge)}): ")
                k+=1
                corner_hole_start_edge=edge[0]
                log_to_file(file_path,f"    角隅孔{cornerhole_idx+1}({get_edge_des(edge)})")
                cornerhole_idx+=1
                if len(edge)>1 and is_vu(edge):
                    anno_des=""
                    for anno in all_edge_map[corner_hole_start_edge]["短边尺寸标注"]:
                        anno_des=f"{anno_des},{anno[1]}"
                    des=f"短边是否平行于相邻边:{all_edge_map[corner_hole_start_edge]["短边是否平行于相邻边"]}\n短边尺寸标注:{anno_des}"
                    log_to_file(file_path,f"        VU孔标注:\n\t\t\t{des}")
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
                constarint_idx+=1
                for seg in edge:
                    if isinstance(seg.ref, DArc):
                        # actual_radius= seg.ref.radius if seg not in r_map else r_map[seg].content.lstrip("R")
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
        log_to_file(file_path, f"标准肘板")
        if classification_res == "Unclassified":
            # log_to_file("./output/Unclassified.txt", f"{os.path.splitext(os.path.basename(segmentation_config.json_path))[0]}_infopoly{index}")
            return poly_refs, classification_res
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
        new_all_edge_map={}
        for s,dic in all_edge_map.items():
            new_dic={}
            for ty,ds in dic.items():
                if isinstance(ds,list):
                    new_ds=[]
                    for d_t in ds:
                        if len(d_t)==3:
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
        file_path = os.path.join(segmentation_config.poly_info_dir, f'info{index}.txt')

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
                    for anno in all_edge_map[corner_hole_start_edge]["短边尺寸标注"]:
                        anno_des=f"{anno_des},{anno[1]}"
                    des=f"短边是否平行于相邻边:{all_edge_map[corner_hole_start_edge]["短边是否平行于相邻边"]}\n短边尺寸标注:{anno_des}"
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
                for anno in all_edge_map[seg]["半径标注"]:
                    anno_des=f"{anno_des},{anno[1]}"
                des=f"是否相切:{all_edge_map[seg]["是否相切"]}\n\t\t\t半径标注:{anno_des}\n\t\t\t圆心是否在趾端延长线上:{all_edge_map[seg]["圆心是否在趾端延长线上"]}"
            
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
        log_to_file(file_path, f"标准肘板")
        return poly_refs, classification_res

    return poly_refs, classification_res
