import json 
from  element import *
import math
from plot_geo import plot_geometry,plot_polys, plot_info_poly
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import os
from sklearn.cluster import DBSCAN
import re
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed,TimeoutError
import time
from functools import partial
from itertools import combinations
from bracket_parameter_extraction import *
from shapely.geometry import Polygon,Point
import shutil
from datetime import datetime

def create_folder_safe(folder_path):
    """
    安全创建文件夹，如果已存在则重命名为.old后缀
    
    参数:
        folder_path (str): 要创建的文件夹路径
        
    返回:
        bool: 是否成功创建新文件夹
    """
    # 标准化路径
    folder_path = os.path.normpath(folder_path)
    
    # 如果文件夹不存在，直接创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"成功创建文件夹: {folder_path}")
        return True
    
    # 如果已经存在，尝试重命名
    old_folder_path = f"{folder_path}.old"
    
    # 如果.old文件夹已存在，添加时间戳
    if os.path.exists(old_folder_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        old_folder_path = f"{folder_path}.old_{timestamp}"
    
    try:
        # 重命名原有文件夹
        shutil.move(folder_path, old_folder_path)
        print(f"已重命名原有文件夹: {folder_path} -> {old_folder_path}")
        
        # 创建新文件夹
        os.makedirs(folder_path)
        print(f"成功创建新文件夹: {folder_path}")
        return True
    except Exception as e:
        print(f"操作失败: {e}")
        return False
def numberInString(content):
    flag=False
    for i in range(10):
        if str(i) in content:
            flag=True
            break
    return flag
# def is_point_in_polygon(point, polygon_edges):
    
#     polygon_points = set()  # Concave polygon example
#     for edge in polygon_edges:
#         vs,ve=edge.start_point,edge.end_point
#         polygon_points.add((vs.x,vs.y))
#         polygon_points.add((ve.x,ve.y))

#     polygon_points = list(polygon_points)

#     polygon = Polygon(polygon_points)

#     point = Point(point.x, point.y)

#     # Check if the point is inside the polygon
#     is_inside = polygon.contains(point)

#     return is_inside

def computeAreaOfPoly(poly):
    if len(poly)<3:
        return 0.0
    points=[]
    for edge in poly:
        points.append(edge.start_point)
    area=0.0
    for i in range(len(poly)-1):
        p1=points[i]
        p2=points[i+1]
        triArea=(p1.x*p2.y-p1.y*p2.x)/2
        area+=triArea
    p1=points[-1]
    p2=points[0]
    area+=(p1.x*p2.y-p1.y*p2.x)/2
    return math.fabs(area)
    
def are_equal_with_tolerance(a, b, tolerance=1e-6):
    return abs(a - b) < tolerance

def conpute_angle_of_two_segments(seg1,seg2):
   
    dx1 = seg1.end_point.x - seg1.start_point.x
    dy1 = seg1.end_point.y - seg1.start_point.y
    dx2 = seg2.end_point.x - seg2.start_point.x
    dy2 = seg2.end_point.y - seg2.start_point.y

    # 计算两个方向向量的模长
    length1 = math.sqrt(dx1**2 + dy1**2)
    length2 = math.sqrt(dx2**2 + dy2**2)
    
    
    
    # 归一化叉积
    cross_product = math.fabs((dx1 * dy2 - dy1 * dx2) / (length1 * length2+1e-4))
    if cross_product>1:
        cross_product=1

    return math.asin(cross_product)/math.pi*180
def is_parallel(seg1, seg2, tolerance=0.05):
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
    cross_product = (dx1 * dy2 - dy1 * dx2) / (length1 * length2+1e-4)
    
    # 返回是否接近0
    #print(cross_product)
    return are_equal_with_tolerance(cross_product, 0, tolerance)

def point_segment_position(point: DPoint, segment: DSegment, epsilon=0.05,anno=True):
    if point ==segment.start_point or point ==segment.end_point:
        return "on_segment"
    # 向量AB表示线段的方向
    AB = DPoint(segment.end_point.x - segment.start_point.x, segment.end_point.y - segment.start_point.y)
    # 向量AP表示从起点到点的方向
    AP = DPoint(point.x - segment.start_point.x, point.y - segment.start_point.y)
    l1,l2=DSegment(segment.start_point,point).length(),segment.length()
    # 计算叉积，判断点是否在直线上
    cross_product = (AB.x * AP.y - AB.y * AP.x)/(DSegment(segment.start_point,point).length()*segment.length()+1e-4)



     # 向量AB表示线段的方向
    BA = DPoint(segment.start_point.x - segment.end_point.x, segment.start_point.y - segment.end_point.y)
    # 向量AP表示从起点到点的方向
    BP = DPoint(point.x - segment.end_point.x, point.y - segment.end_point.y)
    # 计算叉积，判断点是否在直线上
    cross_product2 = (BA.x * BP.y - BA.y * BP.x)/(DSegment(segment.end_point,point).length()*segment.length()+1e-4)
    if abs(cross_product) > epsilon or abs(cross_product2) > epsilon:
        return "not_on_line"  # 点不在直线上

    # 计算点积，判断点是否在线段上
    dot_product = (AB.x * AP.x + AB.y * AP.y)/(DSegment(segment.start_point,point).length()*segment.length()+1e-4)
    if dot_product < 0:
        if l2>40 and l1>0.75*l2 and anno:
            return "not_on_line"
        return "before_start"  # 点在线段起点之前
    elif DSegment(segment.start_point,point).length()>segment.length():
        if l2>40 and l1>1.75*l2 and anno:
            return "not_on_line"
        return "after_end"  # 点在线段终点之后
    else:
        return "on_segment"  # 点在线段上
def point_free_segment_position(point: DPoint, segment: DSegment, epsilon=0.05):
    # 向量AB表示线段的方向
    AB = DPoint(segment.end_point.x - segment.start_point.x, segment.end_point.y - segment.start_point.y)
    # 向量AP表示从起点到点的方向
    AP = DPoint(point.x - segment.start_point.x, point.y - segment.start_point.y)
    l1,l2=DSegment(segment.start_point,point).length(),segment.length()
    if l1>4*l2:
        return "not_on_line"
    # 计算叉积，判断点是否在直线上
    cross_product = (AB.x * AP.y - AB.y * AP.x)/(DSegment(segment.start_point,point).length()*segment.length()+1e-4)
    if abs(cross_product) > epsilon:
        return "not_on_line"  # 点不在直线上

    # 计算点积，判断点是否在线段上
    dot_product = (AB.x * AP.x + AB.y * AP.y)/(DSegment(segment.start_point,point).length()*segment.length()+1e-4)
    if dot_product < 0:
        if l2>40 and l1>0.75*l2:
            return "not_on_line"
        return "before_start"  # 点在线段起点之前
    elif DSegment(segment.start_point,point).length()>segment.length():
        if l2>40 and l1>1.75*l2:
            return "not_on_line"
        return "after_end"  # 点在线段终点之后
    else:
        return "on_segment"  # 点在线段上
def point_on_segments(point,segments,epsilon=0.05):
    segs=[]
    for i,s in enumerate(segments):
        pos = point_segment_position(point, s,epsilon=0.05)
        if pos !="not_on_line":
            segs.append(s)
    new_segs=[]
    for s in segs:
        l1,l2=DSegment(s.start_point,point).length(),DSegment(s.end_point,point).length()
        l=min(l1,l2)
        if l<=2*s.length():
            new_segs.append(s)
    return new_segs

def point_on_segments_idx(point,segments,epsilon=0.05):
    segs=[]
    for i,s in enumerate(segments):
        pos = point_segment_position(point, s,epsilon=0.05)
        if pos !="not_on_line":
            segs.append((s,i))
    return segs
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
def check_whole(v1,v2,seg,s_constraint_edges,ori_cons_edges):
    idx=None
    for i ,s in enumerate(ori_cons_edges):
        if s==seg:
            idx=i
    if idx is None:
        return False
    pre,nex=idx-1,idx+1
    if pre<0:
        pre=None
    if nex>=len(ori_cons_edges):
        nex=None
    next_seg=ori_cons_edges[nex] if nex is not None else None
    last_seg=ori_cons_edges[pre] if pre is not None else None
    new_seg=seg
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
    for point in [v1,v2]:
        l1,l2=DSegment(point,new_seg.start_point).length(),DSegment(point,new_seg.end_point).length()
        l=min(l1,l2)
        if l>18.2:
            return False
    return True

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
def check_parallel_anno(point1: DPoint, point2: DPoint, constraint_edges: list[DSegment],free_edges:list[DSegment], epsilon=0.05):
    para_set={}
    cons1=point_on_segments(point1,constraint_edges,epsilon)
    cons2=point_on_segments(point2,constraint_edges,epsilon)
    free1=point_on_segments(point1,free_edges,epsilon)
    free2=point_on_segments(point2,free_edges,epsilon)
    # print(len(cons1),len(cons2),len(free1),len(free2))
    key=None
    if len(cons1)>=1 and len(cons2)==0 and len(free1)==0 and len(free2)>=1:
        key=(cons1[0],free2[0])
        for i ,s1 in enumerate(cons1):
            for j,s2 in enumerate(free2):
                if is_parallel_(s1,s2,0.1):
                    key=(s1,s2)
        
    if len(cons1)==0 and len(cons2)>=1 and len(free1)>=1 and len(free2)==0:
        key=(cons2[0],free1[0])
        for i ,s1 in enumerate(cons2):
            for j,s2 in enumerate(free1):
                if is_parallel_(s1,s2,0.1):
                    key=(s1,s2)
    return key

def is_on_free_edges(point,free_edges):
    for edge in free_edges:
        l1,l2=DSegment(edge.start_point,point).length(),DSegment(edge.end_point,point).length()
        if l1<100 or l2<100:
            return True
    return False
def nearest_free_edge(point,free_edges):
    free_edge=None
    distance=float('inf')
    for edge in free_edges:
        if isinstance(edge.ref,DArc):
            continue
        l1,l2=DSegment(edge.start_point,point).length(),DSegment(edge.end_point,point).length()
        dis=min(l1,l2)
        if distance>dis:
            free_edge=edge
            distance=dis    
    return free_edge
def check_non_parallel_anno(point1: DPoint, point2: DPoint, constraint_edges: list[DSegment],free_edges:list[DSegment], epsilon=0.05):
    cons1=point_on_segments(point1,constraint_edges,epsilon)
    cons2=point_on_segments(point2,constraint_edges,epsilon)
    if len(cons1)==1 and len(cons2)==0 and is_on_free_edges(point2,free_edges):
        free=nearest_free_edge(point2,free_edges)
        if free is None or point_segment_position(point2,free,epsilon=0.05)=="not_on_line":
            return None
        return (cons1[0],free)
    if len(cons2)==1 and len(cons1)==0 and is_on_free_edges(point1,free_edges):
        free=nearest_free_edge(point1,free_edges)
        if free is None or point_segment_position(point1,free,epsilon=0.05)=="not_on_line":
            return None
        return (cons2[0],free)
    return None
def is_vertical(point1,point2,segment,epsilon=0.05):
    v1=DPoint(point1.x-point2.x,point1.y-point2.y)
    v2=DPoint(segment.start_point.x-segment.end_point.x,segment.start_point.y-segment.end_point.y)
    cross_product=(v1.x*v2.x+v1.y*v2.y)/(DSegment(point1,point2).length()*segment.length()+1e-4)
    if  abs(cross_product) <epsilon:
        return True
    return False 
def check_vertical_anno(point1: DPoint, point2: DPoint, constraint_edges: list[DSegment], epsilon=0.05):
    cons1=point_on_segments_idx(point1,constraint_edges,epsilon)
    cons2=point_on_segments_idx(point2,constraint_edges,epsilon)
    if len(cons1)==0 or len(cons2)==0:
        return None
    if len(cons1)==1 and len(cons2)>1 and  is_vertical(point1,point2,cons1[0][0],epsilon):
        flag=False
        idx1=cons1[0][1]
        idx2=None
        for cons in cons2:
            if (cons[1]+1)%len(constraint_edges)==idx1 or (cons[1]-1+len(constraint_edges))%len(constraint_edges)==idx1:
                flag=True
                idx2=cons[1]
                break
        if flag and (not is_vertical(constraint_edges[idx2].start_point,constraint_edges[idx2].end_point,constraint_edges[idx1],epsilon)):
            key=(constraint_edges[idx2],constraint_edges[idx1])
            return key
    if len(cons2)==1 and len(cons1)>1 and is_vertical(point1,point2,cons2[0][0],epsilon):
        flag=False
        idx1=cons2[0][1]
        idx2=None
        for cons in cons1:
            if (cons[1]+1)%len(constraint_edges)==idx1 or (cons[1]-1+len(constraint_edges))%len(constraint_edges)==idx1:
                flag=True
                idx2=cons[1]
                break
        if flag and (not is_vertical(constraint_edges[idx2].start_point,constraint_edges[idx2].end_point,constraint_edges[idx1],epsilon)):
            key=(constraint_edges[idx2],constraint_edges[idx1])
            return key
    return None


def check_points_against_segments(point1: DPoint, point2: DPoint, segments: list[DSegment], epsilon=0.05):
    """
    检查两个点与线段列表中的所有线段的位置关系。
    返回是否存在一条线段满足以下条件之一：
    1. 两个点分别在线段的两端。
    2. 一个点在线段的一端，另一个点在线段上（包括端点）。
    3. 两个点都在线段的一端。

    :param point1: 第一个点
    :param point2: 第二个点
    :param segments: 线段列表
    :param epsilon: 几何误差阈值
    :return: 是否存在满足条件的线段
    """
    for i,segment in enumerate(segments):
        # 获取线段的起点和终点
        start = segment.start_point
        end = segment.end_point

        # 判断点1和点2与线段的位置关系
        pos1 = point_segment_position(point1, segment, epsilon)
        pos2 = point_segment_position(point2, segment, epsilon)
        # if i==1:
        #     print(pos1,pos2)
        #     print(point1,point2)
        #     print(start,end)
        # 条件1：两个点分别在线段的两端
        if (pos1 == "before_start" and pos2 == "after_end" ) or \
           (pos1 == "after_end" and pos2 == "before_start"):
            return "whole",i

        # 条件2：一个点在线段的一端，另一个点在线段上（包括端点）
        if (pos1 == "on_segment" and (pos2 == "before_start" or pos2=="after_end")) or \
           (pos2 == "on_segment" and (pos1 == "before_start" or pos1=="after_end")):
            return "half",i

        # 条件3：两个点都在线段的一端
        if (pos1 == "before_start"  and pos2 == "before_start") or \
           (pos1 == "after_end"  and pos2 == "after_end"):
            return "cornor",i

    # 如果遍历完所有线段都没有满足条件的，返回 False
    return None,None
def check_points_against_free_segments(point1: DPoint, point2: DPoint, segments: list[DSegment], epsilon=0.05):
    """
    检查两个点与线段列表中的所有线段的位置关系。
    返回是否存在一条线段满足以下条件之一：
    1. 两个点分别在线段的两端。
    2. 一个点在线段的一端，另一个点在线段上（包括端点）。
    3. 两个点都在线段的一端。

    :param point1: 第一个点
    :param point2: 第二个点
    :param segments: 线段列表
    :param epsilon: 几何误差阈值
    :return: 是否存在满足条件的线段
    """
    for i,segment in enumerate(segments):
        # 获取线段的起点和终点
        start = segment.start_point
        end = segment.end_point

        # 判断点1和点2与线段的位置关系
        pos1 = point_free_segment_position(point1, segment, epsilon)
        pos2 = point_free_segment_position(point2, segment, epsilon)
        # if i==1:
        #     print(pos1,pos2)
        #     print(point1,point2)
        #     print(start,end)
        # 条件1：两个点分别在线段的两端
        if (pos1 == "before_start" and pos2 == "after_end" ) or \
           (pos1 == "after_end" and pos2 == "before_start"):
            return "whole",i

        # 条件2：一个点在线段的一端，另一个点在线段上（包括端点）
        if (pos1 == "on_segment" and (pos2 == "before_start" or pos2=="after_end")) or \
           (pos2 == "on_segment" and (pos1 == "before_start" or pos1=="after_end")):
            return "half",i

        # 条件3：两个点都在线段的一端
        if (pos1 == "before_start"  and pos2 == "before_start") or \
           (pos1 == "after_end"  and pos2 == "after_end"):
            return "cornor",i

    # 如果遍历完所有线段都没有满足条件的，返回 False
    return None,None
def angleOfTwoVectors(A,B):
    lengthA = math.sqrt(A[0]**2 + A[1]**2)  
    lengthB = math.sqrt(B[0]**2 + B[1]**2)  
    dotProduct = A[0] * B[0] + A[1] * B[1]   
    cos_angle=dotProduct / (lengthA * lengthB+1e-4)
    if math.fabs(cos_angle)>1:
        if cos_angle<0:
            cos_angle=-1
        else:
            cos_angle=1
    angle = math.acos(cos_angle)
    angle_degrees = angle * (180 / math.pi)  
    return angle_degrees

def angleOfTwoSegmentsWithCommonStarter(p,a,b):
    A=a.start_point if p==a.end_point else a.end_point
    B=b.start_point if p==b.end_point else b.end_point
    return angleOfTwoVectors([A.x-p.x,A.y-p.y],[B.x-p.x,B.y-p.y])

# def isParallel(a,b,eps=1.0):
#     dx1=a.end_point.x-a.start_point.x
#     dy1=a.end_point.y-a.start_point.y
#     dx2=b.end_point.x-b.start_point.x
#     dy2=b.end_point.y-b.start_point.y
#     return math.fabs(dx1*dy2-dx2*dy1)<eps
# Ramer-Douglas-Peucker algorithm for line simplification
def rdp(points, epsilon):
    p_set=set()
    ps=[]
    for p in points:
        if p not in p_set:
            p_set.add(p)
            ps.append(p)
    return ps
    # if len(points) < 3:
    #     return points
    # # Find the point with the maximum distance from the line segment (first to last point)
    # start = points[0]
    # end = points[-1]
    # line_vec = np.array([end.x - start.x, end.y - start.y])
    # line_len = np.linalg.norm(line_vec)

    # max_dist = 0
    # index = 0

    # for i in range(1, len(points) - 1):
    #     p = points[i]
    #     vec = np.array([p.x - start.x, p.y - start.y])
    #     proj_len = np.dot(vec, line_vec) / line_len
    #     proj_point = np.array([start.x, start.y]) + proj_len * (line_vec / line_len)
    #     dist = np.linalg.norm([p.x, p.y] - proj_point)

    #     if dist > max_dist:
    #         max_dist = dist
    #         index = i

    # # If the max distance is greater than epsilon, recursively simplify
    # if max_dist > epsilon:
    #     left = rdp(points[:index + 1], epsilon)
    #     right = rdp(points[index:], epsilon)
    #     return left[:-1] + right
    # else:
    #     return [start, end]  

def process_lwpoline(vs,vs_type,color,handle,meta,isClosed,hasArc,line_type):
    #print(len(vs_type),len(vs))
    new_vs=[]
    elements=[]
    arcs=[]
    segments=[]
    arc_set=set()
    for i,v in enumerate(vs):
        if len(v)==4:
            new_vs.append([v[0], v[1]])
            new_vs.append([v[2], v[3]])        

        elif len(v)==5:
            x,y,s_a,e_a,r=v[0],v[1],v[2],v[3],v[4]
            s_a=s_a/math.pi*180
            e_a=e_a/math.pi*180
            if s_a>e_a:
                e_a+=360
            x_new = x + r * math.cos(s_a * math.pi / 180)
            y_new = y + r * math.sin(s_a * math.pi / 180)
            start_point=DPoint(x_new,y_new)
            x_new = x + r * math.cos(e_a * math.pi / 180)
            y_new = y + r * math.sin(e_a * math.pi / 180)
            end_point=DPoint(x_new,y_new)
            if len(new_vs)>0:
                prev=DPoint(new_vs[-1][0],new_vs[-1][1])
                l1,l2=DSegment(prev,start_point),DSegment(end_point,prev)
                if l1.length()<l2.length():
                    next,other=start_point,end_point
                else:
                    next,other=end_point,start_point
                if next==prev:
                    if other==prev:
                        pass
                    else:
                        new_vs.append([other.x,other.y])
                else:
                    new_vs.append([next.x,next.y])
                    new_vs.append([other.x,other.y])
            else:
                new_vs.append([start_point.x,start_point.y])
                new_vs.append([end_point.x,end_point.y])
            if r>1500:
                # line=DLine(start_point,end_point,color,handle,meta)
                # elements.append(line)
                # segments.append(DSegment(line.start_point,line.end_point,line))
                pass
            else:
                arc=DArc(DPoint(x,y),r,s_a,e_a,line_type,color,handle,meta)
                elements.append(arc)
                arcs.append(arc)
                s1,s2=DSegment(start_point,end_point,None),DSegment(end_point,start_point,None)
                if s1 not in arc_set:
                    arc_set.add(s1)
                if s2 not in arc_set:
                    arc_set.add(s2)
            # total_angle = e_a - s_a
            # num_segments = max(2, int(math.fabs(total_angle) / 45))  # 每45度一个分段
            # if r>10000 or math.fabs(total_angle)<20:
            #     num_segments=1
            # angle_step = total_angle / num_segments
            # print(total_angle,num_segments,s_a,e_a)
            # for j in range(num_segments+1):
            #     angle = s_a + j * angle_step
            #     x_new = x + r * math.cos(angle * math.pi / 180)
            #     y_new = y + r * math.sin(angle * math.pi / 180)
            #     new_vs.append([x_new,y_new])
    for i in range(len(new_vs)-1):
        s,e=DPoint(new_vs[i][0],new_vs[i][1]),DPoint(new_vs[i+1][0],new_vs[i+1][1])
        seg=DSegment(s,e,None)
        if seg.length()>0 and seg not in arc_set:
            line=DLine(s,e,line_type,color,handle,meta)
            segments.append(DSegment(line.start_point,line.end_point,line))
            elements.append(line)
    
    if isClosed:
        line=DLine(DPoint(new_vs[-1][0],new_vs[-1][1]),DPoint(new_vs[0][0],new_vs[0][1]),line_type,color,handle,meta)
        elements.append(line)
        segments.append(DSegment(line.start_point,line.end_point,line))
    #print(len(elements),len(arcs),len(segments),handle)
    return elements,arcs,segments
            
def coordinatesmap(p:DPoint,insert,scales,rotation):
    rr=rotation/180*math.pi
    cosine=math.cos(rr)
    sine=math.sin(rr)

    # x,y=(p[0]*scales[0]+100)/200,(p[1]*scales[1]+100)/200
    x,y=((cosine*p[0]*scales[0]-sine*p[1]*scales[1]))+insert[0],((sine*p[0]*scales[0]+cosine*p[1]*scales[1]))+insert[1]
    return DPoint(x,y)
def transform_point(point,meta):
    return coordinatesmap(point,meta.insert,meta.scales,meta.rotation)
def transform_elements(elements):
    for e in elements:
        if e.meta.blockName!="TOP":
            e.transform()

def segment_arc_intersection(segment: DSegment, arc: DArc):
    # Parametrize the segment
    p1 = segment.start_point
    p2 = segment.end_point
    cx, cy = arc.center.x, arc.center.y
    r = arc.radius

    dx = p2.x - p1.x
    dy = p2.y - p1.y
    a = dx**2 + dy**2
    b = 2 * (dx * (p1.x - cx) + dy * (p1.y - cy))
    c = (p1.x - cx)**2 + (p1.y - cy)**2 - r**2

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return []  # No intersection

    t1 = (-b - math.sqrt(discriminant)) / (2 * a)
    t2 = (-b + math.sqrt(discriminant)) / (2 * a)

    points = []

    for t in [t1, t2]:
        if 0 <= t <= 1:
            ix = p1.x + t * dx
            iy = p1.y + t * dy
            intersection_point = DPoint(ix, iy)

            # Check if the intersection point is on the arc
            angle = math.degrees(math.atan2(intersection_point.y - cy, intersection_point.x - cx))
            angle = (angle + 360) % 360
            start_angle = (arc.start_angle + 360) % 360
            end_angle = (arc.end_angle + 360) % 360

            if start_angle > end_angle:
                if angle >= start_angle or angle <= end_angle:
                    points.append(intersection_point)
            else:
                if start_angle <= angle <= end_angle:
                    points.append(intersection_point)

    return points

def angle_from_center(point,start_point,arc):
    if point==start_point:
        return -10
    angle = math.degrees(math.atan2(point.y - arc.center.y, point.x - arc.center.x))
    # Normalize angle to [0, 360)
    angle = (angle + 360) % 360
    return angle
def angle_from_center_cross_zero(point,st_ag,start_point,arc):
    if point==start_point:
        return -10
    ag=angle_from_center(point,start_point,arc)
    if ag<st_ag:
        ag+=360
    return ag
# Function to sort points by angle on the arc
def sort_points_on_arc(points, arc):
    start_point, end_point = arc.points_on_arc()
    start_angle = (arc.start_angle + 360) % 360
    end_angle = (arc.end_angle+ 360) % 360

    if start_angle > end_angle:
        # If the arc crosses 0 degrees, handle the wrap-around
        return sorted(points, key=lambda pt: angle_from_center_cross_zero(pt,start_angle,start_point,arc))
    else:
        return sorted(points, key=lambda pt: angle_from_center(pt,start_point,arc))

def shrinkFixedLength(segList,dist):
    

    new_seglist=[] 
    n=len(segList)
    # if verbose:
    #     pbar=tqdm(total=n,desc="Preprocess")
    for seg in segList:
        # if verbose:
        #     pbar.update()
        p1=seg[0]
        p2=seg[1]
        v=(p2[0]-p1[0],p2[1]-p1[1])
        l=math.sqrt(v[0]*v[0]+v[1]*v[1])

       
        
        # elif l<=dist:
        #     l*=1.5
        v=(v[0]/l*dist,v[1]/l*dist)
        if l <=2*dist:
            l*=2
        vs=DPoint(p1[0]+v[0],p1[1]+v[1]) 
        ve=DPoint(p2[0]-v[0],p2[1]-v[1])
        new_seglist.append(DSegment(vs,ve,seg.ref))
    # if verbose:
    #     pbar.close()
    return new_seglist
#expand lines by fixed length
def expandFixedLength(segList,dist,both=True,verbose=True,ignore_length=False):


    new_seglist=[] 
    n=len(segList)
    # if verbose:
    #     pbar=tqdm(total=n,desc="Preprocess")
    for seg in segList:
        # if verbose:
        #     pbar.update()
        p1=seg[0]
        p2=seg[1]
        v=(p2[0]-p1[0],p2[1]-p1[1])
        l=math.sqrt(v[0]*v[0]+v[1]*v[1])
        if l<0.25:
            continue
        if l<=dist and ignore_length==False:
            l=l*dist/0.5
        
        # elif l<=dist:
        #     l*=1.5

        v=(v[0]/l*dist,v[1]/l*dist)
        vs=DPoint(p1[0]-v[0],p1[1]-v[1]) if both else DPoint(p1[0],p1[1])
        ve=DPoint(p2[0]+v[0],p2[1]+v[1])
        new_seglist.append(DSegment(vs,ve,seg.ref))
    # if verbose:
    #     pbar.close()
    return new_seglist
# Function to split arcs using segments
def split_arcs(arcs, segments):
    split_segments = []
    pbar=tqdm(desc="split arcs",total=len(arcs)*len(segments))
    for arc in arcs:
        start_point, end_point = arc.points_on_arc()
        intersection_points = [start_point,end_point]

        for segment in segments:
            pbar.update()
            intersection_points.extend(segment_arc_intersection(segment, arc))

        # Remove duplicates and sort by angle
        intersection_points = list(set(intersection_points))
        intersection_points=sort_points_on_arc(intersection_points, arc)


         # Convert to segments

        for i in range(len(intersection_points) - 1):
            sp, ep = intersection_points[i], intersection_points[i + 1]

            # Calculate angles of the segment's points on the arc
            sp_angle = math.degrees(math.atan2(sp.y - arc.center.y, sp.x - arc.center.x))
            ep_angle = math.degrees(math.atan2(ep.y - arc.center.y, ep.x - arc.center.x))

            sp_angle = (sp_angle + 360) % 360
            ep_angle = (ep_angle + 360) % 360

            if sp_angle > ep_angle:
                ep_angle += 360
            delta_angle=ep_angle-sp_angle
            num_subdivisions = max(2, int((ep_angle - sp_angle) / 45))  # At least 2 segments, split every 10 degrees
            if delta_angle<30:
                num_subdivisions=1
            for j in range(num_subdivisions):
                t1 = j / num_subdivisions
                t2 = (j + 1) / num_subdivisions

                sub_sp_angle = sp_angle + t1 * (ep_angle - sp_angle)
                sub_ep_angle = sp_angle + t2 * (ep_angle - sp_angle)

                sub_sp = DPoint(arc.center.x + arc.radius * math.cos(math.radians(sub_sp_angle)),
                                arc.center.y + arc.radius * math.sin(math.radians(sub_sp_angle)))
                sub_ep = DPoint(arc.center.x + arc.radius * math.cos(math.radians(sub_ep_angle)),
                                arc.center.y + arc.radius * math.sin(math.radians(sub_ep_angle)))

                split_segments.append(DSegment(sub_sp, sub_ep,arc))
        
    pbar.close()
    return split_segments

def transform_segments(segments,scales,rotation,insert):
    new_segments=[]
    for segment in segments:
        vs,ve=segment.start_point,segment.end_point
        vs=coordinatesmap(vs,insert,scales,rotation)
        ve=coordinatesmap(ve,insert,scales,rotation)
        segment=DSegment(vs,ve,segment.ref)
        new_segments.append(segment)
    return new_segments
#json --> elements
def process_block(T_is_contained,block_datas,blockName,scales,rotation,insert,block_linetype,attribs,bound,block_elements,segmentation_config):
    print(f"正在处理块:{blockName}")
    # if blockName!="L125X75X7" or insert[1]>-36900:
    #     return [],[],[]
    filtered_block_elements=[]
    if segmentation_config.bracket_layer is  None:
        filtered_block_elements=block_elements
    else:
        for element in block_elements:
            if element["layerName"]==segmentation_config.bracket_layer:
                filtered_block_elements.append(element)
    block_elements=filtered_block_elements
    block_meta_data=DInsert(blockName,scales,rotation,insert,attribs,bound)
    #print(blockName,scales,rotation,insert,attribs,block_elements)
    elements=[]
    arcs=[]
    segments=[]
    stiffeners=[]
    arc_splits=[]
    ori_segments=[]
    for attrib in attribs:
        text=attrib["attribText"]
        if is_useful_text(text.strip()):
            elements.append(DText(bound={"x1":-50,"x2":50,"y1":-50,"y2":50},content=text,handle=attrib["attribHandle"],meta=block_meta_data))
    color = segmentation_config.color
    linetype =segmentation_config.remove_linetype
    elementtype=segmentation_config.element_type
    layname=segmentation_config.remove_layername
    for ele in block_elements:
        if  (T_is_contained and ele.get("layerName") is not None and ele["layerName"]!="T") or (T_is_contained and ele.get("layerName") is None) :
            continue
        if ele["type"]=="line":
           
            ele_linetype=ele["linetype"].upper()
            if ele_linetype =="BYBLOCK":
                ele_linetype=block_linetype.upper()
            
            if ele.get("layerName") is not None and ele["layerName"] in layname:
                if ele["layerName"] in segmentation_config.stiffener_name:
                    e=DLine(DPoint(ele["start"][0],ele["start"][1]),DPoint(ele["end"][0],ele["end"][1]),ele_linetype,ele["color"],ele["handle"],meta=block_meta_data)
                    stiffeners.append(DSegment(e.start_point,e.end_point,e))
                if ele["layerName"] in segmentation_config.remove_layername:
                    continue
            # 虚线过滤
            # if ele.get("linetype") is not None and ele["linetype"] in linetype:
            #     continue
            # 颜色过滤
            if ele["color"]  in color:
                continue
            e=DLine(DPoint(ele["start"][0],ele["start"][1]),DPoint(ele["end"][0],ele["end"][1]),ele_linetype,ele["color"],ele["handle"],meta=block_meta_data)
            elements.append(e)
            segments.append(DSegment(e.start_point,e.end_point,e))
        elif ele["type"] == "arc":
            # 颜色过滤
            # if ele["color"] not in color:
            #     continue
            ele_linetype=ele["linetype"].upper()
            if ele_linetype =="BYBLOCK":
                ele_linetype=block_linetype.upper()
            if ele.get("layerName") is not None and ele["layerName"] in layname:
                if ele["layerName"] in segmentation_config.stiffener_name:
                    e = DArc(DPoint(ele["center"][0], ele["center"][1]), ele["radius"], ele["startAngle"], ele["endAngle"],ele_linetype,ele["color"],ele["handle"],meta=block_meta_data)
                    stiffeners.append(DSegment(e.start_point,e.end_point,e))
                if ele["layerName"] in segmentation_config.remove_layername:
                    continue
            if ele["color"]  in color:
                continue
            # 虚线过滤
            # if ele.get("linetype") is not None and ele["linetype"] in linetype:
            #     continue
            # 创建DArc对象
            e = DArc(DPoint(ele["center"][0], ele["center"][1]), ele["radius"], ele["startAngle"], ele["endAngle"],ele_linetype,ele["color"],ele["handle"],meta=block_meta_data)

            elements.append(e)
            arcs.append(e)
            continue
        elif ele["type"]=="spline":
            # 颜色过滤
            # if ele["color"] not in color:
            #     continue
            # 虚线过滤
            # if ele.get("linetype") is None or ele["linetype"] not in linetype:
            #     continue
            
            # 虚线过滤
            # if ele.get("linetype") is not None and ele["linetype"] in linetype:
            #     continue
            vs = ele["vertices"]
            ps = [DPoint(v[0], v[1]) for v in vs]

            # Apply line simplification
            simplified_ps = rdp(ps, epsilon=5.0)  # Adjust epsilon for simplification level

            ele_linetype=ele["linetype"].upper()
            if ele_linetype =="BYBLOCK":
                ele_linetype=block_linetype.upper()
            if ele.get("layerName") is not None and ele["layerName"] in layname:
                if ele["layerName"] in segmentation_config.stiffener_name:
                    l = len(simplified_ps)
                    for i in range(l - 1):
                        # if simplified_ps[i].y>-48500 or simplified_ps[i+1].y>-48500:
                        e=DLine(simplified_ps[i], simplified_ps[i + 1],ele_linetype,ele["color"],ele["handle"],meta=block_meta_data)
                        stiffeners.append(DSegment(simplified_ps[i], simplified_ps[i + 1], e))
                    
                if ele["layerName"] in segmentation_config.remove_layername:
                    continue
            if ele["color"]  in color:
                continue
            l = len(simplified_ps)
            for i in range(l - 1):
                # if simplified_ps[i].y>-48500 or simplified_ps[i+1].y>-48500:
                e=DLine(simplified_ps[i], simplified_ps[i + 1],ele_linetype,ele["color"],ele["handle"],meta=block_meta_data)
                segments.append(DSegment(simplified_ps[i], simplified_ps[i + 1], e))
                elements.append(e)
        elif ele["type"]=="lwpolyline" :
                # 颜色过滤
            # if ele["color"] not in color:
            #     continue
            # 虚线过滤
            # if ele.get("linetype") is None or ele["linetype"] not in linetype:
            #     continue
            # 虚线过滤
            # if ele.get("linetype") is not None and ele["linetype"] in linetype:
            #     continue
            vs = ele["vertices"]
            vs_type=ele["verticesType"]
            ele_linetype=ele["linetype"].upper()
            if ele_linetype =="BYBLOCK":
                ele_linetype=block_linetype.upper()
            lwe,lwa,lws=process_lwpoline(vs,vs_type,ele["color"],ele["handle"],block_meta_data,ele["isClosed"],ele["hasArc"],ele_linetype)
            if ele.get("layerName") is not None and ele["layerName"] in layname:
                if ele["layerName"] in segmentation_config.stiffener_name:
                    
                    for s in lws:
                    # if simplified_ps[i].y>-48500 or simplified_ps[i+1].y>-48500:
                        stiffeners.append(s)
                    for s in lwa:
                        stiffeners.append(DSegment(s.start_point,s.end_point,s))
                if ele["layerName"] in segmentation_config.remove_layername:
                    continue
            
            if ele["color"]  in color:
                continue
            elements.extend(lwe)
            arcs.extend(lwa)
            segments.extend(lws)
            
            #ps = [DPoint(v[0], v[1]) for v in vs]

            # Apply line simplification
            #simplified_ps = rdp(ps, epsilon=5.0)  # Adjust epsilon for simplification level

            #e = DLwpolyline(simplified_ps, ele["color"], ele["isClosed"],ele["handle"],True,ele["verticesType"],ele["vertices"],ele["hasArc"],meta=block_meta_data)
            #elements.append(e)
            #l = len(simplified_ps)
            #for i in range(l - 1):
                # if simplified_ps[i].y>-48500 or simplified_ps[i+1].y>-48500:
                #segments.append(DSegment(simplified_ps[i], simplified_ps[i + 1], e))
            #if ele["isClosed"]:
                # if simplified_ps[-1].y>-48500 or simplified_ps[0].y>-48500:
                #segments.append(DSegment(simplified_ps[-1], simplified_ps[0], e))
        elif  ele["type"]=="polyline":

            # 颜色过滤
            # if ele["color"] not in color:
            #     continue
            # 虚线过滤
            # if ele.get("linetype") is None or ele["linetype"] not in linetype:
            #     continue
            # 虚线过滤
            # if ele.get("linetype") is not None and ele["linetype"] in linetype:
            #     continue
            vs = ele["vertices"]
            ps = [DPoint(v[0], v[1]) for v in vs]

            # Apply line simplification
            simplified_ps = rdp(ps, epsilon=5.0)  # Adjust epsilon for simplification level
            #print(simplified_ps)
            ele_linetype=ele["linetype"].upper()
            if ele_linetype =="BYBLOCK":
                ele_linetype=block_linetype.upper()

            if ele.get("layerName") is not None and ele["layerName"] in layname:
                if ele["layerName"] in segmentation_config.stiffener_name:
                    
                    # e = DLwpolyline(simplified_ps,ele_linetype, ele["color"], ele["isClosed"],ele["handle"],meta=block_meta_data)
                    l = len(simplified_ps)
                    for i in range(l - 1):
                        # if simplified_ps[i].y>-48500 or simplified_ps[i+1].y>-48500:
                        e=DLine(simplified_ps[i], simplified_ps[i + 1],ele_linetype,ele["color"],ele["handle"],meta=block_meta_data)
                        stiffeners.append(DSegment(simplified_ps[i], simplified_ps[i + 1], e))
                    if ele["isClosed"]:
                        # if simplified_ps[-1].y>-48500 or simplified_ps[0].y>-48500:
                        e=DLine(simplified_ps[-1], simplified_ps[0],ele_linetype,ele["color"],ele["handle"],meta=block_meta_data)
                        stiffeners.append(DSegment(simplified_ps[-1], simplified_ps[0], e))
                if ele["layerName"] in segmentation_config.remove_layername:
                    continue
            if ele["color"]  in color:
                continue
            #e = DLwpolyline(simplified_ps,ele_linetype, ele["color"], ele["isClosed"],ele["handle"],meta=block_meta_data)
            # elements.append(e)
            l = len(simplified_ps)
            for i in range(l - 1):
                # if simplified_ps[i].y>-48500 or simplified_ps[i+1].y>-48500:
                e=DLine(simplified_ps[i], simplified_ps[i + 1],ele_linetype,ele["color"],ele["handle"],meta=block_meta_data)
                segments.append(DSegment(simplified_ps[i], simplified_ps[i + 1], e))
                elements.append(e)
            if ele["isClosed"]:
                # if simplified_ps[-1].y>-48500 or simplified_ps[0].y>-48500:
                e=DLine(simplified_ps[-1], simplified_ps[0],ele_linetype,ele["color"],ele["handle"],meta=block_meta_data)
                segments.append(DSegment(simplified_ps[-1], simplified_ps[0], e))
                elements.append(e)
        elif ele["type"]=="insert":
                if ele.get("layerName") is not None and ele["layerName"] in layname:
                    continue
                if  "color" in ele and ele.get("color") is not None and ele["color"]  in color:
                    continue
                # 虚线过滤
                # if ele.get("linetype") is not None and ele["linetype"] in linetype:
                #     continue
                sub_blockName=ele["blockName"]
                # x1,x2,y1,y2=ele["bound"]["x1"],ele["bound"]["x2"],ele["bound"]["y1"],ele["bound"]["y2"]
                sub_scales=ele["scales"]
                sub_rotation=ele["rotation"]
                sub_insert=ele["insert"]
                sub_attribs=ele['attribs']
                sub_block_data=block_datas[sub_blockName]
                sub_bound=ele["bound"]
                ele_linetype=ele["linetype"].upper() if ele.get("linetype") is not None else "CONTINUOUS"
                if ele_linetype =="BYBLOCK":
                    ele_linetype=block_linetype.upper()
                #pre-check
                sub_T_is_contained=False
                for sube in sub_block_data:
                    if sube.get("layerName") is not None and sube["layerName"]=="T":
                        sub_T_is_contained=True
                        break
                sub_elements,sub_segments,sub_arc_splits,sub_ori_segments,sub_stiffeners=process_block(sub_T_is_contained,block_datas,sub_blockName,sub_scales,sub_rotation,sub_insert,ele_linetype,sub_attribs,sub_bound,sub_block_data,segmentation_config)
                for sube in sub_elements:
                    sube.meta=block_meta_data
                elements.extend(sub_elements)
                segments.extend(sub_segments)
                arc_splits.extend(sub_arc_splits)
                ori_segments.extend(sub_ori_segments)
                stiffeners.extend(sub_stiffeners)
        elif ele["type"]=="text":
                if ele.get("layerName") is not None and ele["layerName"] in layname:
                    continue
                # 虚线过滤
                # if ele.get("linetype") is not None and ele["linetype"] in linetype:
                #     continue
                content=ele["content"].strip()
                for ct in content.split("/"):
                    if ct.strip()!="":
                        if "R" in ct.strip():
                            e=DText(ele["bound"],[ele["insert"][0],ele["insert"][1]-5], ele["color"],ct.strip(),ele["height"],ele["handle"],ele["rotation"],meta=block_meta_data)
                            elements.append(e)
                        else:
                            e=DText(ele["bound"],ele["insert"], ele["color"],ct.strip(),ele["height"],ele["handle"],ele["rotation"],meta=block_meta_data)
                            elements.append(e)           
        elif  ele["type"]=="mtext":
            if ele.get("layerName") is not None and ele["layerName"] in layname:
                    continue
            # 虚线过滤
            # if ele.get("linetype") is not None and ele["linetype"] in linetype:
            #     continue
            string = ele["text"].strip()
            # cleaned_string = re.sub(r"^\\A1;", "", string)
            cleaned_string=string
            for ct in cleaned_string.split("/"):
                if ct.strip()!="":
                    if "R" in ct.strip():
                        e=DText(ele["bound"],[ele["insert"][0],ele["insert"][1]-5], ele["color"],ct.strip(),ele["width"],ele["handle"],ele["rotation"],meta=block_meta_data,is_mtext=True)
                        elements.append(e)
                    else:
                        e=DText(ele["bound"],ele["insert"], ele["color"],ct.strip(),ele["width"],ele["handle"],ele["rotation"],meta=block_meta_data,is_mtext=True)
                        elements.append(e)

        elif ele["type"]=="dimension":
            if ele.get("layerName") is not None and ele["layerName"] in layname:
                continue
            # 虚线过滤
            # if ele.get("linetype") is not None and ele["linetype"] in linetype:
            #     continue
            textpos=ele["textpos"]
            defpoints=[]
            for i in range(5):
                k="defpoint"+str(i+1)
                if k in ele:
                    defpoint=ele[k]
                    defpoints.append(DPoint(defpoint[0],defpoint[1]))
                else:
                    break
            e=DDimension(DPoint(textpos[0],textpos[1]),ele["color"],ele["text"].strip(),ele["measurement"],defpoints,ele["dimtype"],ele["handle"],meta=block_meta_data)
            elements.append(e)
        else:
            pass
    
    
    for s in segments:
        if s.length()>0:
            ori_segments.append(DSegment(s.start_point,s.end_point,s.ref))
        else:
            # print(s)
            pass
    expand_segments=expandFixedLength(segments,segmentation_config.line_expand_length)
    arc_splits.extend(split_arcs(arcs,expand_segments))
    for s in arc_splits:
        if s.length()>0:
            ori_segments.append(DSegment(s.start_point,s.end_point,s.ref))
        else:
            # print(s)
            pass
    # arc_splits=expandFixedLength(arc_splits,segmentation_config.arc_expand_length)
    # segments=segments+arc_splits

    segments=transform_segments(segments,scales,rotation,insert)
    arc_splits=transform_segments(arc_splits,scales,rotation,insert)
    ori_segments=transform_segments(ori_segments,scales,rotation,insert)
    stiffeners=transform_segments(stiffeners,scales,rotation,insert)
    transform_elements(elements)
    new_segments=[]
    new_arc_splits=[]
    new_ori_segments=[]
    new_stiffeners=[]
    x1,x2,y1,y2=bound["x1"]-20,bound["x2"]+20,bound["y1"]-20,bound["y2"]+20
    for s in segments:
        ps,pe=s.start_point,s.end_point
        if ps.x>=x1 and ps.x <=x2 and ps.y>=y1 and ps.y<=y2 and pe.x>=x1 and pe.x <=x2 and pe.y>=y1 and pe.y<=y2:
            new_segments.append(s)
    for s in arc_splits:
        ps,pe=s.start_point,s.end_point
        if ps.x>=x1 and ps.x <=x2 and ps.y>=y1 and ps.y<=y2 and pe.x>=x1 and pe.x <=x2 and pe.y>=y1 and pe.y<=y2:
            new_arc_splits.append(s)
    for s in ori_segments:
        ps,pe=s.start_point,s.end_point
        if ps.x>=x1 and ps.x <=x2 and ps.y>=y1 and ps.y<=y2 and pe.x>=x1 and pe.x <=x2 and pe.y>=y1 and pe.y<=y2:
            new_ori_segments.append(s)
    for s in stiffeners:
        ps,pe=s.start_point,s.end_point
        if ps.x>=x1 and ps.x <=x2 and ps.y>=y1 and ps.y<=y2 and pe.x>=x1 and pe.x <=x2 and pe.y>=y1 and pe.y<=y2:
            new_stiffeners.append(s)



    return elements,new_segments,new_arc_splits,new_ori_segments,new_stiffeners
# def process_blocks(block_sub_datas,segmentation_config):
#     elements=[]
#     segments=[]
#     stiffeners=[]
#     ori_segments=[]
#     for block_sub_data in block_sub_datas:
#         blockName,scales,rotation,insert,attribs,bound,block_elements=block_sub_data[0],block_sub_data[1],block_sub_data[2],block_sub_data[3],block_sub_data[4],block_sub_data[5],block_sub_data[6]
#         block_e,block_s,block_o,block_sf=process_block(blockName,scales,rotation,insert,attribs,bound,block_elements,segmentation_config)
#         elements.extend(block_e)
#         segments.extend(block_s)
#         ori_segments.extend(block_o)
#         stiffeners.extend(block_sf)
#     return elements,segments,ori_segments,stiffeners
def readJson(path,segmentation_config):
    # elements=[]
    block_sub_datas=[]
    # arcs=[]
    # segments=[]
    # color = [3, 7, 8, 4,2,140]
    # linetype = ["BYLAYER", "Continuous","Bylayer","CONTINUOUS","ByBlock","BYBLOCK"]
    # elementtype=["line","arc","lwpolyline","polyline","spline"]
    # layname=["Stiffener_Invisible"]
    try:  
        with open(path, 'r', encoding='utf-8') as file:  
            data_list = json.load(file)
        block_datas=data_list[1]
        elements,segments,arc_splits,ori_segments,stiffeners=process_block(False,block_datas,"TOP",[1.0,1.0],0,[0,0],"CONTINUOUS",[],{"x1":float("-inf"),"x2":float("inf"),"y1":float("-inf"),"y2":float("inf")},data_list[0],segmentation_config)
        new_segments=[]
        new_arc_splits=[]
        new_ori_segments=[]
        for s in segments:
            if s.ref.linetype in segmentation_config.line_type:
                new_segments.append(s)
        for s in arc_splits:
            if s.ref.linetype in segmentation_config.line_type:
                new_arc_splits.append(s)
        for s in ori_segments:
            if s.ref.linetype in segmentation_config.line_type:
                new_ori_segments.append(s)
        segments=new_segments
        arc_splits=new_arc_splits
        ori_segments=new_ori_segments
        segments=expandFixedLength(segments,segmentation_config.line_expand_length)
        arc_splits=expandFixedLength(arc_splits,segmentation_config.arc_expand_length)
        sign_handles=[]
        polyline_handles=[]
        hatch_polys = []
        for ele in data_list[0]:
            if ele["type"]=="lwpolyline":
                vs=ele["vertices"]
                vs_type=ele["verticesType"]
                vs_width=ele["verticesWidth"]
                if len(vs)==3 and len(vs_type)==3 and vs_type==["line","line","line"] and len(vs_width)==4 and vs_width[1]==[0,0] and vs_width[3]==[0,0] and vs_width[0][0]==0 and vs_width[0][1]>0 and vs_width[2][0]>0 and vs_width[2][1]==0:
                    start=DPoint(vs[0][0],vs[0][1])
                    end=DPoint(vs[-1][2],vs[-1][3])
                    if DSegment(start,end).length()>100 and DSegment(start,end).length() <1000:
                        sign_handles.append(ele["handle"])
                elif len(vs)==5 and len(vs_type)==5 and vs_type==["line","line","line","line","line"] and len(vs_width)==6 and vs_width[0]==[0,0] and vs_width[2]==[0,0] and vs_width[4]==[0,0] and vs_width[5]==[0,0] and vs_width[3][0]==0 and vs_width[3][1]>0 and vs_width[1][0]>0 and vs_width[1][1]==0:
                    start=DPoint(vs[0][0],vs[0][1])
                    end=DPoint(vs[-1][2],vs[-1][3])
                    if DSegment(start,end).length()>100 and DSegment(start,end).length() <1000:
                        sign_handles.append(ele["handle"])
            elif ele["type"]=="polyline":
                polyline_handles.append(ele["handle"])
            elif ele["type"]=="hatch":
                hatch_paths = ele["paths"]
                for hatch_edges in hatch_paths:
                    hatch_poly = []
                    for hatch_edge in hatch_edges:
                        if hatch_edge["edge_type"] == "line":
                            point = [[hatch_edge["coords"][0], hatch_edge["coords"][1]], [hatch_edge["coords"][2], hatch_edge["coords"][3]]]
                            hatch_poly.extend(point)
                        elif hatch_edge["edge_type"] == "polyline":
                            hatch_poly.extend(hatch_edge["coords"])
                    hatch_polys.append(hatch_poly)
       

        #inner block 
        jg_s = set()
        for name,sub_data in  data_list[1].items():
            st_prefixs = ["L", "HP"]
            if any(name.startswith(prefix) for prefix in st_prefixs):
                for ele in sub_data:
                    jg_s.add(ele["handle"])


            for ele in sub_data:
                if ele["type"]=="lwpolyline":
                    vs=ele["vertices"]
                    vs_type=ele["verticesType"]
                    vs_width=ele["verticesWidth"]
                    if len(vs)==3 and len(vs_type)==3 and vs_type==["line","line","line"] and len(vs_width)==4 and vs_width[1]==[0,0] and vs_width[3]==[0,0] and vs_width[0][0]==0 and vs_width[0][1]>0 and vs_width[2][0]>0 and vs_width[2][1]==0:
                        start=DPoint(vs[0][0],vs[0][1])
                        end=DPoint(vs[-1][2],vs[-1][3])
                        if DSegment(start,end).length()>100 and DSegment(start,end).length() <1000:
                            sign_handles.append(ele["handle"])
                    elif len(vs)==5 and len(vs_type)==5 and vs_type==["line","line","line","line","line"] and len(vs_width)==6 and vs_width[0]==[0,0] and vs_width[2]==[0,0] and vs_width[4]==[0,0] and vs_width[5]==[0,0] and vs_width[3][0]==0 and vs_width[3][1]>0 and vs_width[1][0]>0 and vs_width[1][1]==0:
                        start=DPoint(vs[0][0],vs[0][1])
                        end=DPoint(vs[-1][2],vs[-1][3])
                        if DSegment(start,end).length()>100 and DSegment(start,end).length() <1000:
                            sign_handles.append(ele["handle"])


        return elements,segments+arc_splits,ori_segments,stiffeners,sign_handles,polyline_handles,hatch_polys,jg_s
    except FileNotFoundError:  
        print("The file does not exist.")
    except json.JSONDecodeError:  
        print("Error decoding JSON.")

#remove duplicate points on the same edge
def remove_duplicates(input_list):  
    seen = set()  
    result = []  
    for item in input_list:  
        if item not in seen:  
            seen.add(item)  
            result.append(item)  
    return result

# Helper function to compute intersection between two segments
def segment_intersection(p1, p2, q1, q2, epsilon=1e-9):
    """ Returns the intersection point between two line segments, or None if they don't intersect. """

    def cross_product(v1, v2):
        return v1[0] * v2[1] - v1[1] * v2[0]

    r = (p2.x - p1.x, p2.y - p1.y)
    s = (q2.x - q1.x, q2.y - q1.y)

    denominator = cross_product(r, s)

    if abs(denominator) < epsilon:  # Parallel or collinear
        return None

    t = cross_product((q1.x - p1.x, q1.y - p1.y), s) / denominator
    u = cross_product((q1.x - p1.x, q1.y - p1.y), r) / denominator

    if -epsilon <= t <= 1+epsilon and -epsilon <= u <= 1+epsilon:  # Intersection occurs within both segments
        intersect_x = p1.x + t * r[0]
        intersect_y = p1.y + t * r[1]
        return DPoint(intersect_x, intersect_y)
    
    return None
def segment_intersection_line(p1, p2, q1, q2, epsilon=1e-9):
    """ Returns the intersection point between two line segments, or None if they don't intersect. """

    def cross_product(v1, v2):
        return v1[0] * v2[1] - v1[1] * v2[0]

    r = (p2.x - p1.x, p2.y - p1.y)
    s = (q2.x - q1.x, q2.y - q1.y)

    denominator = cross_product(r, s)

    if abs(denominator) < epsilon:  # Parallel or collinear
        return None

    t = cross_product((q1.x - p1.x, q1.y - p1.y), s) / denominator
    u = cross_product((q1.x - p1.x, q1.y - p1.y), r) / denominator

    
    intersect_x = p1.x + t * r[0]
    intersect_y = p1.y + t * r[1]
    return DPoint(intersect_x, intersect_y)

    
# Function to find all intersections
# def find_all_intersections(segments, epsilon=1e-9):
#     intersection_dict = {}
#     n=len(segments)
#     pbar=tqdm(total=n*(n-1)/2,desc="计算交点")
#     for i, seg1 in enumerate(segments):
#         for j, seg2 in enumerate(segments):
#             if i >= j :
#                 continue  # Avoid duplicate checks and self-intersections

#             p1, p2 = seg1.start_point, seg1.end_point
#             q1, q2 = seg2.start_point, seg2.end_point
#             intersection = segment_intersection(p1, p2, q1, q2, epsilon)
#             if intersection:
#                 if seg1 not in intersection_dict:
#                     intersection_dict[seg1] = []
#                 if seg2 not in intersection_dict:
#                     intersection_dict[seg2] = []
                
#                 # Append the intersection for both segments
#                 intersection_dict[seg1].append(intersection)
#                 intersection_dict[seg2].append(intersection)
#             pbar.update()
#     # Sort intersections along each segment by their distance from the start point
#     for seg, isects in intersection_dict.items():
        
#         isects.sort(key=lambda p: (p.x - seg.start_point.x)**2 + (p.y - seg.start_point.y)**2)
#         intersection_dict[seg]=isects
#     pbar.close()
#     return intersection_dict

# Function to compute intersections for two subsets of segments
def compute_intersections(chunk1, chunk2, epsilon=1e-9):
    intersection_dict = {}
    n=len(chunk1)
    pbar=tqdm(total=n,desc="计算交点")
    for seg1 in chunk1:
        pbar.update()
        chunck=chunk2.segments_near_segment(seg1)
        for seg2 in chunck:
            
            if seg1 == seg2 :
                continue  # Skip self-intersection
            
            p1, p2 = seg1.start_point, seg1.end_point
            q1, q2 = seg2.start_point, seg2.end_point
            intersection = segment_intersection(p1, p2, q1, q2, epsilon)
            if intersection:
                if seg1 not in intersection_dict:
                    intersection_dict[seg1] = []
                # if seg2 not in intersection_dict:
                #     intersection_dict[seg2] = []
                
                # Append the intersection for both segments
                intersection_dict[seg1].append(intersection)
                # intersection_dict[seg2].append(intersection)
            
    pbar.close()
  
    return intersection_dict

# Function to merge results from multiple processes
def merge_intersections(results,segments):
    merged = {}
    for s in segments:
        if s not in merged:
            merged[s]=[s.start_point,s.end_point]
    for result in results:
        for seg, intersections in result.items():
            if seg not in merged:
                merged[seg] = []
            merged[seg].extend(intersections)
    for seg, isects in merged.items():
    
        isects.sort(key=lambda p: (p.x - seg.start_point.x)**2 + (p.y - seg.start_point.y)**2)
        merged[seg]=isects
    return merged

# Main function to find all intersections using multiprocessing
def find_all_intersections(segments, segmentation_config,epsilon=0.1):
    seg_block=build_initial_block(segments,segmentation_config)
    n = len(segments)
    L=4
    k= max((n+L-1)//L,1)
    # Divide segments into chunks of size k
    segment_chunks = [segments[i:i + k] for i in range(0, n, k)]
    s_l=[len(c) for c in segment_chunks]
    print(len(segment_chunks))
    print(s_l)
    # Use ProcessPoolExecutor for parallel computation
    with ProcessPoolExecutor(max_workers=L) as executor:
        partial_intersections = partial(compute_intersections, chunk2=seg_block,epsilon=epsilon)
        results = list(executor.map(partial_intersections, segment_chunks))

    # Merge results from all processes
    intersection_dict = merge_intersections(results,segments)
    return intersection_dict
from collections import deque

def split_segments(segments,segmentation_config, intersections,epsilon=0.25,expansion_param=0): 
    """根据交点将线段分割并构建 edge_map"""
    new_segments = []
    edge_map = {}
    point_map={}

    for seg, inter_points in intersections.items():
        # 按照坐标顺序排序交点
        inter_points = sorted([seg.start_point] + inter_points + [seg.end_point], key=lambda p: (p.x-seg.start_point.x)*(seg.end_point.x-seg.start_point.x)+(p.y-seg.start_point.y)*(seg.end_point.y-seg.start_point.y))
        points=inter_points
        segList=[]
        
        # 将线段分割为多个部分
        for i in range(len(points) - 1):
            # 比较两个点的大小，确保起点较小
            start_point, end_point = points[i], points[i+1]
            if (end_point.x, end_point.y) < (start_point.x, start_point.y):
                start_point, end_point = end_point, start_point
            if ((start_point.x-end_point.x)*(start_point.x-end_point.x)+(start_point.y-end_point.y)*(start_point.y-end_point.y)) < epsilon:
                continue
            # 创建新线段:
            new_seg = DSegment(start_point, end_point, seg.ref)
            segList.append(new_seg)


        for s in segList:
            new_segments.append(s)
            # 添加到 edge_map 中，包含正向和反向的线段
            if DSegment(s.start_point, s.end_point) not in edge_map:
                edge_map[DSegment(s.start_point, s.end_point)] = s
            if DSegment(s.end_point, s.start_point) not in edge_map:
                edge_map[DSegment(s.end_point, s.start_point)] = s  # 反向线段

    segments_set=set()
    new_seg=[]
    for s in new_segments:
        if s not in segments_set and DSegment(s.end_point,s.start_point) not in segments_set:
            segments_set.add(s)
            segments_set.add(DSegment(s.end_point,s.start_point,s.ref))
        else:
            if s.ref.color in segmentation_config.constraint_color:
                if s in segments_set:
                    segments_set.remove(s)
                if DSegment(s.end_point,s.start_point,s.ref) in segments_set:
                    segments_set.remove( DSegment(s.end_point,s.start_point,s.ref))
                segments_set.add(s)
                segments_set.add(DSegment(s.end_point,s.start_point,s.ref))
    seg_has_add=set()
    for s in segments_set:
        if s not in seg_has_add and DSegment(s.end_point,s.start_point) not in seg_has_add:
            seg_has_add.add(s)
            seg_has_add.add(DSegment(s.end_point,s.start_point,s.ref))
            new_seg.append(s)
    new_segments=new_seg
    for s in new_segments:
        vs,ve=s.start_point,s.end_point
        if vs not in point_map:
            point_map[vs]=set()
        if ve not in point_map:
            point_map[ve]=set()
        point_map[vs].add(s)
        point_map[ve].add(s)

    return new_segments, edge_map,point_map


def filter_segments(segments,segmentation_config,intersections,point_map,expansion_param=12,iters=3,interval=1,sign_handles=[]):
    new_segments=[]
    edge_map = {}
    new_point_map={}
    for seg, inter_points in intersections.items():
        # 按照坐标顺序排序交点
        
        inter_points = sorted([seg.start_point] + inter_points + [seg.end_point], key=lambda p: (p.x-seg.start_point.x)*(seg.end_point.x-seg.start_point.x)+(p.y-seg.start_point.y)*(seg.end_point.y-seg.start_point.y))
        points=inter_points
        segList=[]
        
        # 将线段分割为多个部分
        for i in range(len(points) - 1):
            # 比较两个点的大小，确保起点较小
            start_point, end_point = points[i], points[i+1]
            if (end_point.x, end_point.y) < (start_point.x, start_point.y):
                start_point, end_point = end_point, start_point
            if start_point == end_point:
                continue
            if ((start_point.x-end_point.x)*(start_point.x-end_point.x)+(start_point.y-end_point.y)*(start_point.y-end_point.y)) < expansion_param*expansion_param:
                continue
   
            new_seg = DSegment(start_point, end_point, seg.ref)
            segList.append(new_seg)


        # #filter lines
        # if len(segList)>1:
        for s in segList:
            new_segments.append(s)
            # 添加到 edge_map 中，包含正向和反向的线段
            if DSegment(s.start_point, s.end_point) not in edge_map:
                edge_map[DSegment(s.start_point, s.end_point)] = s
            if DSegment(s.end_point, s.start_point) not in edge_map:
                edge_map[DSegment(s.end_point, s.start_point)] = s  # 反向线段
        # elif len(segList)==1:
        #     if isinstance(segList[0].ref, DArc):
        #         #arc
        #         new_segments.append(segList[0])
        #         s=segList[0]
        #         if DSegment(s.start_point, s.end_point) not in edge_map:
        #             edge_map[DSegment(s.start_point, s.end_point)] = s
        #         if DSegment(s.end_point, s.start_point) not in edge_map:
        #             edge_map[DSegment(s.end_point, s.start_point)] = s  # 反向线段
        #     else:
        #         vs,ve=segList[0].start_point,segList[0].end_point
        #         s_deg,e_deg=len(point_map[vs]),len(point_map[ve])
        #         if s_deg==1 or e_deg==1:
        #             continue
        #         else:
        #             new_segments.append(segList[0])
        #             s=segList[0]
        #             if DSegment(s.start_point, s.end_point) not in edge_map:
        #                 edge_map[DSegment(s.start_point, s.end_point)] = s
        #             if DSegment(s.end_point, s.start_point) not in edge_map:
        #                 edge_map[DSegment(s.end_point, s.start_point)] = s  # 反向线段

    segments_set=set()
    new_seg=[]
    for s in new_segments:
        if s not in segments_set and DSegment(s.end_point,s.start_point) not in segments_set:
            segments_set.add(s)
            segments_set.add(DSegment(s.end_point,s.start_point,s.ref))
        else:
            if s.ref.color in segmentation_config.constraint_color:
                if s in segments_set:
                    segments_set.remove(s)
                if DSegment(s.end_point,s.start_point,s.ref) in segments_set:
                    segments_set.remove( DSegment(s.end_point,s.start_point,s.ref))
                segments_set.add(s)
                segments_set.add(DSegment(s.end_point,s.start_point,s.ref))
    seg_has_add=set()
    for s in segments_set:
        if s not in seg_has_add and DSegment(s.end_point,s.start_point) not in seg_has_add:
            seg_has_add.add(s)
            seg_has_add.add(DSegment(s.end_point,s.start_point,s.ref))
            new_seg.append(s)
    new_segments=new_seg
    for s in new_segments:
        vs,ve=s.start_point,s.end_point
        if vs not in new_point_map:
            new_point_map[vs]=set()
        if ve not in new_point_map:
            new_point_map[ve]=set()
        new_point_map[vs].add(s)
        new_point_map[ve].add(s)
    for p,ss in new_point_map.items():
        new_point_map[p]=list(ss)
    

    pbar=tqdm(total=iters,desc="过滤线段")
    for i in range(iters):
        filtered_segments=[]
        filtered_edge_map={}
        filtered_point_map={}

        for s in new_segments:
            vs,ve=s.start_point,s.end_point
            div_s,div_e=len(new_point_map[vs]),len(new_point_map[ve])
            if vs==ve:
                continue
            if div_s==1 or div_e==1:
                continue
            flag=False
            for ns in new_point_map[vs]:
                if ns==s:
                    continue
                if ns.ref is not None and ns.ref.handle is not None and ns.ref.handle in sign_handles:
                    flag=True
                    break
            for ns in new_point_map[ve]:
                if ns==s:
                    continue
                if ns.ref is not None and ns.ref.handle is not None and ns.ref.handle in sign_handles:
                    flag=True
                    break
            if flag:
                # print(s.ref.handle)
                continue
            # if div_s==3 and (i+1)%interval==0:
            #     ns=[ss for ss in new_point_map[vs] if ss !=s]
            #     if isinstance(ns[0].ref,DArc) and ns[0].ref == ns[1].ref:
            #         continue
            # if div_e==3 and (i+1)%interval==0:
            #     ns=[ss for ss in new_point_map[ve] if ss !=s]
            #     if isinstance(ns[0].ref,DArc) and ns[0].ref == ns[1].ref:
            #         continue
            
            filtered_segments.append(s)
            if DSegment(s.start_point, s.end_point) not in filtered_edge_map:
                filtered_edge_map[DSegment(s.start_point, s.end_point)] = s
            if DSegment(s.end_point, s.start_point) not in filtered_edge_map:
                filtered_edge_map[DSegment(s.end_point, s.start_point)] = s  # 反向线段
        for s in filtered_segments:
            vs,ve=s.start_point,s.end_point
            if vs not in filtered_point_map:
                filtered_point_map[vs]=set()
            if ve not in filtered_point_map:
                filtered_point_map[ve]=set()
            filtered_point_map[vs].add(s)
            filtered_point_map[ve].add(s)
        for p,ss in filtered_point_map.items():
            filtered_point_map[p]=list(ss)
        new_segments=filtered_segments
        edge_map=filtered_edge_map
        new_point_map=filtered_point_map
        pbar.update()
    pbar.close()
    return new_segments,edge_map,new_point_map
def build_graph(segments):
    """根据分割后的线段构建图，保存每条边及其引用的ref"""
    graph = {}

    for seg in segments:
        p1, p2 = seg.start_point, seg.end_point
        
        if p1 not in graph:
            graph[p1] = []
        if p2 not in graph:
            graph[p2] = []
        
        # 保存连接的点以及线段引用信息（ref）
        graph[p1].append((p2, seg.ref))
        graph[p2].append((p1, seg.ref))

    return graph


def bfs_paths(graph, start_point, end_point,max_length,timeout=5):
    """基于广度优先搜索找到所有从start_point到end_point的路径，返回路径中的Dsegment"""
    start_time=time.time()
    queue = deque([(start_point, [start_point], [])])  # (当前点，路径中的点，路径中的线段)
    all_paths = []

    while queue:
        current_time=time.time()
        if (current_time-start_time)>=timeout:
            print(f'{start_point}->{end_point}已超时')
            break
        (current_point, point_path, seg_path) = queue.popleft()
        if len(seg_path)>=max_length:
            continue
        for neighbor, ref in graph.get(current_point, []):
            if neighbor not in point_path:
                new_point_path = point_path + [neighbor]
                new_seg = DSegment(current_point, neighbor, ref)  # 构建对应的Dsegment
                new_seg_path = seg_path + [new_seg]

                if neighbor == end_point:
                    if len(new_point_path) > 1:  # 避免 trivial paths
                        all_paths.append(new_seg_path)
                else:
                    queue.append((neighbor, new_point_path, new_seg_path))

    return all_paths

def calculate_angle(p1, p2, p3):
    """
    Calculate the angle (in degrees) formed by three points p1, p2, and p3.
    The angle is measured from vector (p2 -> p1) to vector (p2 -> p3).
    """
    v1 = (p1.x - p2.x, p1.y - p2.y)
    v2 = (p3.x - p2.x, p3.y - p2.y)
    
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    det = v1[0] * v2[1] - v1[1] * v2[0]
    angle = math.atan2(det, dot_product)  # Angle in radians
    angle_deg = math.degrees(angle)

    # Ensure the angle is in the range [0, 360]
    if angle_deg < 0:
        angle_deg += 360
    # if p2==DPoint(747,1121):
    #     print(angle_deg)
    return angle_deg



def calculate_prior_angle(p1,p2,p3):
    angle=calculate_angle(p3,p2,p1)
    if angle>=10 and angle<=170:
        angle+=180
    elif angle<=350 and angle>=190:
        angle-=160
    elif angle>=170 and angle <=190:
        angle=10
    else:
        angle=0
    # if p2==DPoint(747,1121):
    #     print(angle)
    return angle
def dfs_paths_with_repline(visited_edges,graph, repline, max_length, timeout=5):
    """
    Perform DFS to find a closed loop starting at repline.start_point,
    with the first segment being repline and ending back at repline.start_point.
    Prioritize edges with the largest counterclockwise angle change within [10, 170] degrees.
    """
    start_time = time.time()
    start_point = repline.start_point
    stack = [(repline.end_point, [start_point, repline.end_point], [repline])]  # Start with repline
    # visited_edges = {(repline.start_point, repline.end_point)}  # Mark the first edge as visited

    while stack:
        current_time = time.time()
        if (current_time - start_time) >= timeout:
            print(f'{start_point} search timed out')
            break
        
        current_point, point_path, seg_path = stack.pop()

        if len(seg_path) >= max_length:
            continue
  
        if current_point == start_point and len(seg_path) > 1:
            return seg_path  # Return the first valid closed loop
    
        # Get neighbors and sort them by counterclockwise angle change
        neighbors = []
        for neighbor, ref in graph.get(current_point, []):
            edge = (current_point, neighbor)
            prev_point = point_path[-2]
            if edge not in visited_edges and prev_point!=neighbor:
                neighbors.append((neighbor, ref))

        # Sort neighbors by the counterclockwise angle change
        if len(point_path) > 1:
            prev_point = point_path[-2]
            neighbors.sort(key=lambda nbr: calculate_angle(prev_point, current_point, nbr[0]), reverse=True)
        flag=False
        for neighbor, ref in neighbors:
           
            # angle = calculate_angle(point_path[-2], current_point, neighbor) if len(point_path) > 1 else 0
            
            # if 10 <= angle <= 170:  
            edge = (current_point, neighbor)
            visited_edges.add(edge)

            new_point_path = point_path + [neighbor]
            new_seg = DSegment(current_point, neighbor, ref)
            new_seg_path = seg_path + [new_seg]

            stack.append((neighbor, new_point_path, new_seg_path))
            flag=True
            break
        if flag==False:
            return []

    return []  # Return an empty path if no closed loop is found

def process_repline_with_repline_dfs(visited_edges,repline, graph, segmentation_config):
    """
    Process a single repline using the DFS algorithm to find a closed loop
    starting and ending at repline.start_point.
    """
    path = dfs_paths_with_repline(visited_edges,graph, repline, segmentation_config.dfs_path_max_length, segmentation_config.timeout)
    return path


# def compute_arc_replines(new_segments,point_map):
#     """
#     计算arc_replines，筛选出ref是弧线且半径在20~160之间的线段。
#     :param new_segments: 分割后的线段列表
#     :return: arc_replines 列表
#     """
#     arc_replines_map={}
#     arc_replines = []

#     for segment in new_segments:
#         # 检查线段的ref是否是弧线，且半径在 20 到 160 之间
#         if isinstance(segment.ref, DArc):
#             radius = segment.ref.radius
#             if 20 <= radius and radius <= 160:
#                 arc_tuple=(segment.ref.center.x,segment.ref.center.y,segment.ref.radius,segment.ref.start_angle,segment.ref.end_angle)
#                 if  arc_tuple not in arc_replines_map:
#                     arc_replines_map[arc_tuple]=[]
#                 arc_replines_map[arc_tuple].append(segment)
#     for arc_tuple,segments in arc_replines_map.items():
#         # arc_replines_map[arc_tuple]=sorted(segments,key= lambda s: DSegment(s.ref.center,s.mid_point()).slope())
#         for s in segments:
#             if len(point_map[s.start_point])>2 or len(point_map[s.end_point])>2:
#                 arc_replines.append(s)
        
#     return arc_replines

def compute_star_replines(new_segments,elements):
    vertical_lines=[]
    star_replines=[]
    star_pos_map={}
    star_pos_set=set()
    for e in elements:
        if isinstance(e,DText) and is_star_text(e.content):
            x,y=(e.bound["x1"]+e.bound["x2"])/2,(e.bound["y1"]+e.bound["y2"])/2
            vertical_lines.append(DSegment(DPoint(x,y),DPoint(x,y+5000)))
    for i, seg1 in enumerate(vertical_lines):
        y_min=None
        s=None

        for j, seg2 in enumerate(new_segments):
            p1, p2 = seg1.start_point, seg1.end_point
            q1, q2 = seg2.start_point, seg2.end_point
            intersection = segment_intersection(p1, p2, q1, q2)
            if intersection:
                if y_min is None:
                    y_min=intersection[1]
                    s=seg2
                else:
                    if y_min>intersection[1]:
                        y_min=intersection[1]
                        s=seg2
        if s is not None:
            star_replines.append(s)
            rs=DSegment(s.end_point,s.start_point,s.ref)
            if s not in star_pos_map:
                star_pos_map[s]=set()
            if rs not in star_pos_map:
                star_pos_map[rs]=set()
            star_pos_map[s].add(seg1.start_point)
            star_pos_map[rs].add(seg1.start_point)
            star_pos_set.add(seg1.start_point)
    for pos,ss in star_pos_map.items():
        star_pos_map[pos]=list(ss)
    return star_replines,star_pos_map,list(star_pos_set)


def compute_line_replines(new_segments,point_map):
    line_replines=[]
    for s in new_segments:
        if s.length()<13 or s.length()>30:
            continue
        vs,ve=s.start_point,s.end_point
        div_s,div_e=len(point_map[vs]),len(point_map[ve])
        if div_s==3 and div_e==3:
            sn=[neighbor for neighbor in point_map[vs] if neighbor!=s]
            if sn[0].length()<sn[1].length():
                a=sn[0]
            else:
                a=sn[1]
            en=[neighbor for neighbor in point_map[ve] if neighbor!=s]
            if en[0].length()<en[1].length():
                b=en[0]
            else:
                b=en[1]
            if a.start_point==vs:
                p=a.end_point
            else:
                p=a.start_point
            if p in [b.start_point,b.end_point]:
                line_replines.append(s)
    return line_replines

def is_repline(s,segmentation_config):
    #print( s.ref.meta.scales[0])
    if isinstance(s.ref,DArc) and s.ref.radius<=segmentation_config.arc_repline_max_length and s.ref.radius>=segmentation_config.arc_repline_min_length:
        return True
    elif(not isinstance(s.ref,DArc)) and s.length()>=segmentation_config.line_repline_min_length and s.length()<=segmentation_config.line_repline_max_length:
        return True
    return False


def checkRefAndSlope(p,segments,tolerance,point_map,segmentation_config):
   
    #print(slopes)
    flag=False
    idx=None
    lines=[]
    # if p==DPoint(1170.958216976533, 3938.019285178291):
    #     print(segments)
    for i,s1 in enumerate(segments):
        for j,s2 in enumerate(segments):
            if i>=j:
                continue
            if is_parallel(s1,s2,tolerance):
                flag=True
                l1,l2=segments[i].length(),segments[j].length()
                # if p==DPoint(777,1321) or p==DPoint():
                #     print(l1,l2)
                #     print(s1,s2)
                if l1<segmentation_config.repline_neighbor_min_length:
                    l=l1
                    q=s1.start_point if s1.end_point==p else s1.end_point
                    nq=[s for s in point_map[q] if s!=s1 and is_parallel(s,s1,tolerance)]
                    if len(nq)>0:
                        l+=nq[0].length()
                    k=0
                    while len(nq)>0:
                        q=nq[0].start_point if nq[0].end_point==q else nq[0].end_point
                        nq=[s for s in point_map[q] if s!=nq[0] and is_parallel(s,nq[0],tolerance)]
                        if len(nq)>0:
                            l+=nq[0].length()
                            k+=1
                            if k>10:
                                break
                    # if p==DPoint(1170.958216976533, 3938.019285178291):
                    #     print(l)
                    if l<segmentation_config.repline_neighbor_min_length:
                        flag=False
                if flag and l2<segmentation_config.repline_neighbor_min_length:
                    l=l2
                    q=s2.start_point if s2.end_point==p else s2.end_point
                    nq=[s for s in point_map[q] if s!=s2 and is_parallel(s,s2,tolerance)]
                    if len(nq)>0:
                        l+=nq[0].length()
                    k=0
                    while len(nq)>0:
                        q=nq[0].start_point if nq[0].end_point==q else nq[0].end_point
                        nq=[s for s in point_map[q] if s!=nq[0] and is_parallel(s,nq[0],tolerance)]
                        if k>10:
                            break
                        if len(nq)>0:
                            l+=nq[0].length()
                            k+=1
                    # if p==DPoint(1170.958216976533, 3938.019285178291):
                    #     print(l)
                    if l<segmentation_config.repline_neighbor_min_length:
                        flag=False
                
                if flag:
                    lines.append(s1)
                    lines.append(s2)
                    break
        if flag:
            break
    return [flag,lines]

def checkValid(repline,segments,tolerance,segmentation_config):
    flag=True
    for i,segment in enumerate(segments):
        if segment.length()>segmentation_config.check_valid_min_length and is_parallel(segment,repline,tolerance) and segment.ref==repline.ref:
            flag=False
            break
        
    return flag
                    
def checkTwoEndLines(ls,le,segmentation_config,single=False):
    flag=True
    if len(ls)!=2 or len(le)!=2:
        return False
    for i,s1 in enumerate(ls):
        for j,s2 in enumerate(le):
            if s1==s2 or s1==DSegment(s2.end_point,s2.start_point,s2.ref):
                continue
            if is_parallel(s1,s2,segmentation_config.is_parallel_tolerance_neighobor):
                flag=False
            if flag==False:
                break
        if flag==False:
            break
    if single and isinstance(ls[0].ref,DLine) and isinstance(ls[1].ref,DLine) and isinstance(le[0].ref,DLine) and isinstance(le[1].ref,DLine) and (ls[0].ref==ls[1].ref or le[0].ref==le[1].ref):
        count_map={}
        for h in [ls[0].ref.handle,ls[1].ref.handle,le[0].ref.handle,le[1].ref.handle]:
            if h not in count_map:
                count_map[h]=0
            count_map[h]+=1
            if count_map[h]>2:
                return False

        if ls[0].ref==ls[1].ref:
            p=None
            for s1 in [ls[0].start_point,ls[0].end_point]:
                for s2 in [ls[1].start_point,ls[1].end_point]:
                    if s1==s2:
                        p=s1
            if p==ls[0].ref.start_point or p==ls[0].ref.end_point :
                return False
        
        if le[0].ref==le[1].ref:
            p=None
            for s1 in [le[0].start_point,le[0].end_point]:
                for s2 in [le[1].start_point,le[1].end_point]:
                    if s1==s2:
                        p=s1
            if p==le[0].ref.start_point or p==le[0].ref.end_point :
                return False
    return flag
def compute_cornor_holes(filtered_segments,filtered_point_map,segmentation_config):
    cornor_holes=[]
    segment_is_visited=set()
    for s in filtered_segments:
        vs,ve=s.start_point,s.end_point
        degs,dege=len(filtered_point_map[vs]),len(filtered_point_map[ve])
        if s not in segment_is_visited and (degs>2 or dege>2) and is_repline(s,segmentation_config):
            if degs>2 and dege>2:
                ns=[ss for ss in filtered_point_map[vs] if ss!=s ]
                ne=[ss for ss in filtered_point_map[ve] if ss!=s ]
                #print(222)
                if checkRefAndSlope(vs,ns,segmentation_config.is_parallel_tolerance_neighobor,filtered_point_map,segmentation_config)[0] and checkRefAndSlope(ve,ne,segmentation_config.is_parallel_tolerance_neighobor,filtered_point_map,segmentation_config)[0] and checkValid(s,ns,segmentation_config.is_parallel_tolerance_neighobor,segmentation_config) and checkValid(s,ne,segmentation_config.is_parallel_tolerance_neighobor,segmentation_config):
                    # a,b=checkRefAndSlope(ns)[1],checkRefAndSlope(ne)[1]
                    # pa=a.start_point if a.start_point!=vs else a.end_point
                    # pb=b.start_point if b.start_point!=ve else b.end_point
                    # sss=expandFixedLength([DSegment(vs,pa),DSegment(ve,pb)],150,False)
                    # if len(sss) <2:
                    #     break
                    # inter=segment_intersection(sss[0].start_point,sss[0].end_point,sss[1].start_point,sss[1].end_point)
                    # if inter is not None:
                    
                    ls=checkRefAndSlope(vs,ns,segmentation_config.is_parallel_tolerance_neighobor,filtered_point_map,segmentation_config)[1]
                    le=checkRefAndSlope(ve,ne,segmentation_config.is_parallel_tolerance_neighobor,filtered_point_map,segmentation_config)[1]
                    single=True
                    if isinstance(s.ref,DArc):
                        single=False
                    if checkTwoEndLines(ls,le,segmentation_config,single=single):
                        segment_is_visited.add(s)
                        cornor_holes.append(DCornorHole([s]))
            else:
                other=None
                segments=[]
                if degs>2:
                    start=vs
                    other=ve
                else:
                    start=ve
                    other=vs
                dego=len(filtered_point_map[other])
                segments.append(s)
                current=s
                flag=True
                while dego==2 and flag:
                    
                    cs=[ss for ss in filtered_point_map[other] if ss!=current]
                    if  len(cs)!=1:
                        flag=False
                    
                        break
                    else:
                        current=cs[0]
               
                    os=[p for p in [current.start_point,current.end_point] if p!=other]
                    if len(os)!=1 :
                        flag=False
                     
                        break
                    else:
                        other=os[0]
                 
                    dego=len(filtered_point_map[other])
                    if  current not in segment_is_visited and is_repline(current,segmentation_config):
                            segments.append(current)
                    else:
                        flag=False
                        break
                if flag:
                    ns=[ss for ss in filtered_point_map[start] if ss!=s ]
                    ne=[ss for ss in filtered_point_map[other] if ss!=current ]
                    #print(111)
                    # if start==DPoint(1170.958216976533, 3938.019285178291) or other==DPoint(1170.958216976533, 3938.019285178291):
                    #     print(checkRefAndSlope(start,ns,segmentation_config.is_parallel_tolerance_neighobor,filtered_point_map,segmentation_config)[0])
                    #     print(checkRefAndSlope(other,ne,segmentation_config.is_parallel_tolerance_neighobor,filtered_point_map,segmentation_config)[0]) 
                    #     print(checkValid(s,ns,segmentation_config.is_parallel_tolerance_neighobor,segmentation_config))
                    #     print(checkValid(current,ne,segmentation_config.is_parallel_tolerance_neighobor,segmentation_config))
                    if checkRefAndSlope(start,ns,segmentation_config.is_parallel_tolerance_neighobor,filtered_point_map,segmentation_config)[0] and checkRefAndSlope(other,ne,segmentation_config.is_parallel_tolerance_neighobor,filtered_point_map,segmentation_config)[0] and checkValid(s,ns,segmentation_config.is_parallel_tolerance_neighobor,segmentation_config) and checkValid(current,ne,segmentation_config.is_parallel_tolerance_neighobor,segmentation_config):
                        ls=checkRefAndSlope(start,ns,segmentation_config.is_parallel_tolerance_neighobor,filtered_point_map,segmentation_config)[1]
                        le=checkRefAndSlope(other,ne,segmentation_config.is_parallel_tolerance_neighobor,filtered_point_map,segmentation_config)[1]
                        

                        if checkTwoEndLines(ls,le,segmentation_config):
                            for ss in segments:
                                segment_is_visited.add(ss)
                            cornor_holes.append(DCornorHole(segments))
    for cornor_hole in cornor_holes:
        for s in cornor_hole.segments:
            s.isCornerhole=True
    #print(cornor_holes)
    return cornor_holes


def filter_cornor_holes(cornor_holes,filtered_point_map,segmentation_config):
    filtered_cornor_holes=[]
    for cornor_hole in cornor_holes:
        total=0
        n=0
        for s in cornor_hole.segments:
            total+=s.length()
            n+=1

        if n>0 and total>=segmentation_config.cornor_hole_total_length and total/n>=segmentation_config.cornor_hole_average_length:
            filtered_cornor_holes.append(cornor_hole)
    return filtered_cornor_holes

def isBraketHints(e):
    if isinstance(e,DText) and bool(re.match(r".*(F?B\d+X\d+).*", e.content)):
        return  True
    return False
def findBraketByHints(filtered_segments,text_map):
    vertical_lines=[]
    for pos,ts in text_map.items():
        for t in ts:
            result=t[1]
            if result["Type"]=="B" or result["Type"]=="BK":
                vertical_lines.append(DSegment(pos,DPoint(pos.x,pos.y+5000)))
                break
    braket_start_lines=[]

    #print(vertical_lines)
    for i, seg1 in enumerate(vertical_lines):
        y_min=None
        s=None
        for j, seg2 in enumerate(filtered_segments):
            p1, p2 = seg1.start_point, seg1.end_point
            q1, q2 = seg2.start_point, seg2.end_point
            intersection = segment_intersection(p1, p2, q1, q2)
            if intersection:
                if y_min is None:
                    y_min=intersection[1]
                    s=seg2
                else:
                    if y_min>intersection[1]:
                        y_min=intersection[1]
                        s=seg2
        if s is not None:
            braket_start_lines.append(s)
           
    

    return braket_start_lines

def findBracketByPoints(pos,filtered_segments):
    braket_start_lines=[]
    vertical_lines=[]
    for p in pos:
        x,y=p.x,p.y
        vertical_lines.append(DSegment(DPoint(x,y),DPoint(x,y-5000)))
    for i, seg1 in enumerate(vertical_lines):
        y_max=None
        s=None
        for j, seg2 in enumerate(filtered_segments):
          
            p1, p2 = seg1.start_point, seg1.end_point
            q1, q2 = seg2.start_point, seg2.end_point
            intersection = segment_intersection(p1, p2, q1, q2)
            if intersection:
                if y_max is None:
                    y_max=intersection[1]
                    s=seg2
                else:
                    if y_max<intersection[1]:
                        y_max=intersection[1]
                        s=seg2
        if s is not None:
            braket_start_lines.append(s)
    return braket_start_lines

def removeReferenceLines(elements,texts,initial_segments,all_segments,point_map,segmentation_config):
    vertical_lines=[]
    vl2=[]
    v1e=[]
    v2e=[]
    text_pos=[]
    for e in texts:
        mid_point=DPoint((e.bound["x1"]+e.bound["x2"])/2,(e.bound["y1"]+e.bound["y2"])/2)
        x,y=mid_point.x,mid_point.y
        x_1=(x+e.bound["x1"])/2
        x_2=(x+e.bound["x2"])/2
        vertical_lines.append(DSegment(DPoint(x_1,y),DPoint(x_1,y-500)))
        vertical_lines.append(DSegment(DPoint(x_2,y),DPoint(x_2,y-500)))
        vertical_lines.append(DSegment(DPoint(x,y),DPoint(x,y-500)))
        v1e.append(e)
        v1e.append(e)
        v1e.append(e)
        v2e.append(e)
        v2e.append(e)
        v2e.append(e)
        vl2.append(DSegment(DPoint(x,y),DPoint(x,y+500)))
        vl2.append(DSegment(DPoint(x_1,y),DPoint(x,y+500)))
        vl2.append(DSegment(DPoint(x_2,y),DPoint(x,y+500)))
    #print(len(vertical_lines))
    print(len(vertical_lines)*len(all_segments))
    print(len(vl2)*len(all_segments))
    horizontal_line=[]
    hl2=[]
    h1e=[]
    h2e=[]
            
    for i, seg1 in enumerate(vertical_lines):
        y_max=None
        s=None
        for j, seg2 in enumerate(all_segments):
            p1, p2 = seg1.start_point, seg1.end_point
            q1, q2 = seg2.start_point, seg2.end_point
            intersection = segment_intersection(p1, p2, q1, q2)
            if intersection:
                if y_max is None:
                    y_max=intersection[1]
                    s=seg2
                else:
                    if y_max<intersection[1]:
                        y_max=intersection[1]
                        s=seg2
        if s is not None:
            text_pos=seg1.start_point
            if (text_pos.y-y_max)<=segmentation_config.reference_text_max_distance and s.ref.color==7 and (len(point_map[s.start_point])==1 or len(point_map[s.end_point])==1):
                horizontal_line.append(s)
                h1e.append(v1e[i])
    for i, seg1 in enumerate(vl2):
        y_min=None
        s=None
        for j, seg2 in enumerate(all_segments):
            p1, p2 = seg1.start_point, seg1.end_point
            q1, q2 = seg2.start_point, seg2.end_point
            intersection = segment_intersection(p1, p2, q1, q2)
            if intersection:
                if y_min is None:
                    y_min=intersection[1]
                    s=seg2
                else:
                    if y_min>intersection[1]:
                        y_min=intersection[1]
                        s=seg2
        if s is not None:
            text_pos=seg1.start_point
            if (y_min-text_pos.y)<=segmentation_config.reference_text_max_distance and s.ref.color==7 and (len(point_map[s.start_point])==1 or len(point_map[s.end_point])==1):
                hl2.append(s)
                h2e.append(v2e[i])
    # print(len(horizontal_line))
    reference_lines=[]
    text_pos_map={}
    text_set=set()
    for i,line in enumerate(horizontal_line):
        if len(point_map[line.start_point])==1:
            p=line.end_point
        else:
            p=line.start_point
        sr=[s.ref for s in point_map[p] if s.length()>segmentation_config.reference_line_min_length  and (isinstance(s.ref, DLine) or isinstance(s.ref,DLwpolyline)) and s!=line and angleOfTwoSegmentsWithCommonStarter(p,s,line)>90]
        ss=[s for s in point_map[p] if s.length()>segmentation_config.reference_line_min_length  and (isinstance(s.ref, DLine) or isinstance(s.ref,DLwpolyline)) and s!=line and angleOfTwoSegmentsWithCommonStarter(p,s,line)>90]
        for j,s in enumerate(ss):
            reference_lines.append(sr[j])
            start_point=p
            current_point=s.start_point if s.start_point!=start_point else s.end_point
            current_line=s
            while True:
                if len(point_map[current_point])<=1:
                    break
                current_nedge=[sss for sss in point_map[current_point] if sss!=current_line and is_parallel(current_line,sss,segmentation_config.is_parallel_tolerance)]
                if len(current_nedge)==0:
                    break
                current_line=current_nedge[0]
                current_point=current_line.start_point if current_line.start_point!=current_point else current_line.end_point
           
            if current_point not in text_pos_map:
                text_pos_map[current_point]=set()
                text_pos_map[current_point].add(h1e[i])
                if h1e[i] not in text_set:
                    text_set.add(h1e[i])
                h1e[i].textpos=True
            else:
                text_pos_map[current_point].add(h1e[i])
                if h1e[i] not in text_set:
                    text_set.add(h1e[i])
                h1e[i].textpos=True
        
        reference_lines.append(line.ref)
    for i,line in enumerate(hl2):
        if len(point_map[line.start_point])==1:
            p=line.end_point
        else:
            p=line.start_point
        sr=[s.ref for s in point_map[p] if s.length()>segmentation_config.reference_line_min_length  and (isinstance(s.ref, DLine) or isinstance(s.ref,DLwpolyline)) and s!=line and angleOfTwoSegmentsWithCommonStarter(p,s,line)>segmentation_config.reference_min_angle]
        ss=[s for s in point_map[p] if s.length()>segmentation_config.reference_line_min_length  and (isinstance(s.ref, DLine) or isinstance(s.ref,DLwpolyline)) and s!=line and angleOfTwoSegmentsWithCommonStarter(p,s,line)>segmentation_config.reference_min_angle]
        for j,s in enumerate(ss):
            reference_lines.append(sr[j])
            start_point=p
            current_point=s.start_point if s.start_point!=start_point else s.end_point
            current_line=s
            while True:
                if len(point_map[current_point])<=1:
                    break
                current_nedge=[sss for sss in point_map[current_point] if sss!=current_line and is_parallel(current_line,sss,segmentation_config.is_parallel_tolerance)]
                if len(current_nedge)==0:
                    break
                current_line=current_nedge[0]
                current_point=current_line.start_point if current_line.start_point!=current_point else current_line.end_point
            if current_point not in text_pos_map:
                text_pos_map[current_point]=set()
                text_pos_map[current_point].add(h2e[i])
                if h2e[i] not in text_set:
                    text_set.add(h2e[i])
                h2e[i].textpos=True
            else:
                text_pos_map[current_point].add(h2e[i])
                if h2e[i] not in text_set:
                    text_set.add(h2e[i])
                h2e[i].textpos=True
        reference_lines.append(line.ref)
    print(len(reference_lines)*len(initial_segments))
    new_segments=[]
    for s in initial_segments:
        
        if s.ref not in reference_lines:
            new_segments.append(s)
            #print(s.ref)
    return new_segments,reference_lines,text_pos_map,text_set

import itertools

def process_intersections(chunck,segments,point_map,segmentation_config):
    """Compute the intersection for a single vertical line and all segments."""
    vertical_lines=chunck[0]
    v1e=chunck[1]
    horizontal_line=[]
    h1e=[]
    pbar=tqdm(desc="find reference line",total=len(vertical_lines))
    for i, seg1 in enumerate(vertical_lines):
        y_max=None
        s=None
        pbar.update()
        all_seg=segments.segments_near_segment(seg1)
        for j, seg2 in enumerate(all_seg):
            
            if len(point_map[seg2.start_point])==1 and len(point_map[seg2.end_point])==1:
                continue
            p1, p2 = seg1.start_point, seg1.end_point
            q1, q2 = seg2.start_point, seg2.end_point
            intersection = segment_intersection(p1, p2, q1, q2)
            if intersection:
                if y_max is None:
                    y_max=intersection[1]
                    s=seg2
                else:
                    if y_max<intersection[1]:
                        y_max=intersection[1]
                        s=seg2
        if s is not None:
            text_pos=seg1.start_point
            if (text_pos.y-y_max)<=segmentation_config.reference_text_max_distance and (isinstance(s.ref, DLine) or isinstance(s.ref,DLwpolyline))  and (s.ref.color==7 or s.ref.color==1) and ((len(point_map[s.start_point])==1 and  len(point_map[s.end_point])>1) or (len(point_map[s.start_point])>1 and  len(point_map[s.end_point])==1)):
                horizontal_line.append(s)
                h1e.append(v1e[i])
            elif (text_pos.y-y_max)<=segmentation_config.reference_text_max_distance and (isinstance(s.ref, DLine) or isinstance(s.ref,DLwpolyline))  and (s.ref.color==7 or s.ref.color==1) and (len(point_map[s.start_point])>1 and  len(point_map[s.end_point])>1) and math.fabs(s.start_point.y-s.end_point.y)<5:
                horizontal_line.append(s)
                h1e.append(v1e[i])
    pbar.close()
    return [horizontal_line,h1e]

def process_intersections2(chunck,segments,point_map,segmentation_config):
    vl2=chunck[0]
    v2e=chunck[1]
    hl2=[]
    h2e=[]
    pbar=tqdm(desc="find reference lines",total=len(vl2))
    for i, seg1 in enumerate(vl2):
        y_min=None
        s=None
        pbar.update()
        all_segs=segments.segments_near_segment(seg1)
        for j, seg2 in enumerate(all_segs):
            
            p1, p2 = seg1.start_point, seg1.end_point
            q1, q2 = seg2.start_point, seg2.end_point
            intersection = segment_intersection(p1, p2, q1, q2)
            if intersection:
                if y_min is None:
                    y_min=intersection[1]
                    s=seg2
                else:
                    if y_min>intersection[1]:
                        y_min=intersection[1]
                        s=seg2
        if s is not None:
            text_pos=seg1.start_point
            if (y_min-text_pos.y)<=segmentation_config.reference_text_max_distance and (isinstance(s.ref, DLine) or isinstance(s.ref,DLwpolyline))  and (s.ref.color==7 or s.ref.color==1) and ((len(point_map[s.start_point])==1 and  len(point_map[s.end_point])>1) or (len(point_map[s.start_point])>1 and  len(point_map[s.end_point])==1)):
                hl2.append(s)
                h2e.append(v2e[i])
            elif (y_min-text_pos.y)<=segmentation_config.reference_text_max_distance and (isinstance(s.ref, DLine) or isinstance(s.ref,DLwpolyline))  and (s.ref.color==7 or s.ref.color==1) and (len(point_map[s.start_point])>1 and len(point_map[s.end_point])>1) and math.fabs(s.start_point.y-s.end_point.y)<5:
                hl2.append(s)
                h2e.append(v2e[i])
                
    pbar.close()
    return [hl2,h2e]
def checkReferenceLine(p,ns,ss,segmentation_config):
    if len(ns)==1:
        if len(ss)==1:
            return True
        return False
    if len(ns)>3:
        return False
    if len(ss)!=len(ns):
        return False
    return True
def check_in_arc(sss,point_map):
    vs,ve=sss.start_point,sss.end_point
    if len(point_map[vs])<2 or len(point_map[ve])<2:
        return False

    ns,ne=point_map[vs],point_map[ve]
    ns=[s for s in ns if isinstance(s.ref,DArc)]
    ne=[s for s in ne if isinstance(s.ref,DArc)]
    if len(ns)==0 or len(ne)==0:
        return False
    if ns[0].ref==ne[0].ref:
        return True
    else:
        return False

def removeReferenceLines(elements,texts,initial_segments,all_segments,point_map,segmentation_config):


 


    vertical_lines=[]
    vl2=[]
    v1e=[]
    v2e=[]
    text_pos=[]
    for e in texts:
        mid_point=DPoint((e.bound["x1"]+e.bound["x2"])/2,(e.bound["y1"]+e.bound["y2"])/2)
        x,y=mid_point.x,mid_point.y
        x_1=(x+e.bound["x1"])/2
        x_2=(x+e.bound["x2"])/2
        vertical_lines.append(DSegment(DPoint(x_1,y),DPoint(x_1,y-500)))
        vertical_lines.append(DSegment(DPoint(x_2,y),DPoint(x_2,y-500)))
        vertical_lines.append(DSegment(DPoint(x,y),DPoint(x,y-500)))
        v1e.append(e)
        v1e.append(e)
        v1e.append(e)
        v2e.append(e)
        v2e.append(e)
        v2e.append(e)
        vl2.append(DSegment(DPoint(x,y),DPoint(x,y+500)))
        vl2.append(DSegment(DPoint(x_1,y),DPoint(x,y+500)))
        vl2.append(DSegment(DPoint(x_2,y),DPoint(x,y+500)))
    #print(len(vertical_lines))
    print(len(vertical_lines)*len(all_segments))
    print(len(vl2)*len(all_segments))
    horizontal_line=[]
    hl2=[]
    h1e=[]
    h2e=[]
    all_block=build_initial_block(all_segments,segmentation_config)
    n = len(vertical_lines)
    L=4
    k= max((n+L-1)//L,1)
    # Divide segments into chunks of size k
    segment_chunks = [vertical_lines[i:i + k] for i in range(0, n, k)]
    element_chunks=  [v1e[i:i + k] for i in range(0, n, k)]
    chuncks=[]
    for i in range(len(segment_chunks)):
        chuncks.append([segment_chunks[i],element_chunks[i]])
    s_l=[len(c) for c in segment_chunks]
    print(len(segment_chunks))
    print(s_l)
    # Use ProcessPoolExecutor for parallel computation
    with ProcessPoolExecutor(max_workers=L) as executor:
        partial_intersections = partial(process_intersections, segments=all_block,point_map=point_map,segmentation_config=segmentation_config)
        results = list(executor.map(partial_intersections, chuncks))
    for result in results:
        horizontal_line.extend(result[0])
        h1e.extend(result[1])
    n = len(vl2)
    L=4
    k= max((n+L-1)//L,1)
    # Divide segments into chunks of size k
    segment_chunks = [vl2[i:i + k] for i in range(0, n, k)]
    element_chunks=  [v2e[i:i + k] for i in range(0, n, k)]
    chuncks=[]
    for i in range(len(segment_chunks)):
        chuncks.append([segment_chunks[i],element_chunks[i]])
    s_l=[len(c) for c in segment_chunks]
    print(len(segment_chunks))
    print(s_l)
    # Use ProcessPoolExecutor for parallel computation
    with ProcessPoolExecutor(max_workers=L) as executor:
        partial_intersections = partial(process_intersections2, segments=all_block,point_map=point_map,segmentation_config=segmentation_config)
        results = list(executor.map(partial_intersections, chuncks))
    for result in results:
        hl2.extend(result[0])
        h2e.extend(result[1])
  
    reference_lines=[]
    text_pos_map={}
    text_set=set()
    text_pos_map2={}
    text_set2=set()
    for i,line in enumerate(horizontal_line):
        if (len(point_map[line.start_point])==1 and  len(point_map[line.end_point])>1) or (len(point_map[line.start_point])>1 and  len(point_map[line.end_point])==1):
            if len(point_map[line.start_point])==1:
                p=line.end_point
            else:
                p=line.start_point
            
            ns=[s for s in point_map[p] if s!=line and s.length()>segmentation_config.reference_line_min_length] 
            sr=[s.ref for s in point_map[p] if s.length()>segmentation_config.reference_line_min_length  and (isinstance(s.ref, DLine) or isinstance(s.ref,DLwpolyline)) and s!=line and angleOfTwoSegmentsWithCommonStarter(p,s,line)>segmentation_config.reference_min_angle and angleOfTwoSegmentsWithCommonStarter(p,s,line)<segmentation_config.reference_max_angle]
            ss=[s for s in point_map[p] if s.length()>segmentation_config.reference_line_min_length  and (isinstance(s.ref, DLine) or isinstance(s.ref,DLwpolyline)) and s!=line and angleOfTwoSegmentsWithCommonStarter(p,s,line)>segmentation_config.reference_min_angle and angleOfTwoSegmentsWithCommonStarter(p,s,line)<segmentation_config.reference_max_angle]
            flag=checkReferenceLine(p,ns,ss,segmentation_config)
            if flag:
                refs=[]
                for j,s in enumerate(ss):
                    refs.append(sr[j])
                    start_point=p
                    current_point=s.start_point if s.start_point!=start_point else s.end_point
                    current_line=s
                    k=0
                    total_length=current_line.length()
                    while True:
                        # print(k)
                        k+=1
                        if k>10:
                            flag=False
                            break
                        if len(point_map[current_point])<=1:
                            break
                        current_nedge=[sss for sss in point_map[current_point] if sss!=current_line and (isinstance(sss.ref, DLine) or isinstance(sss.ref,DLwpolyline)) and is_parallel(current_line,sss,segmentation_config.is_parallel_tolerance) and (sss.length()>8 or check_in_arc(sss,point_map))]
                        if len(current_nedge)==0:
                            flag=False
                            break
                        current_line=current_nedge[0]
                        total_length+=current_line.length()
                        current_point=current_line.start_point if current_line.start_point!=current_point else current_line.end_point
                    # print(total_length)
                    if total_length<100 or total_length>3000:
                        flag=False
                    if flag and current_point not in text_pos_map:
                        text_pos_map[current_point]=set()
                        text_pos_map[current_point].add(h1e[i])
                        if h1e[i] not in text_set:
                            text_set.add(h1e[i])
                        h1e[i].textpos=True
                    elif flag:
                        text_pos_map[current_point].add(h1e[i])
                        if h1e[i] not in text_set:
                            text_set.add(h1e[i])
                        h1e[i].textpos=True
                    if flag==False:
                        break
                if flag:
                    reference_lines.extend(refs)
                    reference_lines.append(line.ref)
        else:

            p=line.start_point
            
            ns=[s for s in point_map[p] if s!=line and s.length()>segmentation_config.reference_line_min_length] 
            sr=[s.ref for s in point_map[p] if s.length()>segmentation_config.reference_line_min_length  and (isinstance(s.ref, DLine) or isinstance(s.ref,DLwpolyline)) and s!=line and angleOfTwoSegmentsWithCommonStarter(p,s,line)>segmentation_config.reference_min_angle and angleOfTwoSegmentsWithCommonStarter(p,s,line)<segmentation_config.reference_max_angle]
            ss=[s for s in point_map[p] if s.length()>segmentation_config.reference_line_min_length  and (isinstance(s.ref, DLine) or isinstance(s.ref,DLwpolyline)) and s!=line and angleOfTwoSegmentsWithCommonStarter(p,s,line)>segmentation_config.reference_min_angle and angleOfTwoSegmentsWithCommonStarter(p,s,line)<segmentation_config.reference_max_angle]
            flag1=checkReferenceLine(p,ns,ss,segmentation_config)
            cp1=[]
            refs1=[]
            if flag1:
                
                for j,s in enumerate(ss):
                    refs1.append(sr[j])
                    start_point=p
                    current_point=s.start_point if s.start_point!=start_point else s.end_point
                    current_line=s
                    k=0
                    total_length=current_line.length()
                    while True:
                        # print(k)
                        k+=1
                        if k>5:
                            flag1=False
                            break
                        if len(point_map[current_point])<=1:
                            break
                        current_nedge=[sss for sss in point_map[current_point] if sss!=current_line and (isinstance(sss.ref, DLine) or isinstance(sss.ref,DLwpolyline))  and is_parallel(current_line,sss,segmentation_config.is_parallel_tolerance) and (sss.length()>8 or check_in_arc(sss,point_map))]
                        if len(current_nedge)==0:
                            flag1=False
                            break
                        current_line=current_nedge[0]
                        total_length+=current_line.length()
                        current_point=current_line.start_point if current_line.start_point!=current_point else current_line.end_point
                    # print(total_length)
                    if total_length<100 or total_length>3000:
                        flag1=False
                    if flag1:
                        cp1.append(current_point)
                    # if flag and current_point not in text_pos_map:
                    #     text_pos_map[current_point]=set()
                    #     text_pos_map[current_point].add(h1e[i])
                    #     if h1e[i] not in text_set:
                    #         text_set.add(h1e[i])
                    #     h1e[i].textpos=True
                    # elif flag:
                    #     text_pos_map[current_point].add(h1e[i])
                    #     if h1e[i] not in text_set:
                    #         text_set.add(h1e[i])
                    #     h1e[i].textpos=True
                    if flag1==False:
                        break
                # if flag:
                #     reference_lines.extend(refs)
                #     reference_lines.append(line.ref)


            p=line.end_point
            
            ns=[s for s in point_map[p] if s!=line and s.length()>segmentation_config.reference_line_min_length] 
            sr=[s.ref for s in point_map[p] if s.length()>segmentation_config.reference_line_min_length  and (isinstance(s.ref, DLine) or isinstance(s.ref,DLwpolyline)) and s!=line and angleOfTwoSegmentsWithCommonStarter(p,s,line)>segmentation_config.reference_min_angle and angleOfTwoSegmentsWithCommonStarter(p,s,line)<segmentation_config.reference_max_angle]
            ss=[s for s in point_map[p] if s.length()>segmentation_config.reference_line_min_length  and (isinstance(s.ref, DLine) or isinstance(s.ref,DLwpolyline)) and s!=line and angleOfTwoSegmentsWithCommonStarter(p,s,line)>segmentation_config.reference_min_angle and angleOfTwoSegmentsWithCommonStarter(p,s,line)<segmentation_config.reference_max_angle]
            flag2=checkReferenceLine(p,ns,ss,segmentation_config)
            cp2=[]
            cn2=[]
            refs2=[]
            if flag2:
                
                for j,s in enumerate(ss):
                    refs2.append(sr[j])
                    start_point=p
                    current_point=s.start_point if s.start_point!=start_point else s.end_point
                    current_line=s
                    k=0
                    total_length=current_line.length()
                    while True:
                        # print(k)
                        k+=1
                        if k>5:
                            flag2=False
                            break
                        if len(point_map[current_point])<=1:
                            break
                        current_nedge=[sss for sss in point_map[current_point] if sss!=current_line and (isinstance(sss.ref, DLine) or isinstance(sss.ref,DLwpolyline))  and is_parallel(current_line,sss,segmentation_config.is_parallel_tolerance) and (sss.length()>8 or check_in_arc(sss,point_map))]
                        if len(current_nedge)==0:
                            flag2=False
                            break
                        current_line=current_nedge[0]
                        total_length+=current_line.length()
                        current_point=current_line.start_point if current_line.start_point!=current_point else current_line.end_point
                    # print(total_length)
                    if total_length<100 or total_length>3000:
                        flag2=False
                    if flag2:
                        cp2.append(current_point)
                        cn2.append(current_line)
                    # if flag and current_point not in text_pos_map:
                    #     text_pos_map[current_point]=set()
                    #     text_pos_map[current_point].add(h1e[i])
                    #     if h1e[i] not in text_set:
                    #         text_set.add(h1e[i])
                    #     h1e[i].textpos=True
                    # elif flag:
                    #     text_pos_map[current_point].add(h1e[i])
                    #     if h1e[i] not in text_set:
                    #         text_set.add(h1e[i])
                    #     h1e[i].textpos=True
                    if flag2==False:
                        break
                # if flag:
                #     reference_lines.extend(refs)
                #     reference_lines.append(line.ref)
            flag=flag1 and flag2
            if flag:
                for current_point in cp1+cp2:
                    if current_point not in text_pos_map:
                        text_pos_map[current_point]=set()
                        text_pos_map[current_point].add(h1e[i])
                        if h1e[i] not in text_set:
                            text_set.add(h1e[i])
                        h1e[i].textpos=True
                    else :
                        text_pos_map[current_point].add(h1e[i])
                        if h1e[i] not in text_set:
                            text_set.add(h1e[i])
                        h1e[i].textpos=True
                reference_lines.extend(refs1+refs2)
                reference_lines.append(line.ref)
    for i,line in enumerate(hl2):
        if (len(point_map[line.start_point])==1 and  len(point_map[line.end_point])>1) or (len(point_map[line.start_point])>1 and  len(point_map[line.end_point])==1):
            if len(point_map[line.start_point])==1:
                p=line.end_point
            else:
                p=line.start_point
            ns=[s for s in point_map[p] if s!=line and s.length()>segmentation_config.reference_line_min_length] 
            sr=[s.ref for s in point_map[p] if s.length()>segmentation_config.reference_line_min_length  and (isinstance(s.ref, DLine) or isinstance(s.ref,DLwpolyline)) and s!=line and angleOfTwoSegmentsWithCommonStarter(p,s,line)>segmentation_config.reference_min_angle and angleOfTwoSegmentsWithCommonStarter(p,s,line)<segmentation_config.reference_max_angle]
            ss=[s for s in point_map[p] if s.length()>segmentation_config.reference_line_min_length  and (isinstance(s.ref, DLine) or isinstance(s.ref,DLwpolyline)) and s!=line and angleOfTwoSegmentsWithCommonStarter(p,s,line)>segmentation_config.reference_min_angle and angleOfTwoSegmentsWithCommonStarter(p,s,line)<segmentation_config.reference_max_angle]
            flag=checkReferenceLine(p,ns,ss,segmentation_config)
            if flag:
                refs=[]
                for j,s in enumerate(ss):
                    refs.append(sr[j])
                    start_point=p
                    current_point=s.start_point if s.start_point!=start_point else s.end_point
                    current_line=s
                    total_length=current_line.length()
                    k=0
                    while True:
                        k+=1
                        if k>10:
                            flag=False
                            break
                        if len(point_map[current_point])<=1:
                            break
                        current_nedge=[sss for sss in point_map[current_point] if sss!=current_line and (isinstance(sss.ref, DLine) or isinstance(sss.ref,DLwpolyline))  and is_parallel(current_line,sss,segmentation_config.is_parallel_tolerance) and (sss.length()>8 or check_in_arc(sss,point_map))]
                        if len(current_nedge)==0:
                            flag=False
                            break
                        current_line=current_nedge[0]
                        total_length+=current_line.length()
                        current_point=current_line.start_point if current_line.start_point!=current_point else current_line.end_point
                    # print(total_length)
                    if total_length<100 or total_length>3000:
                        flag=False
                    if flag and current_point not in text_pos_map2:
                        text_pos_map2[current_point]=set()
                        text_pos_map2[current_point].add(h2e[i])
                        if h2e[i] not in text_set2:
                            text_set2.add(h2e[i])
                        h2e[i].textpos=True
                    elif flag:
                        text_pos_map2[current_point].add(h2e[i])
                        if h2e[i] not in text_set2:
                            text_set2.add(h2e[i])
                        h2e[i].textpos=True
                    if flag==False:
                        break
                if flag:
                    reference_lines.extend(refs)
                    reference_lines.append(line.ref)
        else:
            p=line.start_point
            ns=[s for s in point_map[p] if s!=line and s.length()>segmentation_config.reference_line_min_length] 
            sr=[s.ref for s in point_map[p] if s.length()>segmentation_config.reference_line_min_length  and (isinstance(s.ref, DLine) or isinstance(s.ref,DLwpolyline)) and s!=line and angleOfTwoSegmentsWithCommonStarter(p,s,line)>segmentation_config.reference_min_angle and angleOfTwoSegmentsWithCommonStarter(p,s,line)<segmentation_config.reference_max_angle]
            ss=[s for s in point_map[p] if s.length()>segmentation_config.reference_line_min_length  and (isinstance(s.ref, DLine) or isinstance(s.ref,DLwpolyline)) and s!=line and angleOfTwoSegmentsWithCommonStarter(p,s,line)>segmentation_config.reference_min_angle and angleOfTwoSegmentsWithCommonStarter(p,s,line)<segmentation_config.reference_max_angle]
            flag1=checkReferenceLine(p,ns,ss,segmentation_config)
            cp1=[]
            refs1=[]
            if flag1:
                for j,s in enumerate(ss):
                    refs1.append(sr[j])
                    start_point=p
                    current_point=s.start_point if s.start_point!=start_point else s.end_point
                    current_line=s
                    total_length=current_line.length()
                    k=0
                    while True:
                        k+=1
                        if k>5:
                            flag1=False
                            break
                        if len(point_map[current_point])<=1:
                            break
                        current_nedge=[sss for sss in point_map[current_point] if sss!=current_line and (isinstance(sss.ref, DLine) or isinstance(sss.ref,DLwpolyline))  and is_parallel(current_line,sss,segmentation_config.is_parallel_tolerance) and (sss.length()>8 or check_in_arc(sss,point_map))]
                        if len(current_nedge)==0:
                            flag1=False
                            break
                        current_line=current_nedge[0]
                        total_length+=current_line.length()
                        current_point=current_line.start_point if current_line.start_point!=current_point else current_line.end_point
                    # print(total_length)
                    if total_length<100 or total_length>3000:
                        flag1=False
                    if flag1:
                        cp1.append(current_point)
                    # if flag and current_point not in text_pos_map2:
                    #     text_pos_map2[current_point]=set()
                    #     text_pos_map2[current_point].add(h2e[i])
                    #     if h2e[i] not in text_set2:
                    #         text_set2.add(h2e[i])
                    #     h2e[i].textpos=True
                    # elif flag:
                    #     text_pos_map2[current_point].add(h2e[i])
                    #     if h2e[i] not in text_set2:
                    #         text_set2.add(h2e[i])
                    #     h2e[i].textpos=True
                    if flag1==False:
                        break
                # if flag:
                #     reference_lines.extend(refs)
                #     reference_lines.append(line.ref)

            p=line.end_point
            ns=[s for s in point_map[p] if s!=line and s.length()>segmentation_config.reference_line_min_length] 
            sr=[s.ref for s in point_map[p] if s.length()>segmentation_config.reference_line_min_length  and (isinstance(s.ref, DLine) or isinstance(s.ref,DLwpolyline)) and s!=line and angleOfTwoSegmentsWithCommonStarter(p,s,line)>segmentation_config.reference_min_angle and angleOfTwoSegmentsWithCommonStarter(p,s,line)<segmentation_config.reference_max_angle]
            ss=[s for s in point_map[p] if s.length()>segmentation_config.reference_line_min_length  and (isinstance(s.ref, DLine) or isinstance(s.ref,DLwpolyline)) and s!=line and angleOfTwoSegmentsWithCommonStarter(p,s,line)>segmentation_config.reference_min_angle and angleOfTwoSegmentsWithCommonStarter(p,s,line)<segmentation_config.reference_max_angle]
            flag2=checkReferenceLine(p,ns,ss,segmentation_config)
            cp2=[]
            refs2=[]
            if flag2:
                # refs=[]
                for j,s in enumerate(ss):
                    refs2.append(sr[j])
                    start_point=p
                    current_point=s.start_point if s.start_point!=start_point else s.end_point
                    current_line=s
                    total_length=current_line.length()
                    k=0
                    while True:
                        k+=1
                        if k>5:
                            flag2=False
                            break
                        if len(point_map[current_point])<=1:
                            break
                        current_nedge=[sss for sss in point_map[current_point] if sss!=current_line and (isinstance(sss.ref, DLine) or isinstance(sss.ref,DLwpolyline))  and is_parallel(current_line,sss,segmentation_config.is_parallel_tolerance) and (sss.length()>8 or check_in_arc(sss,point_map))]
                        if len(current_nedge)==0:
                            flag2=False
                            break
                        current_line=current_nedge[0]
                        total_length+=current_line.length()
                        current_point=current_line.start_point if current_line.start_point!=current_point else current_line.end_point
                    # print(total_length)
                    if total_length<100 or total_length>3000:
                        flag2=False
                    # if flag and current_point not in text_pos_map2:
                    #     text_pos_map2[current_point]=set()
                    #     text_pos_map2[current_point].add(h2e[i])
                    #     if h2e[i] not in text_set2:
                    #         text_set2.add(h2e[i])
                    #     h2e[i].textpos=True
                    # elif flag:
                    #     text_pos_map2[current_point].add(h2e[i])
                    #     if h2e[i] not in text_set2:
                    #         text_set2.add(h2e[i])
                    #     h2e[i].textpos=True
                    if flag2:
                        cp2.append(current_point)
                    if flag2==False:
                        break
                # if flag:
                #     reference_lines.extend(refs)
                #     reference_lines.append(line.ref)
            flag=flag1 and flag2
            if flag:
                for current_point in cp1+cp2:
                    if  current_point not in text_pos_map2:
                        text_pos_map2[current_point]=set()
                        text_pos_map2[current_point].add(h2e[i])
                        if h2e[i] not in text_set2:
                            text_set2.add(h2e[i])
                        h2e[i].textpos=True
                    else:
                        text_pos_map2[current_point].add(h2e[i])
                        if h2e[i] not in text_set2:
                            text_set2.add(h2e[i])
                        h2e[i].textpos=True
                reference_lines.extend(refs1+refs2)
                reference_lines.append(line.ref)
    print(len(reference_lines)*len(initial_segments))
    new_segments=[]
    removed_segments=[]
    removed_handles=[]
    for s in initial_segments:
        
        if s.ref not in reference_lines:
            new_segments.append(s)
            #print(s.ref)
        else:
            removed_segments.append(s)
            removed_handles.append(s.ref.handle)
    print("==============")
    print(len(removed_segments))
    return new_segments,reference_lines,text_pos_map,text_set,text_pos_map2,text_set2,removed_segments,removed_handles
    
# def convert_ref_to_tuple(ref):
#     """
#     Convert different ref objects (Line, Arc, etc.) to a unified tuple format for comparison.
#     """
#     if isinstance(ref, DLine):
#         # Convert Line to a tuple of its start and end points
#         return ('Line', (ref.start_point.x, ref.start_point.y), (ref.end_point.x, ref.end_point.y))
#     elif isinstance(ref, DArc):
#         # Convert Arc to a tuple of its defining properties: center, radius, start_angle, end_angle
#         return ('Arc', (ref.center.x, ref.center.y), ref.radius, ref.start_angle, ref.end_angle)
#     else:
#         # Handle other types or return a generic string for unknown types
#         return ('Unknown', str(ref))


def computeCenterCoordinates(poly):

    # w=0
    # x=0
    # y=0
    # for edge in poly:
    #     # if not isinstance(edge.ref,DArc):
    #         # l=edge.ref.weight
    #         # w=w+l
    #         # x=x+l*edge.ref.bc.x
    #         # y=y+l*edge.ref.bc.y
    #     l=edge.length()
    #     w+=l
    #     x+=l*(edge.start_point.x+edge.end_point.x)/2
    #     y+=l*(edge.start_point.y+edge.end_point.y)/2
    points = []
    for segment in poly:
        points.append(segment.start_point)
        points.append(segment.end_point)
        x = sum(point.x for point in points) / len(points)
        y = sum(point.y for point in points) / len(points)
   
    return DPoint(x,y)


#count the number of replines in a poly
def countReplines(poly,replines_set):
    count=0
    for edge in poly:
        if edge in replines_set:
            count+=1
    return count


def remove_duplicate_polygons(closed_polys,eps=25.0,min_samples=1):
    """
    Remove duplicate closed_polys based on the set of unique 'ref' values for each polygon.
    :param closed_polys: List of polygons, where each polygon is a list of Segment objects
    :return: Deduplicated list of polygons
    """
    if len(closed_polys)==0:
        return []
    unique_polys = []
    bcs=[]

    # seen_refs = set()  # To store unique sets of refs for comparison

    # for poly in closed_polys:
    #     # Convert all ref objects in the polygon to a unified comparable format
    #     refs = {convert_ref_to_tuple(seg.ref) for seg in poly}

    #     # If this set of refs is unique, add the polygon to the result
    #     refs_tuple = tuple(sorted(refs))  # Sorting to ensure consistent order for comparison

    #     if refs_tuple not in seen_refs:
    #         unique_polys.append(poly)
    #         seen_refs.add(refs_tuple)
    # poly_map={}
    # for poly in closed_polys:
    #     bc=computeCenterCoordinates(poly)
    #     #bc=DPoint(int(bc.x/span)*span,int(bc.y/span)*span)
    #     if bc not in poly_map:
    #         poly_map[bc]=poly
    #     else:
    #         if len(poly_map[bc])<len(poly):
    #             poly_map[bc]=poly
    tps=[]
    for poly in closed_polys:
        bc=computeCenterCoordinates(poly)
        tps.append((bc,poly))
    # for bc,poly in poly_map.items():
        # unique_polys.append(poly)
        # bcs.append(bc)
        # tps.append((bc,poly))
    # tps=sorted(tps,key=lambda item: item[0].as_tuple())
    
    #filter tps
    
    points =[]
    areas=[]
    for tp in tps:
        points.append([tp[0].x,tp[0].y])
        # replines_count.append(countReplines(tp[1],replines_set))
        areas.append(computeAreaOfPoly(tp[1]))
    #print(replines_count)
    points=np.array(points)


    # eps: 两点之间的最大距离，可以认为是邻居的距离。这个值可以根据你的数据集进行调整。
    # min_samples: 一个点被认为是核心点所需的最小邻居数。
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)

    labels = db.labels_

    poly_map={}
    #print("聚类标签：", labels)
    for idx,label in enumerate(labels):
        if label ==-1:
            unique_polys.append(tps[idx][1])
        else:
            if label not in poly_map:
                poly_map[label]=idx
            else:
                if areas[poly_map[label]]>areas[idx]:
                    #print(len(poly_map[label]),len(tps[idx][1]))
                    poly_map[label]=idx
    
    for label,idx in poly_map.items():
        unique_polys.append(tps[idx][1])
        #print(tps[idx][0])
    
    return unique_polys

def computeBoundingBox(poly):
    x_min,x_max=poly[0].start_point.x,poly[0].start_point.x
    y_min,y_max=poly[0].start_point.y,poly[0].start_point.y
    if x_min>poly[0].end_point.x:
        x_min=poly[0].end_point.x
    if x_max< poly[0].end_point.x:
        x_max=poly[0].end_point.x
    if y_min>poly[0].end_point.y:
        y_min=poly[0].end_point.y
    if y_max<poly[0].end_point.y:
        y_max=poly[0].end_point.y
    for i in range(len(poly)):
        if i==0:
            continue
        edge=poly[i]
        if x_min>edge.start_point.x:
            x_min=edge.start_point.x
        if x_max< edge.start_point.x:
            x_max=edge.start_point.x
        if y_min>edge.start_point.y:
            y_min=edge.start_point.y
        if y_max<edge.start_point.y:
            y_max=edge.start_point.y

        if x_min>edge.end_point.x:
            x_min=edge.end_point.x
        if x_max< edge.end_point.x:
            x_max=edge.end_point.x
        if y_min>edge.end_point.y:
            y_min=edge.end_point.y
        if y_max<edge.end_point.y:
            y_max=edge.end_point.y
    return x_min,x_max,y_min,y_max


# filter polys by area of polys's bounding boxes 
def filterPolys(polys,max_length=15,min_length=3,t_min=5000,t_max=1000*1500,d=5):
    
    for poly in polys:
        l=len(poly)
        for i in range(l-1):
            if poly[i].start_point==poly[i+1].start_point:
                p=poly[i].start_point
                poly[i].start_point=poly[i].end_point
                poly[i].end_point=p
            elif poly[i].start_point==poly[i+1].end_point:
                p=poly[i].start_point
                poly[i].start_point=poly[i].end_point
                poly[i].end_point=p
                p=poly[i+1].start_point
                poly[i+1].start_point=poly[i+1].end_point
                poly[i+1].end_point=p
            elif poly[i].end_point==poly[i+1].end_point:
                p=poly[i+1].start_point
                poly[i+1].start_point=poly[i+1].end_point
                poly[i+1].end_point=p
        if poly[-1].start_point==poly[0].start_point:
            p=poly[-1].start_point
            poly[-1].start_point=poly[-1].end_point
            poly[-1].end_point=p
        elif poly[-1].start_point==poly[0].end_point:
            p=poly[-1].start_point
            poly[-1].start_point=poly[-1].end_point
            poly[-1].end_point=p
            p=poly[0].start_point
            poly[0].start_point=poly[0].end_point
            poly[0].end_point=p
        elif poly[-1].end_point==poly[0].end_point:
            p=poly[0].start_point
            poly[0].start_point=poly[0].end_point
            poly[0].end_point=p
    filtered_polys=[]
    for poly in polys:

        if len(poly)>max_length or len(poly)<=min_length:
            continue
       
        #area and box
        x_min,x_max,y_min,y_max=computeBoundingBox(poly)
        area=computeAreaOfPoly(poly)
        # print(area)
        yy=y_max-y_min
        xx=x_max-x_min
        if yy<20 or xx<20:
            continue
        if yy>xx:
            div=yy/xx
        else:
            div=xx/yy
        if area<t_min or area>t_max or div>=d:
            continue
        #parellel line
        flag=False
        for i,segment in enumerate(poly):
            dx_1 = segment.end_point.x - segment.start_point.x
            dy_1 = segment.end_point.y - segment.start_point.y
            l=(dx_1**2+dy_1**2)**0.5
            v_1=(dy_1/l*50.0,-dx_1/l*50.0)
            mid_p=DPoint((segment.start_point.x+segment.end_point.x)/2,(segment.start_point.y+segment.end_point.y)/2)

            for j,other in  enumerate(poly):
                if i==j:
                    continue
                dx_2 = other.end_point.x - other.start_point.x
                dy_2 = other.end_point.y - other.start_point.y
                
                # 计算斜率
                k_1 = math.pi/2 if dx_1 == 0 else math.atan(dy_1 / dx_1)
                k_2 = math.pi/2 if dx_2 == 0 else math.atan(dy_2 / dx_2)
                if (math.fabs(dx_1) <=0.025 and math.fabs(dx_2) <=0.025) or (math.fabs(k_1-k_2)<=0.1):
                    s1=DSegment(DPoint(segment.start_point.x+v_1[0],segment.start_point.y+v_1[1]),DPoint(segment.start_point.x-v_1[0],segment.start_point.y-v_1[1]))
                    s2=DSegment(DPoint(segment.end_point.x+v_1[0],segment.end_point.y+v_1[1]),DPoint(segment.end_point.x-v_1[0],segment.end_point.y-v_1[1]))                                
                    s3=DSegment(DPoint(mid_p.x+v_1[0],mid_p.y+v_1[1]),DPoint(mid_p.x-v_1[0],mid_p.y-v_1[1]))
                    i1=segment_intersection(s1.start_point,s1.end_point,other.start_point,other.end_point)
                    if i1==other.end_point or i1==other.start_point:
                        i1=None
                    i2=segment_intersection(s2.start_point,s2.end_point,other.start_point,other.end_point)

                    if i2==other.end_point or i2==other.start_point:
                        i2=None
                    i3=segment_intersection(s3.start_point,s3.end_point,other.start_point,other.end_point)
                    
                    if i3==other.end_point or i3==other.start_point:
                        i3=None
                    if (i1 is not None) or (i2 is not None) or (i3 is not None):
                        flag=True
                        break
            if flag:
                break
        if not flag:
            filtered_polys.append(poly)
    
    #print(filtered_polys[0])
    valid_polys=filtered_polys

    # for poly in filtered_polys:
    #     valid=True
    #     for edge in poly:
    #         if edge in arc_replines_set:
    #             o=DPoint((edge.ref.start_point.x+edge.ref.end_point.x)/2,
    #               (edge.ref.start_point.y+edge.ref.end_point.y)/2)
    #             if  is_point_in_polygon(o,poly):
    #                 valid=False
    #                 break
    #     if  True:
    #         valid_polys.append(poly)

    return valid_polys

# 保留简单路径的多边形
def points_are_close(p1, p2, tolerance=1e-3):
    """
    判断两个点是否在一定误差范围内相等。
    :param p1: 第一个点 (DPoint)
    :param p2: 第二个点 (DPoint)
    :param tolerance: 容差（允许的误差范围）
    :return: 如果两个点的距离小于容差，则返回 True
    """
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5 < tolerance

def remove_complicated_polygons(polys, tolerance=1e-5):
    """
    去除复杂路径的多边形，确保每个点最多被访问两次（容差允许的范围内判断点相同）。
    :param polys: 多边形列表
    :param tolerance: 容差（允许的误差范围）
    :return: 保留简单路径的多边形列表
    """
    res = []

    for poly in polys:
        count = {}
        flag = True
        for seg in poly:
            # 判断start_point是否已存在于count中（基于误差范围判断）
            start_exists = False
            end_exists = False

            for point in count.keys():
                if points_are_close(seg.start_point, point, tolerance):
                    start_exists = True
                    count[point] += 1
                    if count[point] > 2:
                        flag = False
                    break

            if not start_exists:
                count[seg.start_point] = 1

            # 判断end_point是否已存在于count中（基于误差范围判断）
            for point in count.keys():
                if points_are_close(seg.end_point, point, tolerance):
                    end_exists = True
                    count[point] += 1
                    if count[point] > 2:
                        flag = False
                    break

            if not end_exists:
                count[seg.end_point] = 1

        if flag:
            res.append(poly)

    return res

def outputLines(segmentation_config,segments,point_map,polys,cornor_holes,star_pos,texts,text_map,dimensions,replines,linePNGPath,drawIntersections=False,drawLines=False,drawPolys=False):
    def p_minus(a,b):
        return DPoint(a.x-b.x,a.y-b.y)
    def p_add(a,b):
        return DPoint(a.x+b.x,a.y+b.y)
    def p_mul(a,k):
        return DPoint(a.x*k,a.y*k)
    if drawLines:
        for seg in segments:
            vs, ve = seg.start_point, seg.end_point
            plt.plot([vs.x, ve.x], [vs.y, ve.y], 'k-')
    if drawIntersections:
        for p,ss in point_map.items():
            if len(ss)>1:
                # print(p.x,p.y)
                plt.plot(p.x, p.y, 'r.')
    if drawPolys:
        for poly in polys:
            for edge in poly:
                x_values = [edge.start_point.x, edge.end_point.x]
                y_values = [edge.start_point.y, edge.end_point.y]
                plt.plot(x_values, y_values, 'b-', lw=2)
    for cornor_hole in cornor_holes:
        for edge in cornor_hole.segments:
            x_values = [edge.start_point.x, edge.end_point.x]
            y_values = [edge.start_point.y, edge.end_point.y]
            plt.plot(x_values, y_values, 'g-', lw=2)
    for p in star_pos:
        plt.plot(p.x, p.y, 'b.')
    # for p in braket_pos:
    #     plt.plot(p.x, p.y, 'g.')
    for edge in replines:
        x_values = [edge.start_point.x, edge.end_point.x]
        y_values = [edge.start_point.y, edge.end_point.y]
        plt.plot(x_values, y_values, 'r-', lw=2)
    if segmentation_config.draw_texts:
        for p,t in text_map.items():
            plt.plot(p.x, p.y, 'g.')
            plt.text(p.x, p.y, [tt[0].content for tt in t], fontsize=10)
        for d_t in dimensions:
            p=d_t[1]
            d=d_t[0]
            if d.dimtype==32 or d.dimtype==33 or d.dimtype==161 or d.dimtype==160:
                l0=p_minus(d.defpoints[0],d.defpoints[2])
                l1=p_minus(d.defpoints[1],d.defpoints[2])
                d10=l0.x*l1.x+l0.y*l1.y
                d00=l0.x*l0.x+l0.y*l0.y
                if d00 <1e-4:
                    x=d.defpoints[1]
                else:
                    x=p_minus(p_add(d.defpoints[1],l0),p_mul(l0,d10/d00))
                d1,d2,d3,d4=d.defpoints[0], x,d.defpoints[1],d.defpoints[2]

                # for i,p in enumerate([d1,d2,d3,d4]):
                #     if p!=DPoint(0,0):
                #         plt.text(p.x, p.y, str(i+1),color="#EE0000", fontsize=15)
                #         plt.plot(p.x, p.y, 'b.')
                ss=[DSegment(d1,d4),DSegment(d4,d3),DSegment(d3,d2)]
                sss=[DSegment(d2,d1)]
                ss=expandFixedLength(ss,25,True,False,True)
                sss=expandFixedLength(sss,100,True,False,True)
                q=sss[0].end_point
                sss=expandFixedLength(sss,100,False,False,True)
                for s in ss:
                    plt.plot([s.start_point.x,s.end_point.x], [s.start_point.y,s.end_point.y], color="#FF0000", lw=2,linestyle='--')
                for s in sss:
                    plt.plot([s.start_point.x,s.end_point.x], [s.start_point.y,s.end_point.y], color="#FF0000", lw=2,linestyle='--')
                plt.arrow(sss[0].end_point.x, sss[0].end_point.y, d1.x-sss[0].end_point.x, d1.y-sss[0].end_point.y, head_width=20, head_length=20, fc='red', ec='red')
                plt.arrow(sss[0].start_point.x, sss[0].start_point.y, d2.x-sss[0].start_point.x, d2.y-sss[0].start_point.y, head_width=20, head_length=20, fc='red', ec='red')
                perp_vx, perp_vy = sss[0].start_point.x - sss[0].end_point.x, sss[0].start_point.y-sss[0].end_point.y
                rotation_angle = np.arctan2(-perp_vy, -perp_vx) * (180 / np.pi)
                plt.text(q.x, q.y, d.text,rotation=rotation_angle,color="#EEC933", fontsize=15)
            elif d.dimtype==37:
                a,b,o=d.defpoints[1],d.defpoints[2],d.defpoints[3]
                r=DSegment(d.defpoints[0],o).length()
                r0=DSegment(a,o).length()
                oa_=p_mul(p_minus(a,o),r/r0)
                ob_=p_mul(p_minus(b,o),r/r0)
                ao_=p_mul(oa_,-1)
                bo_=p_mul(ob_,-1)
                a_= p_add(o, oa_)
                b_ = p_add(o, ob_)
                ia_=p_add(o,ao_)
                ib_=p_add(o,bo_)
                delta=p_mul(DPoint(oa_.y,-oa_.x),3)
                sp=p_add(delta,a_)
                plt.arrow(ia_.x, ia_.y, a_.x-ia_.x,a_.y-ia_.y, head_width=20, head_length=20, fc='red', ec='red')
                plt.arrow(ib_.x, ib_.y, b_.x-ib_.x,b_.y-ib_.y, head_width=20, head_length=20, fc='red', ec='red')
                plt.plot([sp.x,a_.x], [sp.y,a_.y], color="#FF0000", lw=2,linestyle='--')
                q=p_mul(p_add(a_,sp),0.5)
                rotation_angle = np.arctan2(-delta.y, delta.x) * (180 / np.pi)
                plt.text(q.x, q.y, d.text,rotation=rotation_angle,color="#EEC933", fontsize=15)
            elif d.dimtype==163:
                a,b=d.defpoints[0],d.defpoints[3]
                o=p_mul(p_add(a,b),0.5)
                ss=[DSegment(a,b)]
                ss=expandFixedLength(ss,100,True,False,True)
                ss=expandFixedLength(ss,100,False,False,True)
                a_,b_=ss[0].start_point,ss[0].end_point
                plt.arrow(a_.x, a_.y, a.x-a_.x,a.y-a_.y, head_width=20, head_length=20, fc='red', ec='red')
                plt.arrow(b_.x, b_.y, b.x-b_.x,b.y-b_.y, head_width=20, head_length=20, fc='red', ec='red')
                q=p_add(p_mul(b_,0.7),p_mul(b,0.3))
                ab=p_minus(b,a)
                rotation_angle = np.arctan2(ab.y, -ab.x) * (180 / np.pi)
                plt.text(q.x, q.y, d.text,rotation=rotation_angle,color="#EEC933", fontsize=15)
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

                
                # if d.dimtype==34:
                #     for i,p in enumerate([d1,d2,d3,d4]):
                #         if p!=DPoint(0,0):
                #             plt.text(p.x, p.y, str(i+1),color="#EE0000", fontsize=15)
                #             plt.plot(p.x, p.y, 'b.')
                ss1=DSegment(inter,p1)
                ss2=DSegment(inter,p2)

                plt.arrow(ss1.start_point.x, ss1.start_point.y, ss1.end_point.x-ss1.start_point.x,ss1.end_point.y-ss1.start_point.y, head_width=20, head_length=20, fc='red', ec='red')
                plt.arrow(ss2.start_point.x, ss2.start_point.y, ss2.end_point.x-ss2.start_point.x,ss2.end_point.y-ss2.start_point.y, head_width=20, head_length=20, fc='red', ec='red')
                plt.text(pos.x, pos.y, d.text,rotation=rotation_angle,color="#EEC933", fontsize=15)
                plt.plot(inter.x,inter.y,'g.')
                plt.plot([p1.x,p2.x], [p1.y,p2.y], color="#FF0000", lw=2,linestyle='--')
                # a,b_,b,o=d.defpoints[0],d.defpoints[1],d.defpoints[2],d.defpoints[3]
                # r=DSegment(d.defpoints[4],o).length()
                # ra=DSegment(a,o).length()
                # rb=DSegment(b,o).length()
                # oa_=p_mul(p_minus(a,o),r/ra)
                # ob_=p_mul(p_minus(b,o),r/rb)
                # ao_=p_mul(oa_,-1)
                # bo_=p_mul(ob_,-1)
                # a_= p_add(o, oa_)
                # b_ = p_add(o, ob_)
                # ia_=p_add(o,ao_)
                # ib_=p_add(o,bo_)
                # delta=p_mul(DPoint(oa_.y,-oa_.x),3)
                # sp=p_add(delta,a_)
                # plt.arrow(ia_.x, ia_.y, a_.x-ia_.x,a_.y-ia_.y, head_width=20, head_length=20, fc='red', ec='red')
                # plt.arrow(ib_.x, ib_.y, b_.x-ib_.x,b_.y-ib_.y, head_width=20, head_length=20, fc='red', ec='red')
                # plt.plot([sp.x,a_.x], [sp.y,a_.y], color="#FF0000", lw=2,linestyle='--')
                # q=p_mul(p_add(a_,sp),0.5)
                # rotation_angle = np.arctan2(delta.y, -delta.x) * (180 / np.pi)
                # plt.text(q.x, q.y, d.text,rotation=rotation_angle,color="#EEC933", fontsize=15)
        #plt.plot(p.x, p.y, marker='o', markerfacecolor="#E38C35",markersize=10)
        
    # for cornor_hole in cornor_holes:
    #     print(cornor_hole.segments)
    plt.gca().axis('equal')
    plt.savefig(linePNGPath)
    plt.close('all')
    print(f"直线图保存于:{linePNGPath}")

def outputPolysAndGeometry(point_map,polys,path,draw_polys=False,draw_geometry=False,n=10):
    t=len(polys) if len(polys)<n else n
    
    if draw_geometry:
        pbar=tqdm(total=t,desc="输出几何")
        for i,poly in enumerate(polys):
            if i>=n:
                break
            plot_geometry(poly,os.path.join(path,f"geometry{i}.png"))
            pbar.update()
        pbar.close()
   
    if draw_polys:
        pbar=tqdm(total=t,desc="输出多边形")
        for i,poly in enumerate(polys):
            if i>=n:
                break
            plot_polys(point_map,poly,os.path.join(path,f"poly{i}.png"))
            pbar.update()
        pbar.close()
    print(f"封闭多边形图像保存于:{path}")

def removeOddPoints(filtered_segments,filtered_point_map,segmentation_config):
    edge_set=set()
    for p,ne in filtered_point_map.items():
        for e in ne:
            edge_set.add(e)
    for p,ne in filtered_point_map.items():
        degp=len(ne)
        if p==2 and checkRefAndSlope(ne,segmentation_config.is_parallel_tolerance_neighobor)[0]:
            a=ne[0].start_point if ne[0].start_point!=p else ne[0].end_point
            b=ne[1].start_point if ne[1].start_point!=p else ne[1].end_point
            new_edge=DSegment(a,b,ne[0].ref)
            edge_set.remove(ne[0])
            edge_set.remove(ne[1])
            edge_set.add(new_edge)
    new_segments=list(edge_set)
    new_point_map={}
    for s in new_segments:
        vs,ve=s.start_point,s.end_point
        if vs not in new_point_map:
            new_point_map[vs]=set()
        if ve not in new_point_map:
            new_point_map[ve]=set()
        new_point_map[vs].add(s)
        new_point_map[ve].add(s)
    for p,ss in new_point_map.items():
        new_point_map[p]=list(ss)
    return new_segments,new_point_map
    
def process_repline(repline, graph, segmentation_config):
    """
    处理单个 repline 的逻辑，使用 BFS 查找路径并构成闭合路径。
    """
    start_point = repline.start_point
    end_point = repline.end_point
    # 使用 BFS 查找从 start_point 到 end_point 的所有路径
    paths = bfs_paths(graph, start_point, end_point, segmentation_config.path_max_length,segmentation_config.timeout)

    # 构成闭合路径
    for path in paths:
        path.append(repline)
    return paths

def process_text_map(text_map,removed_segments,segmentation_config):
    new_text_map={}
    text_pos_map1,text_pos_map2,text_pos_map3=text_map["top"][1],text_map["bottom"][1],text_map["other"][1]
    for p,texts in text_pos_map1.items():
        for t in texts:
            result=parse_elbow_plate(t.content, annotation_position="top")
            if result is None:
                result={
                    "Type":"None"
                }
            mid_point=DPoint((t.bound["x1"]+t.bound["x2"])/2,(t.bound["y1"]+t.bound["y2"])/2)
            x,y=mid_point.x,mid_point.y
            x_1=(x+t.bound["x1"])/2
            x_2=(x+t.bound["x2"])/2
            l1=DSegment(DPoint(x_1,y),DPoint(x_1,y-500))
            l2=DSegment(DPoint(x_2,y),DPoint(x_2,y-500))
            l3=DSegment(DPoint(x,y),DPoint(x,y-500))
            d_min=None
            for i, seg1 in enumerate([l1,l2,l3]):
                y_max=None
                s=None
                for j, seg2 in enumerate(removed_segments):
                    p1, p2 = seg1.start_point, seg1.end_point
                    q1, q2 = seg2.start_point, seg2.end_point
                    intersection = segment_intersection(p1, p2, q1, q2)
                    if intersection:
                        if y_max is None:
                            y_max=intersection[1]
                            s=seg2
                        else:
                            if y_max<intersection[1]:
                                y_max=intersection[1]
                                s=seg2
                if s is not None:
                    text_pos=seg1.start_point
                    if (text_pos.y-y_max)<=segmentation_config.reference_text_max_distance :
                        d=math.fabs(text_pos.y-y_max)
                        if d_min is None:
                            d_min=d
                        else:
                            if d_min>d:
                                d_min=d




                       
            if p not in new_text_map and d_min is not None:
                new_text_map[p]=[]
                new_text_map[p].append([t,result,"top",d_min])
            elif d_min is not None:
                new_text_map[p].append([t,result,"top",d_min])

    for p,texts in text_pos_map2.items():
        for t in texts:
            result=parse_elbow_plate(t.content, annotation_position="bottom")
            if result is None:
                result={
                    "Type":"None"
                }
            mid_point=DPoint((t.bound["x1"]+t.bound["x2"])/2,(t.bound["y1"]+t.bound["y2"])/2)
            x,y=mid_point.x,mid_point.y
            x_1=(x+t.bound["x1"])/2
            x_2=(x+t.bound["x2"])/2
            l1=DSegment(DPoint(x_1,y),DPoint(x_1,y+500))
            l2=DSegment(DPoint(x_2,y),DPoint(x_2,y+500))
            l3=DSegment(DPoint(x,y),DPoint(x,y+500))
            d_min=None
            for i, seg1 in enumerate([l1,l2,l3]):
                y_min=None
                s=None
                for j, seg2 in enumerate(removed_segments):
                    p1, p2 = seg1.start_point, seg1.end_point
                    q1, q2 = seg2.start_point, seg2.end_point
                    intersection = segment_intersection(p1, p2, q1, q2)
                    if intersection:
                        if y_min is None:
                            y_min=intersection[1]
                            s=seg2
                        else:
                            if y_min>intersection[1]:
                                y_min=intersection[1]
                                s=seg2
                if s is not None:
                    text_pos=seg1.start_point
                    if (text_pos.y-y_min)<=segmentation_config.reference_text_max_distance :
                        d=math.fabs(text_pos.y-y_min)
                        if d_min is None:
                            d_min=d
                        else:
                            if d_min>d:
                                d_min=d
            if p not in new_text_map and d_min is not None:
                new_text_map[p]=[]
                new_text_map[p].append([t,result,"bottom",-d_min])
            elif d_min is not None:
                new_text_map[p].append([t,result,"bottom",-d_min])
    for p,texts in text_pos_map3.items():
        for t in texts:
            result=parse_elbow_plate(t.content, annotation_position="other")
            if result is None:
                result={
                    "Type":"None"
                }
            
            if p not in new_text_map:
                new_text_map[p]=[]
                new_text_map[p].append([t,result,"other",None])
            else:
                new_text_map[p].append([t,result,"other",None])
    new_text_map_={}
    for p ,texts in new_text_map.items():
        new_texts=[]
        rts=[]
        for t in texts:
            if t[3] is not None and t[1]["Type"]=="R":
                rts.append((abs(t[3]),t))
            else:
                new_texts.append(t)
        rt=None
        idx=None
        distance=float("inf")
        for i,tt in enumerate(rts):
            dis,t=tt
            if dis<distance:
                distance=dis
                idx=i
                rt=t
        if rt is not None:
            new_texts.append(rt)
            for i,tt in enumerate(rts):
                dis,t = tt
                if i!=idx:
                    pos=DPoint((t[0].bound['x1']+t[0].bound['x2'])/2,(t[0].bound['y1']+t[0].bound['y2'])/2)
                    if pos not in new_text_map_:
                        new_text_map_[pos]=[]
                    new_text_map_[pos].append([t[0],parse_elbow_plate(t[0].content,annotation_position="other"),"other",None])
        text_wo_d=[]
        text_w_d=[]
        text_map={}
        for t in new_texts:
            if t[3] is None:
                text_wo_d.append(t)
            else:
                if t[1]["Type"]=="R" or t[1]["Type"]=="BK" or t[1]["Type"]=="B_anno" or t[1]["Type"]=="s_star" or t[1]["Type"]=="m_star":
                    text_wo_d.append(t)
                else:
                    content=t[0].content.strip()
                    if content not in text_map:
                        text_map[content]=(t[3],t)
                    else:
                        if text_map[content][0]<t[3]:
                            text_map[content]=(t[3],t)
        for content,t_t in text_map.items():
            text_w_d.append(t_t[1])

        text_w_d=sorted(text_w_d,key=lambda t:t[3],reverse=True)
        new_text_w_d=[]
        if len(text_w_d)>=1:
            for i in range(len(text_w_d)):
                if i==0:
                    #top
                    t=text_w_d[i][0]
                    result=parse_elbow_plate(t.content, annotation_position="top")
                    if result is None:
                        result={
                            "Type":"None"
                    }
                    new_text_w_d.append([t,result,"top",text_w_d[i][3]])
                if i==1:
                    #bottom
                    t=text_w_d[i][0]
                    result=parse_elbow_plate(t.content, annotation_position="bottom")
                    if result is None:
                        result={
                            "Type":"None"
                    }
                    new_text_w_d.append([t,result,"bottom",text_w_d[i][3]])
                if i>1:
                    new_text_w_d.append(text_w_d[i])
        else:
            new_text_w_d=text_w_d
        new_text_map[p]=[]
        new_text_map[p].extend(new_text_w_d)
        new_text_map[p].extend(text_wo_d)
    for p,texts in new_text_map_.items():
        if p not in new_text_map:
            new_text_map[p]=[]
        new_text_map[p].extend(texts)
    new_text_map_={}
    for p,texts in new_text_map.items():
        new_texts=[]
        for t in texts:
            if t[1] is None:
                new_texts.append([t[0],{"Type":"None"},t[2],t[3]])
            else:
                new_texts.append(t)
        new_text_map_[p]=new_texts
    return new_text_map_

def get_segment_blocks(segment: DSegment, rect, M, N):
    """
    计算给定线段在矩形框划分的 MxN 网格中占据的块。
    
    参数:
        segment: DSegment 实例，线段。
        rect: (rect_x_min, rect_x_max, rect_y_min, rect_y_max)，矩形框范围。
        M: 列数（块的宽方向划分）。
        N: 行数（块的高方向划分）。
    
    返回:
        set: 占据的块索引集合，形式为 (row, col)。
    """
    rect_x_min, rect_x_max, rect_y_min, rect_y_max = rect
    cell_width = (rect_x_max - rect_x_min) / M
    cell_height = (rect_y_max - rect_y_min) / N

    # 获取线段的起点和终点
    x0, y0 = segment.start_point.x, segment.start_point.y
    x1, y1 = segment.end_point.x, segment.end_point.y

    # 计算步长比例
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    # 按网格调整步长
    sx = cell_width if x0 < x1 else -cell_width
    sy = cell_height if y0 < y1 else -cell_height
    sx=sx/4
    sy=sy/4
    # 初始化误差项（考虑网格尺寸）
    err = dx / cell_width - dy / cell_height

    # 存储占据的块索引
    grids = set()
    final_col_index = int((x1 - rect_x_min) / cell_width)
    final_row_index = int((y1 - rect_y_min) / cell_height)
    while True:
        # 计算当前点所属的块索引
        col_index = int((x0 - rect_x_min) / cell_width)
        row_index = int((y0 - rect_y_min) / cell_height)

        # 确保索引在范围内
        if 0 <= col_index < M and 0 <= row_index < N:
            # print(x0,y0,x1,y1,dx,dy)
            # print((x0-rect_x_min)/cell_width,(y0-rect_y_min)/cell_height)
            # print(col_index,row_index,final_col_index,final_row_index)
            grids.add((row_index, col_index))
        else:
            #assert(1==2)
            grids.add((final_row_index, final_col_index))
            break
        # 终止条件
        
        if col_index == final_col_index and row_index == final_row_index:
            break
        if col_index==final_col_index:
            if row_index==final_row_index+1 or row_index==final_row_index-1:
                grids.add((row_index, col_index))
                grids.add((final_row_index, final_col_index))
                break
        if row_index == final_row_index:
            if col_index==final_col_index+1 or col_index==final_col_index-1:
                grids.add((row_index, col_index))
                grids.add((final_row_index, final_col_index))
                break
        # 更新误差项和当前点
        e2 = 2 * err
        if e2 > -dy / cell_height:
            err -= dy / cell_height
            x0 += sx
        if e2 < dx / cell_width:
            err += dx / cell_width
            y0 += sy

    return grids


def segments_in_blocks(segments,segmentation_config):
    rect_x_min, rect_x_max, rect_y_min, rect_y_max=float("inf"),float("-inf"),float("inf"),float("-inf")
    M,N=segmentation_config.M,segmentation_config.N
    for s in segments:
        vs,ve=s.start_point,s.end_point
        rect_x_min=min(vs.x,ve.x,rect_x_min)
        rect_x_max=max(vs.x,ve.x,rect_x_max)
        rect_y_min=min(vs.y,ve.y,rect_y_min)
        rect_y_max=max(vs.y,ve.y,rect_y_max)
    rect_x_min-=segmentation_config.x_padding
    rect_x_max+=segmentation_config.x_padding
    rect_y_min-=segmentation_config.y_padding
    rect_y_max+=segmentation_config.y_padding
    rect=(rect_x_min, rect_x_max, rect_y_min, rect_y_max)
    #print(rect)
    grid=[]
    for i in range(M):
        row=[]
        for j in range(N):
            col=[]
            row.append(col)
        grid.append(row)
    pbar=tqdm(total=len(segments),desc="test")
    for s in segments:
        pbar.update()
        block_idxs=get_segment_blocks(s,rect,M,N)
        for idx in block_idxs:
            i,j=idx
            grid[i][j].append(s)
    pbar.close()
    return grid,(rect,M,N)


def segments_near_poly(poly,grid,meta):
    grid_set=set()
    rect,M,N=meta
    for s in poly:
        g_set=get_segment_blocks(s,rect,M,N)
        for g in g_set:
            grid_set.add(g)
    segments=[]
    for g in grid_set:
        i,j=g
        segments.extend(grid[i][j])
    return segments,list(grid_set)

def visualize_grid_and_segment(segments, poly,rect, M, N, blocks):
    """
    Visualizes the grid and highlights the segment and occupied blocks.

    Parameters:
        segment: The DSegment instance to visualize.
        rect: Tuple defining the rectangular area (x_min, x_max, y_min, y_max).
        M: Number of columns in the grid.
        N: Number of rows in the grid.
        blocks: The set of occupied blocks as (row, col) indices.
    """
    rect_x_min, rect_x_max, rect_y_min, rect_y_max = rect
    cell_width = (rect_x_max - rect_x_min) / M
    cell_height = (rect_y_max - rect_y_min) / N

    # Create the grid
    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(M + 1):
        x = rect_x_min + i * cell_width
        ax.plot([x, x], [rect_y_min, rect_y_max], color='black', linestyle='--', linewidth=0.5)
    for j in range(N + 1):
        y = rect_y_min + j * cell_height
        ax.plot([rect_x_min, rect_x_max], [y, y], color='black', linestyle='--', linewidth=0.5)

    # Highlight the occupied blocks
    for row, col in blocks:
        x = rect_x_min + col * cell_width
        y = rect_y_min + row * cell_height
        ax.add_patch(plt.Rectangle((x, y), cell_width, cell_height, color='blue', alpha=0.3))

    # Plot the segment
    for segment in segments:
        x0, y0 = segment.start_point.x, segment.start_point.y
        x1, y1 = segment.end_point.x, segment.end_point.y
        ax.plot([x0, x1], [y0, y1], color='green', linewidth=2)
    for segment in poly:
        x0, y0 = segment.start_point.x, segment.start_point.y
        x1, y1 = segment.end_point.x, segment.end_point.y
        ax.plot([x0, x1], [y0, y1], color='red', linewidth=2)
    # Set axis limits and labels
    ax.set_xlim(rect_x_min, rect_x_max)
    ax.set_ylim(rect_y_min, rect_y_max)
    ax.set_aspect('equal')
    ax.set_title("Grid and Segment Visualization")
    ax.legend()
    plt.show()
def findClosedPolys_via_BFS(elements,texts,dimensions,segments,sign_handles,segmentation_config):
    verbose=segmentation_config.verbose
    # Step 1: 计算交点
    # if verbose:
    #     print("计算交点")
    isecDic = find_all_intersections(segments,segmentation_config,segmentation_config.intersection_epsilon)

    # Step 2: 根据交点分割线段
    # if verbose:
    #     print("根据交点分割线段")
    new_segments, edge_map,point_map= split_segments(segments,segmentation_config, isecDic,segmentation_config.segment_filter_length)
    filtered_segments, filtered_edge_map,filtered_point_map= filter_segments(segments,segmentation_config,isecDic,point_map,segmentation_config.segment_filter_length,segmentation_config.segment_filter_iters,segmentation_config.segment_remove_interval,[])
    
    
    
    #remove rfernce lines
    initial_segments,reference_lines,text_pos_map1,text_set1,text_pos_map2,text_set2,removed_segments,removed_handles=removeReferenceLines(elements,texts,segments,new_segments,point_map,segmentation_config)
    text_set3=set()
    text_pos_map3={}
    for t in texts:
        if t not in text_set1 and t not in text_set2:
            pos=DPoint((t.bound['x1']+t.bound['x2'])/2,(t.bound['y1']+t.bound['y2'])/2)
            if pos not in text_pos_map3:
                text_pos_map3[pos]=set()
                text_pos_map3[pos].add(t)
            else:
                text_pos_map3[pos].add(t)
    text_map={}
    for pos,ts in text_pos_map3.items():
        text_pos_map3[pos]=list(ts)
    text_map["top"]=(text_set1,text_pos_map1)
    text_map["bottom"]=(text_set2,text_pos_map2)
    text_map["other"]=(text_set3,text_pos_map3)
    text_map=process_text_map(text_map,removed_segments,segmentation_config)
    isecDic = find_all_intersections(initial_segments,segmentation_config,segmentation_config.intersection_epsilon)
    new_segments, edge_map,point_map= split_segments(initial_segments,segmentation_config, isecDic,segmentation_config.segment_filter_length)
    #filter lines

    filtered_segments, filtered_edge_map,filtered_point_map= filter_segments(initial_segments,segmentation_config,isecDic,point_map,segmentation_config.segment_filter_length,segmentation_config.segment_filter_iters,segmentation_config.segment_remove_interval,sign_handles)
    braket_start_lines=findBraketByHints(filtered_segments,text_map)
    # polys=[]
    # outputLines(new_segments,point_map,polys,segmentation_config.line_image_path,segmentation_config.draw_intersections,segmentation_config.draw_segments,segmentation_config.line_image_drawPolys)

    #filtered_segments,filtered_point_map=removeOddPoints(filtered_segments,filtered_point_map,segmentation_config)

    # Step 3: 构建基于分割后线段的图结构
    if verbose:
        print("构建基于分割后线段的图结构")
        print(f"过滤后线段条数:{len(filtered_segments)}")
    graph= build_graph(filtered_segments)

    closed_polys = []

    # 基于角隅孔计算参考边
    #arc_replines = compute_arc_replines(filtered_segments,point_map)
    star_replines,star_pos_map,star_pos=compute_star_replines(filtered_segments,elements)
    #line_replines=compute_line_replines(filtered_segments,filtered_point_map)
    cornor_holes=compute_cornor_holes(filtered_segments,filtered_point_map,segmentation_config)
    cornor_holes=filter_cornor_holes(cornor_holes,filtered_point_map,segmentation_config)
    # for cornor_hole in cornor_holes:
    #     for s in cornor_hole.segments:
    #         s.isCornerhole=True

    if verbose:
        #print(f"圆弧角隅孔个数: {len(arc_replines)}")
        print(f"星形角隅孔个数: {len(star_replines)}")
        print(f"非星形角隅孔个数: {len(cornor_holes)}")
        #print(f"直线形角隅孔个数: {len(line_replines)}")
        print(f"根据标注找到的肘板个数: {len(braket_start_lines)}")
    # for star_repline in star_replines:
    #     print(star_repline)
    #replines=arc_replines+star_replines+line_replines+braket_start_lines
    replines=[]
    for cornor_hole in cornor_holes:
        replines.append(cornor_hole.segments[0])
    star_replines=findBracketByPoints(star_pos,filtered_segments)
    # braket_start_lines=findBracketByPoints(braket_pos,filtered_segments)
    replines=replines+braket_start_lines+star_replines
    replines_set=set()
    #arc_replines_set=set()
    #line_replines_set=set()
    # for arc_rep in arc_replines:
    #     if arc_rep not in arc_replines_set:
    #         arc_replines_set.add(arc_rep)
    #         arc_replines_set.add(DSegment(arc_rep.end_point,arc_rep.start_point))
    # for line_rep in line_replines:
    #     if line_rep not in line_replines_set:
    #         line_replines_set.add(line_rep)
    #         line_replines_set.add(DSegment(line_rep.end_point,line_rep.start_point))
    for rep in replines:
        if rep not in replines_set:
            replines_set.add(rep)
            replines_set.add(DSegment(rep.end_point,rep.start_point))
    # Step 4: 对每个 repline，使用 BFS 查找路径
    # 多线程加速部分
    # if verbose:
    #     pbar = tqdm(total=len(replines), desc="查找闭合路径")

    # # 使用 ThreadPoolExecutor
    # closed_polys = []
    # multi_thread_start_time = time.time()
    # errors=[]
    # if verbose:
    #     print(min(32, os.cpu_count())if segmentation_config.max_workers <=0 else segmentation_config.max_workers)
    # with ProcessPoolExecutor(max_workers=min(32, os.cpu_count()  )if segmentation_config.max_workers <=0 else segmentation_config.max_workers) as executor:
    #     # 提交任务到线程池
    #     future_to_repline = {executor.submit(process_repline, repline, graph, segmentation_config): repline for repline in replines}
        
    #     for future in as_completed(future_to_repline):
    #         repline = future_to_repline[future]
    #         try:
    #             # 获取结果，设置超时时间
    #             paths = future.result()
    #             closed_polys.extend(paths)
    #             # print(f"任务 {repline} 完成")
    #         except Exception as exc:
    #             errors.append(f"处理 {repline} 时发生错误: {exc}")
    #         if verbose:
    #             pbar.update()
    
    # # print("============")
    
    # # print("=========")
    # # closed_polys=[]
    # # for repline in replines:
    # #     paths=bfs_paths(graph,repline.start_point,repline.end_point,segmentation_config.path_max_length,timeout=1000)
    # #     closed_polys.extend(paths)
    # #     pbar.update()
    # if verbose:
    #     pbar.close()
    #     print(errors)
    # multi_thread_end_time = time.time()
    # if verbose:
    #     print(f"回路探测执行部分耗时: {multi_thread_end_time - multi_thread_start_time:.2f} 秒")
    #     print("查找完毕")
    # for closed_poly in closed_polys:
    #     print(closed_poly)
    #HIDDENs
    # if segmentation_config.dfs_optional:
    #     sampled_lines=[]
    #     for s in filtered_segments:
    #         sampled_lines.append(DSegment(s.start_point,s.end_point,s.ref))
    #         sampled_lines.append(DSegment(s.end_point,s.start_point,s.ref))
    #     visited_edges=set()
    #     if verbose:
    #         pbar=tqdm(desc="sampled_lines",total=len(sampled_lines))
    #     for sampled_line in sampled_lines:
    #         if verbose:
    #             pbar.update()
    #         visited_edges.add((sampled_line[0],sampled_line[1]))
    #         path=process_repline_with_repline_dfs(visited_edges,sampled_line,graph,segmentation_config)
    #         if(len(path)>=segmentation_config.path_min_length):
    #             closed_polys.append(path)
    #     if verbose:
    #         pbar.close()

    closed_polys=[]
    sampled_lines=[]
    # repline_has_visited=set()
    for s in replines:
        sampled_lines.append(DSegment(s.start_point,s.end_point,s.ref))
        sampled_lines.append(DSegment(s.end_point,s.start_point,s.ref))
    
    if verbose:
        pbar=tqdm(desc="sampled_lines(all the cornor holes)",total=len(sampled_lines))
    for sampled_line in sampled_lines:
        if verbose:
            pbar.update()
        visited_edges=set()
        visited_edges.add((sampled_line[0],sampled_line[1]))
        path=process_repline_with_repline_dfs(visited_edges,sampled_line,graph,segmentation_config)
        if(len(path)>=segmentation_config.path_min_length):
            closed_polys.append(path)
            # repline_has_visited.add(sampled_line)
            # repline_has_visited.add(DSegment(sampled_line.end_point,sampled_line.start_point))
    if verbose:
        pbar.close()
    sampled_lines=[]
    for s in filtered_segments:
        sampled_lines.append(DSegment(s.start_point,s.end_point,s.ref))
        sampled_lines.append(DSegment(s.end_point,s.start_point,s.ref))
    
    if verbose:
        pbar=tqdm(desc="sampled_lines(all the segments)",total=len(sampled_lines))
    for sampled_line in sampled_lines:
        if verbose:
            pbar.update()
        visited_edges=set()
        visited_edges.add((sampled_line[0],sampled_line[1]))
        path=process_repline_with_repline_dfs(visited_edges,sampled_line,graph,segmentation_config)
        if(len(path)>=segmentation_config.path_min_length):
            closed_polys.append(path)
    if verbose:
        pbar.close()
    
    # repline_unvisited=[]
    # for s in replines:
    #     if s not in repline_has_visited:
    #         repline_unvisited.append(s)

    # if verbose:
    #     pbar = tqdm(total=len(replines), desc="查找闭合路径")

    # multi_thread_start_time = time.time()
    # errors=[]
    # with ProcessPoolExecutor(max_workers=4 ) as executor:
    #     # 提交任务到线程池
    #     future_to_repline = {executor.submit(process_repline, repline, graph, segmentation_config): repline for repline in repline_unvisited}
        
    #     for future in as_completed(future_to_repline):
    #         repline = future_to_repline[future]
    #         try:
    #             # 获取结果，设置超时时间
    #             paths = future.result()
    #             closed_polys.extend(paths)
    #             # print(f"任务 {repline} 完成")
    #         except Exception as exc:
    #             errors.append(f"处理 {repline} 时发生错误: {exc}")
    #         if verbose:
    #             pbar.update()
    
    # # print("============")
    
    # # print("=========")
    # # closed_polys=[]
    # # for repline in replines:
    # #     paths=bfs_paths(graph,repline.start_point,repline.end_point,segmentation_config.path_max_length,timeout=1000)
    # #     closed_polys.extend(paths)
    # #     pbar.update()
    # if verbose:
    #     pbar.close()
    #     print(errors)
    # multi_thread_end_time = time.time()
    # if verbose:
    #     print(f"回路探测执行部分耗时: {multi_thread_end_time - multi_thread_start_time:.2f} 秒")
    #     print("查找完毕")

    
    #poly simplify
    print(len(closed_polys))

    # 根据边框对多边形进行过滤
    #polys = filterPolys(polys,t=3000,d=5)
    if segmentation_config.bracket_layer is None:
        polys = filterPolys(closed_polys,segmentation_config.dfs_path_max_length,segmentation_config.dfs_path_min_length,segmentation_config.bbox_min_area,segmentation_config.bbox_max_area,segmentation_config.bbox_ratio)
    print(len(polys))
    # 剔除重复路径
    polys = remove_duplicate_polygons(polys,segmentation_config.eps,segmentation_config.min_samples)
    #print(bcs)
    # print(polys[0])
    # print(polys[1])
    # 仅保留基本路径
    polys = remove_complicated_polygons(polys,segmentation_config.remove_tolerance)
    if verbose:
        print(f"封闭多边形个数:{len(polys)}")
    #outputPolysAndGeometry(filtered_point_map,polys,segmentation_config.poly_image_dir,segmentation_config.draw_polys,segmentation_config.draw_geometry,segmentation_config.draw_poly_nums)
    if segmentation_config.draw_line_image and segmentation_config.mode=="dev":
        outputLines(segmentation_config,segments,filtered_point_map,closed_polys,cornor_holes,star_pos ,texts,text_map,dimensions,replines,segmentation_config.line_image_path,segmentation_config.draw_intersections,segmentation_config.draw_segments,segmentation_config.line_image_drawPolys,)
    
    return polys, new_segments, point_map,star_pos_map,cornor_holes,text_map,removed_handles


 
def is_numeric(s):
    return bool(re.match(r'^[0-9]+$', s))
 
def is_r_numeric(s):
    return bool(re.match(r'^R[0-9]+$', s))

def is_material(s):
    return bool(re.match(r"^[A-Z]{2}$",s)) or bool(re.match(r"^~[A-Z]{2}$",s)) or bool(re.match(r"^[0-9]{2}[A-Z]{2}$",s))  or bool(re.match(r"^~[0-9]{2}[A-Z]{2}$",s))
# def isUsefulHints(e):
#     if (not isinstance(e,DText)) and (not isinstance(e,DDimension)):
#         return False
#     if  isBraketHints(e):
#         return False
#     content=e.content if isinstance(e,DText) else e.text
#     return is_numeric(content) or is_r_numeric(content)


# def findAllTextAndDimensions(elements):
#     text_and_dimensions=[e for e in elements if isUsefulHints(e)]
#     return text_and_dimensions


def isUsefulHints(e):
    # flag=True

    #return is_numeric(e.content) or is_r_numeric(e.content) or isBraketHints(e) or is_material(e.content)
    return is_useful_text(e.content)
def findAllTextsAndDimensions(elements):
    texts=[]
    dimensions=[]
    for e in elements:
        if isinstance(e,DDimension):
            dimensions.append(e)
        elif isinstance(e,DText):
            texts.append(e)
    return texts,dimensions
def processTexts(texts):
    ts=[]
    for t in texts:
        if isUsefulHints(t):
        # if True:
            ts.append(t)
    return ts
def processDimensions(dimensions):
    ds=[]
    for d in dimensions:
        type=d.dimtype
        if type==32 or type==33 or type==161 or type==160:
            #转角标注& 对齐标注
            dim_pos=DPoint((d.defpoints[1].x+d.defpoints[2].x)/2,(d.defpoints[1].y+d.defpoints[2].y)/2)
            ds.append([d,dim_pos]) 
        elif type==37 or type==34 or type==162:
            #三点角度标注
            dim_pos=DPoint(d.defpoints[4].x,d.defpoints[4].y)
            ds.append([d,dim_pos]) 
        elif type==38:
            #坐标标注
            pass
        elif type==163:
            #角度标注
            dim_pos=DPoint((d.defpoints[0].x+d.defpoints[3].x)/2,(d.defpoints[0].y+d.defpoints[3].y)/2)
            ds.append([d,dim_pos]) 
    return ds
def computePolygon_(poly,tolerance = 0.1):
    polygon_points = list()  # Concave polygon example
    for edge in poly:
        vs,ve=edge.start_point,edge.end_point
        polygon_points.append((vs.x,vs.y))
        polygon_points.append((ve.x,ve.y))
    polygon = Polygon(polygon_points)
    
    # polygon_with_tolerance = polygon.buffer(tolerance)

    # return polygon_with_tolerance
    return polygon
def  check_intersects(bound, bb_poly):
    polygon=computePolygon_(bb_poly)
    for p in bound:
        x,y=p[0],p[1]
        point=Point(x,y)
        if not polygon.contains(point):
            return False
    return True

def readJson_inbbpolys(path,segmentation_config, bb_polys):
    # elements=[]
    block_sub_datas=[]
    # arcs=[]
    # segments=[]
    # color = [3, 7, 8, 4,2,140]
    # linetype = ["BYLAYER", "Continuous","Bylayer","CONTINUOUS","ByBlock","BYBLOCK"]
    # elementtype=["line","arc","lwpolyline","polyline","spline"]
    # layname=["Stiffener_Invisible"]
    try:  
        with open(path, 'r', encoding='utf-8') as file:  
            data_list = json.load(file)

        block_datas=data_list[1]

        # 对data_list[0]中的元素根据bb_polys进行过滤
        data_list_filtered = []
        for ele in data_list[0]:
            # 构造当前元素的bound多边形
            bound = [
                [ele["bound"]["x1"], ele["bound"]["y1"]],
                [ele["bound"]["x1"], ele["bound"]["y2"]],
                [ele["bound"]["x2"], ele["bound"]["y2"]],
                [ele["bound"]["x2"], ele["bound"]["y1"]]
            ]
            # 检查与每个bb_poly是否相交
            for bb_poly in bb_polys:
                if check_intersects(bound, bb_poly):
                    data_list_filtered.append(ele)
                    break
        print("===================")
        print(len(data_list_filtered))
        elements,segments,arc_splits,ori_segments,stiffeners=process_block(False,block_datas,"TOP",[1.0,1.0],0,[0,0],"CONTINUOUS",[],{"x1":float("-inf"),"x2":float("inf"),"y1":float("-inf"),"y2":float("inf")},data_list_filtered,segmentation_config)
        print("===============")
        print(len(elements))
        new_segments=[]
        new_arc_splits=[]
        new_ori_segments=[]
        for s in segments:
            if s.ref.linetype in segmentation_config.line_type:
                new_segments.append(s)
        for s in arc_splits:
            if s.ref.linetype in segmentation_config.line_type:
                new_arc_splits.append(s)
        for s in ori_segments:
            if s.ref.linetype in segmentation_config.line_type:
                new_ori_segments.append(s)
        segments=new_segments
        arc_splits=new_arc_splits
        ori_segments=new_ori_segments
        segments=expandFixedLength(segments,segmentation_config.line_expand_length)
        arc_splits=expandFixedLength(arc_splits,segmentation_config.arc_expand_length)
        
        sign_handles=[]
        polyline_handles=[]
        hatch_polys = []
        for ele in data_list_filtered:
            if ele["type"]=="lwpolyline":
                vs=ele["vertices"]
                vs_type=ele["verticesType"]
                vs_width=ele["verticesWidth"]
                if len(vs)==3 and len(vs_type)==3 and vs_type==["line","line","line"] and len(vs_width)==4 and vs_width[1]==[0,0] and vs_width[3]==[0,0] and vs_width[0][0]==0 and vs_width[0][1]>0 and vs_width[2][0]>0 and vs_width[2][1]==0:
                    start=DPoint(vs[0][0],vs[0][1])
                    end=DPoint(vs[-1][2],vs[-1][3])
                    if DSegment(start,end).length()>100 and DSegment(start,end).length() <1000:
                        sign_handles.append(ele["handle"])
                elif len(vs)==5 and len(vs_type)==5 and vs_type==["line","line","line","line","line"] and len(vs_width)==6 and vs_width[0]==[0,0] and vs_width[2]==[0,0] and vs_width[4]==[0,0] and vs_width[5]==[0,0] and vs_width[3][0]==0 and vs_width[3][1]>0 and vs_width[1][0]>0 and vs_width[1][1]==0:
                    start=DPoint(vs[0][0],vs[0][1])
                    end=DPoint(vs[-1][2],vs[-1][3])
                    if DSegment(start,end).length()>100 and DSegment(start,end).length() <1000:
                        sign_handles.append(ele["handle"])
            elif ele["type"]=="polyline":
                polyline_handles.append(ele["handle"])
            elif ele["type"]=="hatch":
                hatch_paths = ele["paths"]
                for hatch_edges in hatch_paths:
                    hatch_poly = []
                    for hatch_edge in hatch_edges:
                        if hatch_edge["edge_type"] == "line":
                            point = [[hatch_edge["coords"][0], hatch_edge["coords"][1]], [hatch_edge["coords"][2], hatch_edge["coords"][3]]]
                            hatch_poly.extend(point)
                        elif hatch_edge["edge_type"] == "polyline":
                            hatch_poly.extend(hatch_edge["coords"])
                    hatch_polys.append(hatch_poly)
       

        #inner block 
        jg_s = set()
        for name,sub_data in  data_list[1].items():
            st_prefixs = ["L", "HP"]
            if any(name.startswith(prefix) for prefix in st_prefixs):
                for ele in sub_data:
                    jg_s.add(ele["handle"])
                
            for ele in sub_data:
                if ele["type"]=="lwpolyline":
                    vs=ele["vertices"]
                    vs_type=ele["verticesType"]
                    vs_width=ele["verticesWidth"]
                    if len(vs)==3 and len(vs_type)==3 and vs_type==["line","line","line"] and len(vs_width)==4 and vs_width[1]==[0,0] and vs_width[3]==[0,0] and vs_width[0][0]==0 and vs_width[0][1]>0 and vs_width[2][0]>0 and vs_width[2][1]==0:
                        start=DPoint(vs[0][0],vs[0][1])
                        end=DPoint(vs[-1][2],vs[-1][3])
                        if DSegment(start,end).length()>100 and DSegment(start,end).length() <1000:
                            sign_handles.append(ele["handle"])
                    elif len(vs)==5 and len(vs_type)==5 and vs_type==["line","line","line","line","line"] and len(vs_width)==6 and vs_width[0]==[0,0] and vs_width[2]==[0,0] and vs_width[4]==[0,0] and vs_width[5]==[0,0] and vs_width[3][0]==0 and vs_width[3][1]>0 and vs_width[1][0]>0 and vs_width[1][1]==0:
                        start=DPoint(vs[0][0],vs[0][1])
                        end=DPoint(vs[-1][2],vs[-1][3])
                        if DSegment(start,end).length()>100 and DSegment(start,end).length() <1000:
                            sign_handles.append(ele["handle"])

        
                
        return elements,segments+arc_splits,ori_segments,stiffeners,sign_handles,polyline_handles,hatch_polys,jg_s
    except FileNotFoundError:  
        print("The file does not exist.")
    except json.JSONDecodeError:  
        print("Error decoding JSON.")

def intersects(poly_a, poly_b):
    """判断两个凸多边形是否相交，基于分离轴定理（SAT）"""
    
    def get_axes(poly):
        """获取多边形所有边的法线轴"""
        axes = []
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1) % n]
            dx = x2 - x1
            dy = y2 - y1
            # 边的法线轴（垂直于边方向）
            axis = (-dy, dx)
            # 忽略零向量（当两点重合时）
            if axis == (0, 0):
                continue
            axes.append(axis)
        return axes
    
    def project(poly, axis):
        """将多边形投影到轴上，返回投影区间的[min, max]"""
        min_proj = max_proj = poly[0][0] * axis[0] + poly[0][1] * axis[1]
        for x, y in poly[1:]:
            proj = x * axis[0] + y * axis[1]
            if proj < min_proj:
                min_proj = proj
            if proj > max_proj:
                max_proj = proj
        return (min_proj, max_proj)
    
    def overlaps(a, b):
        """判断两个区间是否有重叠"""
        return a[0] <= b[1] and b[0] <= a[1]
    
    # 获取所有需要检测的分离轴
    axes = get_axes(poly_a) + get_axes(poly_b)
    
    for axis in axes:
        # 投影两个多边形到当前轴
        a_proj = project(poly_a, axis)
        b_proj = project(poly_b, axis)
        
        # 如果存在不重叠的投影，则多边形不相交
        if not overlaps(a_proj, b_proj):
            return False
    # 所有轴上投影均重叠，则多边形相交
    return True
def get_hole_text_coor(json_path, hole_layer):
    cornerhole_prefixs = ["KS", "KSNR", "KSNKS", "VU", "VUF", "VUR", "R", "RC"]
    text_coors = []
    try:  
        with open(json_path, 'r', encoding='utf-8') as file:  
            data_list = json.load(file)
        block_elements=data_list[0]
        for ele in block_elements:
            if ele["layerName"]!=hole_layer:
                continue
            if ele["type"]=="text":
                content = ele["content"]
                if any(content.startswith((prefix) for prefix in cornerhole_prefixs)):
                    continue
                else:
                    coor = ((ele["bound"]["x1"] + ele["bound"]["x2"]) / 2, (ele["bound"]["y1"] + ele["bound"]["y2"]) / 2)
                    text_coors.append(coor)
        
    except FileNotFoundError:  
        print("The file does not exist.")
    except json.JSONDecodeError:  
        print("Error decoding JSON.")
    
    return text_coors


def find_dump_bracket_ids(polys_info, classi_res, indices):
    dump_bracket_ids = []
    # 计算bbox
    actual_bboxs = []
    valid_classes = []
    valid_indices = []
    
    for idx, (poly_refs, cls) in enumerate(zip(polys_info, classi_res)):
        max_x = float('-inf')
        min_x = float('inf')
        max_y = float('-inf')
        min_y = float('inf')
        for seg in poly_refs:
            x_coords = [seg.start_point[0], seg.end_point[0]]
            y_coords = [seg.start_point[1], seg.end_point[1]]
            max_x = max(max_x, *x_coords)
            min_x = min(min_x, *x_coords)
            max_y = max(max_y, *y_coords)
            min_y = min(min_y, *y_coords)
        
        # 排除不合法分类
        if cls == 'Unclassified' or cls == 'Unstandard' or ',' in cls or 'ustd' in cls:
            continue

        actual_bboxs.append((min_x - 20, max_x + 20, min_y - 20, max_y + 20))
        valid_classes.append(cls)
        valid_indices.append(indices[idx])

    def compute_iou_area(b1, b2):
        x1 = max(b1[0], b2[0])
        x2 = min(b1[1], b2[1])
        y1 = max(b1[2], b2[2])
        y2 = min(b1[3], b2[3])

        if x1 >= x2 or y1 >= y2:
            return 0.0, 0.0, 0.0  # no overlap

        inter_area = (x2 - x1) * (y2 - y1)
        area1 = (b1[1] - b1[0]) * (b1[3] - b1[2])
        area2 = (b2[1] - b2[0]) * (b2[3] - b2[2])
        return inter_area, area1, area2

    def same_main_class(cls1, cls2):
        return cls1.split('(')[0] == cls2.split('(')[0]


    def count_cor_inside_bracket(cls_str):
        if '(' in cls_str and ')' in cls_str:
            inside = cls_str.split('(', 1)[1].split(')', 1)[0]
            inside = inside.split('-')
            count = 0
            for cor in inside:
                if cor != 'KS':
                    count += 1
            return count
        return 0

    n = len(actual_bboxs)
    for i in range(n):
        for j in range(i + 1, n):
            inter_area, area_i, area_j = compute_iou_area(actual_bboxs[i], actual_bboxs[j])
            if area_i == 0 or area_j == 0:
                continue
            if (inter_area / area_i > 0.9) and (inter_area / area_j > 0.9):
                if same_main_class(valid_classes[i], valid_classes[j]):
                    # 去除KS角孔含量最多的分类结果
                    ks_i = count_cor_inside_bracket(valid_classes[i])
                    ks_j = count_cor_inside_bracket(valid_classes[j])
                    if ks_i > ks_j:
                        dump_bracket_ids.append(valid_indices[j])
                    elif ks_j > ks_i:
                        dump_bracket_ids.append(valid_indices[i])
                    else:
                        dump_bracket_ids.append(valid_indices[j])  # 默认去除靠后的

    return dump_bracket_ids
