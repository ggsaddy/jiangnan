import json
import random
from element import *
import copy


def is_vertical_(point1,point2,segment,epsilon=0.05):
    v1=DPoint(point1.x-point2.x,point1.y-point2.y)
    v2=DPoint(segment.start_point.x-segment.end_point.x,segment.start_point.y-segment.end_point.y)
    cross_product=(v1.x*v2.x+v1.y*v2.y)/(DSegment(point1,point2).length()*segment.length()+1e-4)
    if  abs(cross_product) <epsilon:
        return True
    return False 
def is_tangent(line,arc):
    s1,s2=DSegment(arc.ref.center,arc.ref.start_point),DSegment(arc.ref.center,arc.ref.end_point)
    v1,v2=line.start_point,line.end_point
    if v1==arc.ref.start_point or v2==arc.ref.start_point:
        return is_vertical_(v1,v2,s1)
    if v1==arc.ref.end_point or v2==arc.ref.end_point:
        return is_vertical_(v1,v2,s2)
    return True

def is_free_edges_equal(free_edges_sequence, temp):
    if free_edges_sequence == temp:
        return True
    free_edges_sequence_copy = copy.deepcopy(free_edges_sequence)
    free_edges_sequence_copy2 = copy.deepcopy(free_edges_sequence)
    free_edges_sequence_copy2.append("KS_corner")
    free_edges_sequence_copy.insert(0, "KS_corner")
    if free_edges_sequence_copy2 == temp:
        return True
    if free_edges_sequence_copy == temp:
        return True
    free_edges_sequence_copy2.insert(0, "KS_corner")
    if free_edges_sequence_copy2 == temp:
        return True
    return False

def find_anno_info(matched_type,all_anno,poly_free_edges):
    c_anno,radius_anno,whole_anno,half_anno,cornor_anno,parallel_anno,non_parallel_anno,vertical_anno,d_anno,angle_anno,toe_angle_anno=all_anno
    anno={}
    if ",DAC(VU-R)," in matched_type:
        if len(cornor_anno)==0:
            anno[",DAC(VU-R),"]="no_anno"
        else:
            anno[",DAC(VU-R),"]="short_anno"
    else:
        anno[",DAC(VU-R),"]="none"
    # elif "DAB(VU-KS)" in matched_type:
    #     if len(angle_anno)==0:
    #         return "no_angle"
    #     else:
    #         return "angl"
    # print(matched_type)
    if ",DPK(R)," in matched_type:
        if len(parallel_anno)!=0:
            anno[",DPK(R),"]="short_anno_para"
        elif len(half_anno)!=0:
            anno[",DPK(R),"]= "short_anno"
        else:
            anno[",DPK(R),"]= "long_anno"
    else:
        anno[",DPK(R),"]="none"
    # elif "DAB(R-KS)" in matched_type:
    #     if len(cornor_anno)==0:
    #         return "no_anno"
    #     else:
    #         return "short_anno"
    if ",DPKN(KS-KS)," in matched_type:
        if len(half_anno)!=0:
            anno[",DPKN(KS-KS),"]="short_anno"
        else:
            anno[",DPKN(KS-KS),"]="long_anno"
    else:
        anno[",DPKN(KS-KS),"]="none"
    if ",DAD(R)," in matched_type:
        if len(half_anno)!=0:
            anno[",DAD(R),"]="dist_intersec"
        else:
            anno[",DAD(R),"]="no_dist"
    else:
        anno[",DAD(R),"]="none"
    if ",DPKN(KS-R)," in matched_type:
        if len(half_anno)!=0:
            anno[",DPKN(KS-R),"]="short_anno"
        else:
            anno[",DPKN(KS-R),"]="long_anno"
    else:
        anno[",DPKN(KS-R),"]="none"
    if ",DAB-3(R-KS)," in matched_type:
        free_edges=poly_free_edges[0]
        line=free_edges[1] if isinstance(free_edges[1],DLine) else free_edges[2]
        arc=free_edges[1] if isinstance(free_edges[1],DArc) else free_edges[2]
        if isinstance(line,DLine) and isinstance(arc,DArc) and is_tangent(line,arc):
            if len(angle_anno)!=0:
                anno[",DAB-3(R-KS),"]="angl_non_free"
            else:
            
                anno[",DAB-3(R-KS),"]="dist_intersec"
        else:
            anno[",DAB-3(R-KS),"]="no_tangent"
    else:
        anno[",DAB-3(R-KS),"]="none"
    if ",KL(R)," in matched_type:
        if len(vertical_anno)==2:
            anno[",KL(R),"]="tt"
        elif len(vertical_anno)==1:
            anno[",KL(R),"]= "tf"
        else:
            anno[",KL(R),"]= "ff"
    else:
        anno[",KL(R),"]="none"
    if ",KL(KS)," in matched_type:
        if len(vertical_anno)==2:
            anno[",KL(KS),"]= "tt"
        elif len(vertical_anno)==1:
            anno[",KL(KS),"]= "tf"
        else:
            anno[",KL(KS),"]= "ff"
    else:
        anno[",KL(KS),"]="none"
    if ",BR(R)," in matched_type:
        if len(angle_anno)!=0:
            anno[",BR(R),"]="angl_non_free"
        elif len(half_anno)!=0:
            anno[",BR(R),"]= "dist_intersec"
        else:
            anno[",BR(R),"]= "no_anno"
    else:
        anno[",BR(R),"]="none"
    if ",DPK(R-KS)," in matched_type:
        if len(half_anno)!=0:
            anno[",DPK(R-KS),"]="short_anno"
        else:
            anno[",DPK(R-KS),"]= "long_anno"
    else:
        anno[",DPK(R-KS),"]="none"
    if ",BR-1(KS)," in matched_type:
        if len(angle_anno)!=0:
            anno[",BR-1(KS),"]="angl_non_free"
        else:
            anno[",BR-1(KS),"]= "dist_intersec"
    else:
        anno[",BR-1(KS),"]="none"
    if ",LBMA-1(KS)," in matched_type:
        if len(angle_anno)!=0:
            anno[",LBMA-1(KS),"]="angl"
        else:
            anno[",LBMA-1(KS),"]= "dist"
    else:
        anno[",LBMA-1(KS),"]="none"
    if ",DPV-4(R-KS)," in matched_type:
        if len(parallel_anno)!=0:
            anno[",DPV-4(R-KS),"]="D_anno"
        else:
            anno[",DPV-4(R-KS),"]= "no_anno"
    else:
        anno[",DPV-4(R-KS),"]="none"
    if ",DAC(KS-KS)," in matched_type:
        if len(d_anno)!=0:
            anno[",DAC(KS-KS),"]="D"
        else:
            anno[",DAC(KS-KS),"]="notD"
    else:
        anno[",DAC(KS-KS),"]="none"
    if ",DAC(R-R)," in matched_type:
        if len(d_anno)!=0:
            anno[",DAC(R-R),"]="D_anno"
        else:
            anno[",DAC(R-R),"]= "no_anno"
    else:
        anno[",DAC(R-R),"]="none"
    if ",DPK-1(R)," in matched_type:
        if len(non_parallel_anno)!=0:
            anno[",DPK-1(R),"]="dist_adja"
        elif len(toe_angle_anno)!=0:
            anno[",DPK-1(R),"]= "angl_toe"
        elif len(half_anno)!=0:
            anno[",DPK-1(R),"]= "dist_intersec"
        else:
            anno[",DPK-1(R),"]= "angl_non_free"
    else:
        anno[",DPK-1(R),"]="none"
    if ",DPKN(VU-R)," in matched_type:
       
        if len(half_anno)!=0:
            anno[",DPKN(VU-R),"]="short_anno"
        else:
            anno[",DPKN(VU-R),"]= "long_anno"
    else:
        anno[",DPKN(VU-R),"]="none"
    if ",DPKN(KS-KS)," in matched_type:
           
        if len(half_anno)!=0:
            anno[",DPKN(KS-KS),"]="short_anno"
        else:
            anno[",DPKN(KS-KS),"]= "long_anno"
    else:
        anno[",DPKN(KS-KS),"]="none"
    if ",DPK(VU)," in matched_type:
           
        if len(half_anno)!=0:
            anno[",DPK(VU),"]="short_anno"
        elif len(parallel_anno)!=0:
            anno[",DPK(VU),"]= "short_anno_para"
        else:
            anno[",DPK(VU),"]="long_anno"
    else:
        anno[",DPK(VU),"]="none"
    
    if ",DME-1(R-KS)," in matched_type:
        if len(angle_anno)!=0:
            anno[",DME-1(R-KS),"]="angl"
        else:
            anno[",DME-1(R-KS),"]="dist"
    else:
        anno[",DME-1(R-KS),"]="none"
    
    
    return anno

def load_classification_table(file_path):
    """
    Load the classification table from a JSON file.
    """
    with open(file_path, 'r',encoding='UTF-8') as f:
        classification_table = json.load(f)  # Load the JSON file
    return classification_table


def generate_key(edge):
    # Generate a key that considers both original and reversed order
    if isinstance(edge[0], list):
        new_edge = edge[0]
    else:
        new_edge = edge
    return min(tuple(new_edge), tuple(reversed(new_edge)))


def is_tangent_(line,arc):
    s1,s2=DSegment(arc.ref.center,arc.ref.start_point),DSegment(arc.ref.center,arc.ref.end_point)
    v1,v2=line.start_point,line.end_point
    if v1==arc.ref.start_point or v2==arc.ref.start_point:
        return is_vertical_(v1,v2,s1,0.1)
    if v1==arc.ref.end_point or v2==arc.ref.end_point:
        return is_vertical_(v1,v2,s2,0.1)
    return True
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



# 自由边约束过滤
def free_edges_sequence_classifier(classification_table, free_edges_sequence, reversed_free_edges_sequence, matched_type_list):
    matched_type = None
    # 自由边有趾端的匹配
    for b_type in matched_type_list:
        temp_free_edge_seq = classification_table[b_type]["free_edges"]
        if is_free_edges_equal(free_edges_sequence, temp_free_edge_seq) or is_free_edges_equal(reversed_free_edges_sequence, temp_free_edge_seq):
            matched_type = b_type if matched_type is None else f'{matched_type},{b_type}'
    
    # 自由边无趾端的匹配
    # if matched_type is None:
    #     free_edges_sequence_copy = copy.deepcopy(free_edges_sequence)
    #     reversed_free_edges_sequence_copy = copy.deepcopy(reversed_free_edges_sequence)
    #     if free_edges_sequence_copy[0] == "toe":
    #         free_edges_sequence_copy[0] = "line"
    #     if free_edges_sequence_copy[-1] == "toe":
    #         free_edges_sequence_copy[-1] = "line"
    #     if reversed_free_edges_sequence_copy[0] == "toe":
    #         reversed_free_edges_sequence_copy[0] = "line"
    #     if reversed_free_edges_sequence_copy[-1] == "toe":
    #         reversed_free_edges_sequence_copy[-1] = "line"
    #     for b_type in matched_type_list:
    #         temp_free_edge_seq = classification_table[b_type]["free_edges"]
    #         if is_free_edges_equal(free_edges_sequence_copy, temp_free_edge_seq) or is_free_edges_equal(reversed_free_edges_sequence_copy, temp_free_edge_seq):
    #             matched_type = b_type if matched_type is None else f'{matched_type},{b_type}'
    
    return matched_type if matched_type is not None else "Unclassified"

# 固定边和角隅孔约束匹配
def conerhole_free_classifier(classification_table, conerhole_num, free_edges_sequence, reversed_free_edges_sequence, edges_sequence, reversed_edges_sequence,is_ustd=False):
    matched_type = []
    non_conerhole_edges = []
    reversed_non_conerhole_edges = []
    conerhole_count = {}
    unrestricted_cornerhole_count = {}
    # unrestricted_cornerhole_type = [["line"], ["arc"]]
    if is_ustd:
        unrestricted_cornerhole_type = [["line"], ["arc"]]
        current=[["line"], ["arc"]]
        base=["line","arc"]
        for i in range(9):
            new_current=[]
            for t in base:
                for c in current:
                    new_current.append(c+[t])
            current=new_current
            unrestricted_cornerhole_type.extend(current)
    else:

        unrestricted_cornerhole_type = []
    unrestricted_cornerhole_num = 0
    # 去掉非自由边轮廓中的角隅孔，只保留固定边，对角隅孔进行统计
    for i in range(len(edges_sequence)):
        if(edges_sequence[i][0] != "cornerhole"):
            non_conerhole_edges.append(edges_sequence[i][1])
        # 记录角隅孔字典
        else:
            key = generate_key(edges_sequence[i][1])
            conerhole_count[key] = conerhole_count.get(key, 0) + 1
            if edges_sequence[i][1] not in unrestricted_cornerhole_type:
                unrestricted_cornerhole_count[key] = unrestricted_cornerhole_count.get(key, 0) + 1
            else:
                unrestricted_cornerhole_num += 1
        if(reversed_edges_sequence[i][0] != "conerhole"):
            reversed_non_conerhole_edges.append(reversed_edges_sequence[i][1])

    # 固定边轮廓严格匹配+角隅孔各类型数量的不严格匹配（直线角隅孔可能不画）
    for key_name, row in classification_table.items():
        # step1: 自由边轮廓严格匹配
        # if not is_free_edges_equal(free_edges_sequence, row["free_edges"]) and not is_free_edges_equal(reversed_free_edges_sequence, row["free_edges"]):
        #     continue

        # step2: 非自由边轮廓去除角隅孔后严格匹配
        temp_sequence = []
        temp_conerhole_count = {}
        temp_unrestricted_cornerhole_num = 0
        for i, non_free in enumerate(row["non_free_edges"]):
            # Check if the type and edges of the current non-free match in sequence
            if non_free["type"] == "constraint":
                temp_sequence.append(non_free["edges"])
            elif non_free["type"] == "cornerhole":
                if not (not isinstance(non_free["edges"][0], list) and (non_free["edges"] in unrestricted_cornerhole_type)):
                    key = generate_key(non_free["edges"])
                    temp_conerhole_count[key] = temp_conerhole_count.get(key, 0) + 1
                else:
                    temp_unrestricted_cornerhole_num += 1

        if temp_sequence != non_conerhole_edges and temp_sequence != reversed_non_conerhole_edges:
            continue

        # step3: 角隅孔计数匹配
        if temp_conerhole_count != unrestricted_cornerhole_count:
            continue

        matched_type.append(key_name)

    return matched_type


def tidy_matched_type(matched_type):
    types=matched_type.split(',')
    result=''
    for ty in types:
        if ty!='':
            result=result+ty+','
    if result=='':
        result="Unclassified"        
    return result[:-1]

def find_cons_edge(poly_refs,seg):
    for s in poly_refs:
        if (not s.isCornerhole) and (not s.isConstraint):
            continue
        if s.start_point==seg.start_point or s.start_point == seg.end_point or s.end_point ==seg.start_point or s.end_point==seg.end_point:
            return s

def get_score(free_edges_types,non_free_edges,non_free_edges_types,free_edge_seq,non_free_edges_seq,non_free_edges_types_seq,ignore_types):
    new_free_edges_types=[]
    new_free_edges_types_seq=[]
    if len(free_edge_seq)>1 and len(free_edges_types)>1 and free_edge_seq[0]=="KS_corner" and free_edges_types[0]=="KS_corner":
        new_free_edges_types.append("KS_corner")
        new_free_edges_types_seq.append("KS_corner")
    for t in free_edges_types:
        if t!="KS_corner":
            new_free_edges_types.append(t)

    for t in free_edge_seq:
        if t!="KS_corner":
            new_free_edges_types_seq.append(t)
    if len(free_edge_seq)>1 and len(free_edges_types)>1 and free_edge_seq[-1]=="KS_corner" and free_edges_types[-1]=="KS_corner":
        new_free_edges_types.append("KS_corner")
        new_free_edges_types_seq.append("KS_corner")
    if len(free_edge_seq)>1 and len(free_edges_types)>1 and free_edge_seq[0]=="KS_corner"  and free_edge_seq[-1]!="KS_corner"and free_edges_types[-1]=="KS_corner" and free_edges_types[0]!="KS_corner":
        new_free_edges_types.append("KS_corner")
        new_free_edges_types_seq=["KS_corner"]+new_free_edges_types_seq
    if len(free_edge_seq)>1 and len(free_edges_types)>1 and free_edge_seq[0]!="KS_corner"  and free_edge_seq[-1]=="KS_corner"and free_edges_types[-1]!="KS_corner" and free_edges_types[0]=="KS_corner":
        new_free_edges_types_seq.append("KS_corner")
        new_free_edges_types=["KS_corner"]+new_free_edges_types
    total=0
    for i in range(len(new_free_edges_types)):
        if new_free_edges_types[i]== new_free_edges_types_seq[i]:
            total+=1
    new_non_free_edges=[]
    new_non_free_edges_types=[]
    for i,edge in enumerate(non_free_edges):
        if ignore_types[0]=="line"and  non_free_edges_types[i]=="cornerhole" and len(edge)==1 and isinstance(edge[0].ref,DLine) :
            continue
        if ignore_types[0]=="arc"and  non_free_edges_types[i]=="cornerhole" and len(edge)==1 and isinstance(edge[0].ref,DArc) :
            continue
        new_non_free_edges.append(edge)
        new_non_free_edges_types.append(non_free_edges_types[i])
    new_non_free_edges_seq=[]
    new_non_free_edges_types_seq=[]
    for i,edge in enumerate(non_free_edges_seq):
        if ignore_types[0]=="line"and non_free_edges_types_seq[i]=="cornerhole" and len(edge)==1 and edge[0]=="line":
            continue
        if ignore_types[0]=="arc"and non_free_edges_types_seq[i]=="cornerhole" and len(edge)==1 and edge[0]=="arc":
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
def match_template(edges,detected_free_edges,template,edge_types,thickness):
    if thickness>25:
        ignore_types=["arc"]
    else:
        ignore_types=["line"]
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
    
    
    total=get_score(free_edges_types,non_free_edges,non_free_edges_types,free_edge_seq,non_free_edges_seq,non_free_edges_types_seq,ignore_types)
    r_total=get_score(r_free_edges_types,r_non_free_edges,r_non_free_edges_types,free_edge_seq,non_free_edges_seq,non_free_edges_types_seq,ignore_types)
    

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
        if ignore_types[0]=="line" and non_free_edges_types_seq[i]=="cornerhole" and len(non_free_edges_seq[i])==1 and non_free_edges_seq[i][0]=="line" and j>=len(non_free_edges):
            template_map[key]=[]
        elif ignore_types[0]=="line" and non_free_edges_types_seq[i]=="cornerhole" and len(non_free_edges_seq[i])==1 and non_free_edges_seq[i][0]=="line" and not(non_free_edges_types[j]=="cornerhole" and len(non_free_edges[j])==1 and isinstance(non_free_edges[j][0].ref,DLine)):
            template_map[key]=[]
        elif ignore_types[0]=="arc" and non_free_edges_types_seq[i]=="cornerhole" and len(non_free_edges_seq[i])==1 and non_free_edges_seq[i][0]=="arc" and j>=len(non_free_edges):
            template_map[key]=[]
        elif ignore_types[0]=="arc" and non_free_edges_types_seq[i]=="cornerhole" and len(non_free_edges_seq[i])==1 and non_free_edges_seq[i][0]=="arc" and not(non_free_edges_types[j]=="cornerhole" and len(non_free_edges[j])==1 and isinstance(non_free_edges[j][0].ref,DArc)):
            template_map[key]=[]
        else:
            if j >=len(non_free_edges):
                template_map[key]=[]
            else:
                template_map[key]=non_free_edges[j]
                j+=1
    return template_map
# 标准肘板的匹配分类函数
def poly_classifier(features,all_anno,poly_refs, texts,dimensions,conerhole_num, poly_free_edges, edges, thickness, feature_map, edge_types, standard_classification_file_path, info_json_path, keyname, is_output_json = False):
    classification_table = load_classification_table(standard_classification_file_path)

    # Step 1: 获取角隅孔数

    # Step 2: 获取自由边的轮廓
    free_edges_sequence = []
    max_free_edge_length=float("-inf")
    for seg in poly_free_edges[0]:
        max_free_edge_length=max(max_free_edge_length,seg.length())
    constraint_edges=[]
    for edge in edges:
        if edge[0].isConstraint and (not edge[0].isCornerhole):
            constraint_edges.extend(edge)
    for i, seg in enumerate(poly_free_edges[0]):
        if isinstance(seg.ref, DLine) or isinstance(seg.ref, DLwpolyline):
            if (i == 0 or i == len(poly_free_edges[0]) - 1):
                if i==0:
                    last_free_edge=poly_free_edges[0][1]
                else:
                    last_free_edge=poly_free_edges[0][-2]
                cons_edge=find_cons_edge(constraint_edges,seg)
                # print(cons_edge)
                if is_toe(seg,last_free_edge,cons_edge,max_free_edge_length):
                    free_edges_sequence.append("toe")
                elif is_ks_corner(seg,last_free_edge,cons_edge,max_free_edge_length):
                    free_edges_sequence.append("KS_corner")
                else:
                    free_edges_sequence.append("line")
            else:
                free_edges_sequence.append("line")
        elif isinstance(seg.ref, DArc):
            free_edges_sequence.append("arc")
    reversed_free_edges_sequence = free_edges_sequence[::-1]  # Reverse free edges

    # Step 3: 以自由边为分界，获取固定边和角隅孔组成的顺序轮廓
    cycle_edges = edges + edges
    al_edges = []
    start = False
    for edge in cycle_edges:
        # 如果遇到第一个非 Cornerhole 和非 Constraint 边，开始收集
        if not start and edge[0].isCornerhole == False and edge[0].isConstraint == False:
            start = True
        
        # 如果已经开始收集，且当前边是 Cornerhole 或 Constraint，加入 al_edges
        elif start and (edge[0].isCornerhole or edge[0].isConstraint):
            al_edges.append(edge)
        
        # 如果已经开始收集，且当前边不再是 Cornerhole 或 Constraint，停止收集
        elif start and edge[0].isCornerhole == False and edge[0].isConstraint == False:
            start = False

    # Step 4: 构建 edges_sequence 和 reversed_edges_sequence
    edges_sequence = []
    reversed_edges_sequence = []
    for edge in al_edges:
        if edge[0].isCornerhole:
            type = "cornerhole"
        elif edge[0].isConstraint:
            type = "constraint"
        else:
            type = None
        
        if type is not None:
            seq = []
            for seg in edge:
                if isinstance(seg.ref, DLine) or isinstance(seg.ref, DLwpolyline):
                    seq.append("line")
                elif isinstance(seg.ref, DArc):
                    seq.append("arc")
            edges_sequence.append([type, seq])
            reversed_edges_sequence.insert(0, [type, list(reversed(seq))])
    
    # 根据板厚在没有角隅孔的固定边轮廓之间添加角隅孔
    i = 0
    while i < len(edges_sequence) - 1:
        if edges_sequence[i][0] == "constraint" and edges_sequence[i + 1][0] == "constraint":
            add_seq = ["line"] if thickness <= 25 else ["arc"]
            edges_sequence.insert(i + 1, ["cornerhole", add_seq])
            i += 1  # 跳过新插入的元素
        i += 1
    
    i = 0
    while i < len(reversed_edges_sequence) - 1:
        if reversed_edges_sequence[i][0] == "constraint" and reversed_edges_sequence[i + 1][0] == "constraint":
            add_seq = ["line"] if thickness <= 25 else ["arc"]
            reversed_edges_sequence.insert(i + 1, ["cornerhole", add_seq])
            i += 1  # 跳过新插入的元素
        i += 1


    # 对肘板轮廓进行输出
    if is_output_json:
        geometry_info = {
            keyname: {
                "free_edges_sequence": free_edges_sequence,
                "non_free_edges_sequence": edges_sequence
            }
        }
        try:
            # 检查目标文件是否存在，如果存在则读取并更新
            try:
                with open(info_json_path, "r") as file:
                    existing_data = json.load(file)
            except FileNotFoundError:
                existing_data = {}

            # 更新数据
            existing_data.update(geometry_info)

            # 写入文件
            with open(info_json_path, "w") as file:
                json.dump(existing_data, file, indent=4)

        except Exception as e:
            print(f"Error writing to JSON file: {e}")


    # step5: 固定边轮廓严格匹配+角隅孔非严格匹配（直线角隅孔可能不画）
    matched_type_list = conerhole_free_classifier(classification_table, conerhole_num, free_edges_sequence, reversed_free_edges_sequence, edges_sequence, reversed_edges_sequence)
    
    # step6: 自由边的筛选
    matched_type = free_edges_sequence_classifier(classification_table, free_edges_sequence,reversed_free_edges_sequence, matched_type_list)

    # print(matched_type)
    if len(matched_type.split(","))<=1 and matched_type!="Unclassified":
        if is_new_feature_pass(matched_type, classification_table, edges, poly_free_edges, edge_types, thickness, feature_map):
            return matched_type,classification_table[matched_type]
        else:
            return "Unclassified", None
    elif matched_type=="Unclassified":
        return matched_type,None
    
    
    matched_type=","+matched_type+','

    matched_type=tidy_matched_type(matched_type)

    # step8: 考虑整体轮廓筛选
    # 去除自由边所有"KS_corner"以及轮廓中所有"["cornerhole", ["line"]]"
    free_edges_sequence = [item for item in free_edges_sequence if item != "KS_corner"]
    reversed_free_edges_sequence = [item for item in reversed_free_edges_sequence if item != "KS_corner"]
    # edges_sequence = [item for item in edges_sequence if item != ["cornerhole", ["line"]]]
    # reversed_edges_sequence = [item for item in reversed_edges_sequence if item != ["cornerhole", ["line"]]]
    # print(free_edges_sequence)
    edges_sequence.insert(0,["free", free_edges_sequence])
    reversed_edges_sequence.insert(0, ["free", reversed_free_edges_sequence])
    mixed_types = matched_type.split(',')
    matched_type = refine_poly_classifier(classification_table, mixed_types, edges_sequence, reversed_edges_sequence)


    output_template=None

    # 混淆类分类
    matched_type = tidy_matched_type(matched_type)
    
    if len(matched_type) == 0:
        return "Unclassified", None

    # 匹配最佳标准肘板分类，feature必须是该类别的子集，且特征占比最高
    # for type_name in matched_type.split(','):
    #     if type_name.strip()=="":
    #         continue
    #     free_code = classification_table[type_name]["free_code"]
    #     no_free_code = classification_table[type_name]["no_free_code"]
    #     # 方案1：特征类型统计，不归属到具体边
    #     code = []
        
    #     for c in free_code:
    #         for f in c:
    #             if f not in code:
    #                 code.append(f)
    #     for c in no_free_code:
    #         for f in c:
    #             if f not in code:
    #                 code.append(f)

    #     # 判断是否是子集且计算特征数
    #     flag = True
    #     f_num = 0

    #     for feature in features:
    #         if feature not in strict_feature:
    #             if  feature in code:
    #                 f_num += 1
    #             else:
    #                 flag = False
    #                 break
    #     for c in code:
    #         if c not in features:
    #             flag=False
    #             break

        
    #     # 如果是子集，则比较特征数
    #     if flag:
    #         if f_num > max_feature_num:
    #             max_feature_num = f_num
    #             res_matched_type = [type_name]
    #         elif f_num == max_feature_num:
    #             res_matched_type.append(type_name)
    max_feature_num = -1
    res_matched_type = "Unclassified"
    best_matched_type = None
    for type_name in matched_type.split(','):
        best_match_flag = True
        f_score = 0
        if type_name.strip()=="":
            continue
        free_code = classification_table[type_name]["free_code"]
        no_free_code = classification_table[type_name]["no_free_code"]
        template_map=match_template(edges,poly_free_edges,classification_table[type_name],edge_types,thickness=thickness)
        template_free_code = []
        free_idx = 1
        while f'free{free_idx}' in template_map:
            if len(template_map[f'free{free_idx}']) != 0:
                template_free_code.append(free_code[free_idx - 1])
            free_idx += 1
        r_template_free_code = template_free_code[::-1]
        r_no_free_code = no_free_code[::-1]

        # TODO: 添加轮廓是否对称的调用
        flag = True

        if flag:
            f_score1 = 0
            f_score2 = 0
            best_match_flag1 = True
            best_match_flag2 = True
            # 正向比对
            # 自由边特征比对
            free_idx = 1
            idx = 1
            while f'free{free_idx}' in template_map:
                if len(template_map[f'free{free_idx}'])==0:
                    free_idx += 1
                    continue
                seg=template_map[f'free{free_idx}'][0]
                f = feature_map[seg]
                c = template_free_code[idx - 1]
                idx += 1
                if eva_c_f(c, f):
                    f_score1 += 1
                else:
                    best_match_flag1 = False
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
                        f_score1 += 1
                    else:
                        best_match_flag1 = False
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
                        f_score1 += 1
                    else:
                        best_match_flag1= False
                    cornerhole_idx+=1

            # 反向比对
            # 自由边特征比对
            free_idx = 1
            idx = 1
            while f'free{free_idx}' in template_map:
                if len(template_map[f'free{free_idx}'])==0:
                    free_idx += 1
                    continue
                seg=template_map[f'free{free_idx}'][0]
                f = feature_map[seg]
                c = r_template_free_code[idx - 1]
                idx += 1
                if eva_c_f(c, f):
                    f_score2 += 1
                else:
                    best_match_flag2 = False
                free_idx += 1
            
            # 非自由边特征对比
            constarint_idx=1
            cornerhole_idx=1
            while True:
                if f'constraint{constarint_idx}' in template_map:
                    seg = template_map[f'constraint{constarint_idx}'][0]
                    f = feature_map[seg]
                    c = r_no_free_code[constarint_idx + cornerhole_idx - 2]
                    if eva_c_f(c, f):
                        f_score2 += 1
                    else:
                        best_match_flag2 = False
                    constarint_idx+=1
                else:
                    break
                if f'cornerhole{cornerhole_idx}' in template_map:
                    if len(template_map[f'cornerhole{cornerhole_idx}'])==0:
                        cornerhole_idx+=1
                        break
                    seg = template_map[f'cornerhole{cornerhole_idx}'][0]
                    f = feature_map[seg]
                    c = r_no_free_code[constarint_idx + cornerhole_idx - 2]
                    if eva_c_f(c, f):
                        f_score2 += 1
                    else:
                        best_match_flag2 = False
                    cornerhole_idx+=1
            
            best_match_flag = best_match_flag1 or best_match_flag2
            f_score = max(f_score1, f_score2)

        # 如果不对称则只正向比对
        else:
            # 自由边特征比对
            free_idx = 1
            idx = 1
            while f'free{free_idx}' in template_map:
                if len(template_map[f'free{free_idx}'])==0:
                    free_idx += 1
                    continue
                seg=template_map[f'free{free_idx}'][0]
                f = feature_map[seg]
                c = template_free_code[free_idx - 1]
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
        # print(type_name,best_match_flag,f_score)
        # 如果完全匹配成功，直接作为最终结果；否则则进行分数比较
        if best_match_flag:
            best_matched_type = type_name if best_matched_type is None else f'{best_matched_type},{type_name}'
        elif f_score > max_feature_num:
            max_feature_num = f_score
            res_matched_type = type_name
        elif f_score == max_feature_num:
            res_matched_type = f'{res_matched_type},{type_name}'


    # 如果成功匹配到标准肘板类别，则返回该类别；否则，则仅返回其中一类别模板
    if best_matched_type is not None:
        res_matched_type = best_matched_type
    
    res_matched_type_="Unclassified"
    if len(res_matched_type.split(","))>1:
        max_feature_num = -1
        for type_name in res_matched_type.split(','):
            f_score = 0
            if type_name.strip()=="":
                continue
            free_code = classification_table[type_name]["free_code"]
            no_free_code = classification_table[type_name]["no_free_code"]
            template_map=match_template(edges,poly_free_edges,classification_table[type_name],edge_types,thickness=thickness)

            # 自由边特征比对
            free_idx = 1
            while f'free{free_idx}' in template_map:
                if len(template_map[f'free{free_idx}'])==0:
                    free_idx += 1
                    continue
                seg=template_map[f'free{free_idx}'][0]
                f = feature_map[seg]
                c = free_code[free_idx - 1]
                if eva_c_f(f, c):
                    f_score += 1
                free_idx += 1
            
            # 非自由边特征对比
            constarint_idx=1
            cornerhole_idx=1
            while True:
                if f'constraint{constarint_idx}' in template_map:
                    seg = template_map[f'constraint{constarint_idx}'][0]
                    f = feature_map[seg]
                    c = no_free_code[constarint_idx + cornerhole_idx - 2]
                    if eva_c_f(f, c):
                        f_score += 1

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
                    if eva_c_f(f, c):
                        f_score += 1

                    cornerhole_idx+=1
            # print(type_name,best_match_flag,f_score)
            # 如果完全匹配成功，直接作为最终结果；否则则进行分数比较
            if f_score > max_feature_num:
                max_feature_num = f_score
                res_matched_type_ = type_name
            elif f_score == max_feature_num:
                res_matched_type_ = f'{res_matched_type_},{type_name}'

    else:
        res_matched_type_=res_matched_type

    if res_matched_type_ == "Unclassified":
        matched_type = res_matched_type_
        output_template = None
    else:
        matched_type = res_matched_type_
        output_template = classification_table[res_matched_type_.split(',')[0]]

    # 添加new特征的判断
    matched_type_list = matched_type.split(',')
    matched_type = None
    for type_name in matched_type_list:
        if is_new_feature_pass(type_name, classification_table, edges, poly_free_edges, edge_types, thickness, feature_map):
            matched_type = type_name if matched_type is None else f'{matched_type},{type_name}'
    return matched_type, output_template

def poly_classifier_ustd():
    return "Unstandard"
def eva_c_f(codes, features, p_feature = ["no_tangent", "is_para", "is_ver", "is_ontoe"]):
    # codes中的所有特征都要包含在features中
    for c in codes:
        if c not in features:
            return False
    # features中的所有标注特征都要包含在codes中
    # for f in features:
    #     if f not in p_feature:
    #         if f not in codes:
    #             return False
    return True

def eva_c_f_new(codes, features):
    strict_features = ["new", "is_para"]
    
    # codes中的所有特征都要包含在features中
    for c in codes:
        if c in strict_features and c not in features:
            return False
    # features中的所有标注特征都要包含在codes中
    for f in features:
        if f in strict_features and f not in codes:
            return False
    return True



# 整体轮廓过滤    
def refine_poly_classifier(classification_table, mixed_types, edges_sequence, reversed_edges_sequence):
    matched_type = None
    max_similarity = -1  # 初始化最大相似度为 -1

    for type in mixed_types:
        temp_free_edge_seq = classification_table[type]["free_edges"]
        temp_edge_seq = classification_table[type]["non_free_edges"]
        tmp_edge_seq = []
        for edge in temp_edge_seq:
            if isinstance(edge["edges"][0], list):
                e = edge["edges"][0]
            else:
                e = edge["edges"]
            tmp_edge_seq.append([edge["type"], e])
        temp_free_edge_seq = [item for item in temp_free_edge_seq if item != "KS_corner"]

        tmp_edge_seq.insert(0, ["free", temp_free_edge_seq])
        # 计算与 edges_sequence 的相似度
        similarity = calculate_similarity(tmp_edge_seq, edges_sequence)
        # 计算与 reversed_edges_sequence 的相似度
        reversed_similarity = calculate_similarity(tmp_edge_seq, reversed_edges_sequence)

        # 取两者中的最大值
        current_max_similarity = max(similarity, reversed_similarity)

        # 如果当前相似度大于最大相似度，则更新 matched_type
        if current_max_similarity > max_similarity:
            max_similarity = current_max_similarity
            matched_type = type
        elif current_max_similarity == max_similarity:
            matched_type = type if matched_type is None else f"{matched_type},{type}"

    return matched_type

def calculate_similarity(seq1, seq2):
    """
    计算两个序列的相似度。
    这里使用简单的匹配比例作为相似度度量。
    """
    if not seq1 or not seq2:
        return 0

    match_count = 0
    min_length = min(len(seq1), len(seq2))

    for i in range(min_length):
        if seq1[i] == seq2[i]:
            match_count += 1

    return match_count / min_length

def is_new_feature_pass(matched_type, classification_table,edges, poly_free_edges, edge_types, thickness, feature_map):
    type_name = matched_type.split(',')[0]
    is_match_flag = True
    f_score = 0
    free_code = classification_table[type_name]["free_code"]
    no_free_code = classification_table[type_name]["no_free_code"]
    template_map=match_template(edges,poly_free_edges,classification_table[type_name],edge_types,thickness=thickness)

    template_free_code = []
    free_idx = 1
    while f'free{free_idx}' in template_map:
        if len(template_map[f'free{free_idx}']) != 0:
            template_free_code.append(free_code[free_idx - 1])
            free_idx += 1
    r_template_free_code = template_free_code[::-1]

    # 自由边特征比对（轮廓对称会正反进行）
    # TODO: 添加轮廓是否对称的调用
    flag = True

    if flag:
        f_score1 = 0
        f_score2 = 0
        is_match_flag1 = True
        is_match_flag2 = True
        # 正向比对
        # 自由边特征比对
        free_idx = 1
        idx = 1
        while f'free{free_idx}' in template_map:
            if len(template_map[f'free{free_idx}'])==0:
                free_idx += 1
                continue
            seg=template_map[f'free{free_idx}'][0]
            f = feature_map[seg]
            c = template_free_code[idx - 1]
            idx += 1
            if eva_c_f(c, f):
                f_score1 += 1
            else:
                is_match_flag1 = False
            free_idx += 1
        
        # 反向比对
        # 自由边特征比对
        free_idx = 1
        idx = 1
        while f'free{free_idx}' in template_map:
            if len(template_map[f'free{free_idx}'])==0:
                free_idx += 1
                continue
            seg=template_map[f'free{free_idx}'][0]
            f = feature_map[seg]
            c = r_template_free_code[idx - 1]
            idx += 1
            if eva_c_f(c, f):
                f_score2 += 1
            else:
                is_match_flag2 = False
            free_idx += 1
        
        
        is_match_flag = is_match_flag1 or is_match_flag2
        f_score = max(f_score1, f_score2)

    # 如果不对称则只正向比对
    else:
        # 自由边特征比对
        free_idx = 1
        idx = 1
        while f'free{free_idx}' in template_map:
            if len(template_map[f'free{free_idx}'])==0:
                free_idx += 1
                continue
            seg=template_map[f'free{free_idx}'][0]
            f = feature_map[seg]
            c = template_free_code[free_idx - 1]
            if eva_c_f(c, f):
                f_score += 1
            else:
                is_match_flag = False
            free_idx += 1

    
    # 非自由边特征对比
    constarint_idx=1
    cornerhole_idx=1
    while True:
        if f'constraint{constarint_idx}' in template_map:
            seg = template_map[f'constraint{constarint_idx}'][0]
            f = feature_map[seg]
            c = no_free_code[constarint_idx + cornerhole_idx - 2]
            if eva_c_f_new(c, f):
                f_score += 1
            else:
                is_match_flag = False
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
            if eva_c_f_new(c, f):
                f_score += 1
            else:
                is_match_flag = False
            cornerhole_idx+=1
    return is_match_flag

# 非标肘板判断逻辑，只进行轮廓匹配
def poly_classifier_ustd(poly_free_edges, edges, unstandard_classification_file_path, edge_types, thickness, feature_map):
    classification_table = load_classification_table(unstandard_classification_file_path)

    # 获取自由边轮廓
    free_edges_sequence = []
    max_free_edge_length=float("-inf")
    for seg in poly_free_edges[0]:
        max_free_edge_length=max(max_free_edge_length,seg.length())
    constraint_edges=[]
    for edge in edges:
        if edge[0].isConstraint and (not edge[0].isCornerhole):
            constraint_edges.extend(edge)
    for i, seg in enumerate(poly_free_edges[0]):
        if isinstance(seg.ref, DLine) or isinstance(seg.ref, DLwpolyline):
            if (i == 0 or i == len(poly_free_edges[0]) - 1):
                if i==0:
                    last_free_edge=poly_free_edges[0][1]
                else:
                    last_free_edge=poly_free_edges[0][-2]
                cons_edge=find_cons_edge(constraint_edges,seg)
                # print(cons_edge)
                if is_toe(seg,last_free_edge,cons_edge,max_free_edge_length):
                    free_edges_sequence.append("toe")
                elif is_ks_corner(seg,last_free_edge,cons_edge,max_free_edge_length):
                    free_edges_sequence.append("KS_corner")
                else:
                    free_edges_sequence.append("line")
            else:
                free_edges_sequence.append("line")
        elif isinstance(seg.ref, DArc):
            free_edges_sequence.append("arc")
    reversed_free_edges_sequence = free_edges_sequence[::-1]  # Reverse free edges

    # 以自由边为分界，获取固定边和角隅孔组成的顺序轮廓
    cycle_edges = edges + edges
    al_edges = []
    start = False
    for edge in cycle_edges:
        # 如果遇到第一个非 Cornerhole 和非 Constraint 边，开始收集
        if not start and edge[0].isCornerhole == False and edge[0].isConstraint == False:
            start = True
        
        # 如果已经开始收集，且当前边是 Cornerhole 或 Constraint，加入 al_edges
        elif start and (edge[0].isCornerhole or edge[0].isConstraint):
            al_edges.append(edge)
        
        # 如果已经开始收集，且当前边不再是 Cornerhole 或 Constraint，停止收集
        elif start and edge[0].isCornerhole == False and edge[0].isConstraint == False:
            start = False

    # 构建 edges_sequence 和 reversed_edges_sequence
    edges_sequence = []
    reversed_edges_sequence = []
    for edge in al_edges:
        if edge[0].isCornerhole:
            type = "cornerhole"
        elif edge[0].isConstraint:
            type = "constraint"
        else:
            type = None
        
        if type is not None:
            seq = []
            for seg in edge:
                if isinstance(seg.ref, DLine) or isinstance(seg.ref, DLwpolyline):
                    seq.append("line")
                elif isinstance(seg.ref, DArc):
                    seq.append("arc")
            edges_sequence.append([type, seq])
            reversed_edges_sequence.insert(0, [type, list(reversed(seq))])
    
    # step5: 固定边轮廓严格匹配+角隅孔非严格匹配（直线角隅孔可能不画）
    matched_type_list = conerhole_free_classifier(classification_table, 0, free_edges_sequence, reversed_free_edges_sequence, edges_sequence, reversed_edges_sequence,is_ustd=True)
    # print(matched_type_list)
    # step6: 自由边的筛选
    matched_type = free_edges_sequence_classifier(classification_table, free_edges_sequence,reversed_free_edges_sequence, matched_type_list)

    if matched_type=="Unclassified":
        return "Unclassified"
    
    matched_type=","+matched_type+','

    matched_type=tidy_matched_type(matched_type)
    # print(matched_type)
    free_edges_sequence = [item for item in free_edges_sequence if item != "KS_corner"]
    reversed_free_edges_sequence = [item for item in reversed_free_edges_sequence if item != "KS_corner"]
    edges_sequence.insert(0,["free", free_edges_sequence])
    reversed_edges_sequence.insert(0, ["free", reversed_free_edges_sequence])
    mixed_types = matched_type.split(',')
    matched_type = refine_poly_classifier(classification_table, mixed_types, edges_sequence, reversed_edges_sequence)
    # print(matched_type)
    # 混淆类分类
    matched_type = tidy_matched_type(matched_type)
    
    if matched_type == "Unclassified":
        return "Unclassified"

    # # 非标肘板特征匹配
    # max_feature_num = -1
    # res_matched_type = "Unclassified"
    # best_matched_type = None
    # for type_name in matched_type.split(','):
    #     best_match_flag = True
    #     f_score = 0
    #     if type_name.strip()=="":
    #         continue
    #     free_code = classification_table[type_name]["free_code"]
    #     no_free_code = classification_table[type_name]["no_free_code"]
    #     template_map=match_template(edges,poly_free_edges,classification_table[type_name],edge_types,thickness=thickness)

    #     # 自由边特征比对
    #     free_idx = 1
    #     while f'free{free_idx}' in template_map:
    #         if len(template_map[f'free{free_idx}'])==0:
    #             free_idx += 1
    #             continue
    #         seg=template_map[f'free{free_idx}'][0]
    #         f = feature_map[seg]
    #         c = free_code[free_idx - 1]
    #         if eva_c_f(c, f):
    #             f_score += 1
    #         else:
    #             best_match_flag = False
    #         free_idx += 1
        
    #     # 非自由边特征对比
    #     constarint_idx=1
    #     cornerhole_idx=1
    #     while True:
    #         if f'constraint{constarint_idx}' in template_map:
    #             seg = template_map[f'constraint{constarint_idx}'][0]
    #             f = feature_map[seg]
    #             c = no_free_code[constarint_idx + cornerhole_idx - 2]
    #             if eva_c_f(c, f):
    #                 f_score += 1
    #             else:
    #                 best_match_flag = False
    #             constarint_idx+=1
    #         else:
    #             break
    #         if f'cornerhole{cornerhole_idx}' in template_map:
    #             if len(template_map[f'cornerhole{cornerhole_idx}'])==0:
    #                 cornerhole_idx+=1
    #                 break
    #             seg = template_map[f'cornerhole{cornerhole_idx}'][0]
    #             f = feature_map[seg]
    #             c = no_free_code[constarint_idx + cornerhole_idx - 2]
    #             if eva_c_f(c, f):
    #                 f_score += 1
    #             else:
    #                 best_match_flag = False
    #             cornerhole_idx+=1
    #     # print(type_name,best_match_flag,f_score)
    #     # 如果完全匹配成功，直接作为最终结果；否则则进行分数比较
    #     if best_match_flag:
    #         best_matched_type = type_name if best_matched_type is None else f'{best_matched_type},{type_name}'
    #     elif f_score > max_feature_num:
    #         max_feature_num = f_score
    #         res_matched_type = type_name
    #     elif f_score == max_feature_num:
    #         res_matched_type = f'{res_matched_type},{type_name}'


    # # 如果成功匹配到标准肘板类别，则返回该类别；否则，则仅返回其中一类别模板
    # if best_matched_type is not None:
    #     res_matched_type = best_matched_type
    
    # res_matched_type_="Unclassified"
    # if len(res_matched_type.split(","))>1:
    #     max_feature_num = -1
    #     for type_name in res_matched_type.split(','):
    #         f_score = 0
    #         if type_name.strip()=="":
    #             continue
    #         free_code = classification_table[type_name]["free_code"]
    #         no_free_code = classification_table[type_name]["no_free_code"]
    #         template_map=match_template(edges,poly_free_edges,classification_table[type_name],edge_types,thickness=thickness)

    #         # 自由边特征比对
    #         free_idx = 1
    #         while f'free{free_idx}' in template_map:
    #             if len(template_map[f'free{free_idx}'])==0:
    #                 free_idx += 1
    #                 continue
    #             seg=template_map[f'free{free_idx}'][0]
    #             f = feature_map[seg]
    #             c = free_code[free_idx - 1]
    #             if eva_c_f(f, c):
    #                 f_score += 1
    #             free_idx += 1
            
    #         # 非自由边特征对比
    #         constarint_idx=1
    #         cornerhole_idx=1
    #         while True:
    #             if f'constraint{constarint_idx}' in template_map:
    #                 seg = template_map[f'constraint{constarint_idx}'][0]
    #                 f = feature_map[seg]
    #                 c = no_free_code[constarint_idx + cornerhole_idx - 2]
    #                 if eva_c_f(f, c):
    #                     f_score += 1

    #                 constarint_idx+=1
    #             else:
    #                 break
    #             if f'cornerhole{cornerhole_idx}' in template_map:
    #                 if len(template_map[f'cornerhole{cornerhole_idx}'])==0:
    #                     cornerhole_idx+=1
    #                     break
    #                 seg = template_map[f'cornerhole{cornerhole_idx}'][0]
    #                 f = feature_map[seg]
    #                 c = no_free_code[constarint_idx + cornerhole_idx - 2]
    #                 if eva_c_f(f, c):
    #                     f_score += 1

    #                 cornerhole_idx+=1
    #         # print(type_name,best_match_flag,f_score)
    #         # 如果完全匹配成功，直接作为最终结果；否则则进行分数比较
    #         if f_score > max_feature_num:
    #             max_feature_num = f_score
    #             res_matched_type_ = type_name
    #         elif f_score == max_feature_num:
    #             res_matched_type_ = f'{res_matched_type_},{type_name}'

    # else:
    #     res_matched_type_=res_matched_type

    # if res_matched_type_ == "Unclassified":
    #     matched_type = res_matched_type_
    #     output_template = None
    # else:
    #     matched_type = res_matched_type_
    #     output_template = classification_table[res_matched_type_.split(',')[0]]

    # # 添加new特征的判断
    # matched_type_list = matched_type.split(',')
    # matched_type = None
    # for type_name in matched_type_list:
    #     if is_new_feature_pass(type_name, classification_table, edges, poly_free_edges, edge_types, thickness, feature_map):
    #         matched_type = type_name if matched_type is None else f'{matched_type},{type_name}'
    return matched_type