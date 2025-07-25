import ezdxf
import ezdxf.bbox
import ezdxf.disassemble
import ezdxf.entities 
import json
import os
import math
import numpy as np
import math
from ezdxf.xclip import XClip, ClippingPath

# Entity 
# (g)type
# (g)color
# (g) bound : {x1,y1,x2,y2} # left-bottom # right-top
# (g) layerName

# (text)insert
# (text)content
# (text)height

# (line)start
# (line)end
# (line)linetype
PI_CIRCLE = 2*3.1415

def approximate_equal(a:float ,b :float,ellipse = 1e-6):
    if (abs(a-b) < ellipse):
        return True
    return False

def vector_to_angle(vector):
    x, y = vector
    angle = math.atan2(y, x)    
    if angle < 0:
        angle += PI_CIRCLE
    return angle

def getColor(entity):
    try:
        directColor = entity.dxf.color
        if directColor == 0:
            layer = entity.doc.layers.get(entity.dxf.layer)
            return layer.color
        elif directColor == 256:
            fa = entity.source_block_reference
            if (fa is None):
                layer = entity.doc.layers.get(entity.dxf.layer)
                return layer.color
            else:
                raise NotImplementedError('随块，但是块的颜色是fa.color吗？')
                return fa.color

        else:
            return directColor
    except NotImplementedError as e:
        return 256

def getLineWeight(doc,entity):
    try:
        dircetLw = entity.dxf.lineweight
        if dircetLw == -1:
            layer = doc.layers.get(entity.dxf.layer)
            return getLineWeight(doc,layer)
        elif dircetLw == -2:
            raise NotImplementedError('随块尚未实现')
        elif dircetLw == -3:
            return doc.header.get('$LWDEFAULT',0) # dxf未说明则代表是0
        else:
            return dircetLw
    except NotImplementedError as e:
        print(e)
        exit()


def getLineType(doc, entity):
    try:
        linetype = entity.dxf.linetype

        if linetype == "BYLAYER":
            layer = doc.layers.get(entity.dxf.layer)
            return layer.dxf.linetype
        
        elif linetype == "ByBlock":

            blockname = entity.dxf.owner
            block = doc.blocks.get(blockname)

            print("Byblock", entity.dxf.handle,blockname, block)

            if block:
                linetype = "Byblock " + block.dxf.linetype
            else:
                linetype = "Byblock"

            return linetype
        else:
            return linetype
    except NotImplemented as e:
        print(e)
        exit()


def convertEntity(type,entity):
    bb= ezdxf.bbox.extents([entity])
    return {
        'type' : type,
        'color' : getColor(entity),
        'layerName' : entity.dxf.layer,
        'handle':entity.dxf.handle,
        'bound' :{
            'x1' : bb.extmin[0],
            'y1' : bb.extmin[1],
            'x2' : bb.extmax[0],
            'y2' : bb.extmax[1] ,
        },
    }

def convertCircle(doc, entity : ezdxf.entities.Circle):
    mid = convertEntity('circle',entity)
    mid['center'] = [entity.dxf.center[0],entity.dxf.center[1]]
    mid['radius'] = entity.dxf.radius
    mid['linetype'] = getLineType(doc, entity)
    return mid
  
def convertText(entity: ezdxf.entities.Text):
    mid = convertEntity('text',entity)
    mid['insert'] = [entity.dxf.insert[0],entity.dxf.insert[1]]
    mid['content'] = entity.dxf.text
    mid['height'] = entity.dxf.height
    return mid

def convertLine(doc, entity:ezdxf.entities.LineEdge):
    mid = convertEntity('line',entity)
    mid['start'] = [entity.dxf.start[0],entity.dxf.start[1]]
    mid['end'] = [entity.dxf.end[0],entity.dxf.end[1]]
    mid['linetype'] = getLineType(doc, entity)
    return mid

def convertArc(doc, entity:ezdxf.entities.ArcEdge):
    mid = convertEntity('arc',entity)
    mid['linetype'] = getLineType(doc, entity)
    mid['center'] = [entity.dxf.center[0],entity.dxf.center[1]]
    mid['radius'] = entity.dxf.radius
    mid['startAngle'] = entity.dxf.start_angle
    mid['endAngle'] = entity.dxf.end_angle
    # mid['isClosed'] = entity.is_closed

    return mid

def convertPolyLine(doc, entity:ezdxf.entities.Polyline):
    mid = convertEntity('polyline',entity)
    mid['isClosed'] = entity.is_closed
    mid['linetype'] = getLineType(doc, entity)
    # mid['vertices'] = [[x.dxf.location[0],x.dxf.location[1]] for x in entity.vertices]
    
    mid['vertices'] = []
    
    for i in range(len(entity.vertices) - 1):
        start_point = [entity.vertices[i].dxf.location[0], entity.vertices[i].dxf.location[1]]
        end_point = [entity.vertices[i + 1].dxf.location[0], entity.vertices[i + 1].dxf.location[1]]
        
        x1, y1 = start_point
        x2, y2 = end_point
        mid['vertices'].append([x1, y1, x2, y2])
    
    if entity.is_closed:
        x1 = entity.vertices[-1].dxf.location[0]
        y1 = entity.vertices[-1].dxf.location[1]
        
        x2 = entity.vertices[0].dxf.location[0]
        y2 = entity.vertices[0].dxf.location[1]
        mid["vertices"].append([x1, y1, x2, y2])
        
    return mid

def convertSpline(doc, entity:ezdxf.entities.Spline):
    mid = convertEntity('spline',entity)
    mid['linetype'] = getLineType(doc, entity)
    # mid['vertices'] = [x.tolist()[0:2] for x in entity.control_points]
    
    mid["vertices"] = []
    
    for i in range(len(entity.control_points) - 1):
        start_point = [entity.control_points[i][0], entity.control_points[i][1]]
        end_point =  [entity.control_points[i + 1][0], entity.control_points[i + 1][1]]
        
        x1, y1 = start_point
        x2, y2 = end_point
        mid['vertices'].append([x1, y1, x2, y2])
    return mid

def convertLWPolyline(doc, entity:ezdxf.entities.Spline):
    mid = convertEntity('lwpolyline',entity)
    mid['isClosed'] = entity.is_closed
    mid['linetype'] = getLineType(doc, entity)
    mid['hasArc'] = entity.has_arc

    # mid['vertices'] = [[x[0].item(),x[1].item()] for x in entity.vertices()]
    # if not entity.has_arc:
    #     mid['vertices'] = [[x[0].item(),x[1].item()] for x in entity.vertices()]
    #     mid['verticesType']  = [["line"] for x in entity.vertices()]

    mid['vertices'] = []
    mid['verticesType'] = []
    mid["verticesWidth"] = []
    
    for p in entity.get_points():
        x, y, s_w, e_w, bulge = p
        mid["verticesWidth"].append([s_w, e_w])
    
    # print(entity.dxf.handle, entity.dxf.count)
    for i in range(entity.dxf.count  - 1):
        start_point = entity.__getitem__(i)
        end_point = entity.__getitem__(i + 1)
        # print(start_point[-1])

        if start_point[-1] != 0.:
            arc = ezdxf.math.bulge_to_arc(start_point[:2], end_point[:2], start_point[-1])
            center, start_angle, end_angle, radius  = arc 
            x, y = center
            mid["vertices"].append([x, y, start_angle, end_angle, radius])
            mid['verticesType'].append("arc")
        else:
            x1, y1 = start_point[:2]
            x2, y2 = end_point[:2]
            mid["vertices"].append([x1, y1, x2, y2])
            mid['verticesType'].append("line")
        
    if entity.is_closed:
        start_point = entity.__getitem__(entity.dxf.count - 1)
        end_point = entity.__getitem__(0)

        if start_point[-1] != 0.:
                arc = ezdxf.math.bulge_to_arc(start_point[:2], end_point[:2], start_point[-1])
                center, start_angle, end_angle, radius  = arc 
                x, y = center
                mid["vertices"].append([x, y, start_angle, end_angle, radius])
                mid['verticesType'].append("arc")
        else:
            x1, y1 = start_point[:2]
            x2, y2 = end_point[:2]

            mid["vertices"].append([x1, y1, x2, y2])
            mid['verticesType'].append("line")


    # for i in range(num_vertices):
    #     # print(vertices)
    #     print(vertices[i])
    

    return mid

def convertDimension(entity:ezdxf.entities.Dimension):
    mid = convertEntity('dimension',entity)
    mid['measurement'] = entity.dxf.actual_measurement
    mid['text'] = entity.dxf.text
    mid['dimtype'] = entity.dxf.dimtype
    mid['textpos'] = [entity.dxf.text_midpoint[0],entity.dxf.text_midpoint[1]]
    mid['defpoint1'] = [entity.dxf.defpoint[0],entity.dxf.defpoint[1]]
    mid['defpoint2'] = [entity.dxf.defpoint2[0],entity.dxf.defpoint2[1]]
    mid['defpoint3'] = [entity.dxf.defpoint3[0],entity.dxf.defpoint3[1]]
    mid['defpoint4'] = [entity.dxf.defpoint4[0],entity.dxf.defpoint4[1]]
    mid['defpoint5'] = [entity.dxf.defpoint5[0],entity.dxf.defpoint5[1]]
    

    return mid

def convertSolid(entity:ezdxf.entities.Solid):
    mid = convertEntity('solid',entity)
    mid['vtx0'] = [entity.dxf.vtx0[0],entity.dxf.vtx0[1]]
    mid['vtx1'] = [entity.dxf.vtx1[0],entity.dxf.vtx1[1]]
    mid['vtx2'] = [entity.dxf.vtx2[0],entity.dxf.vtx2[1]]
    mid['vtx3'] = [entity.dxf.vtx3[0],entity.dxf.vtx3[1]]
    return mid

def convertAttdef(entity:ezdxf.entities.AttDef):
    mid = convertEntity('attdef',entity)
    mid['insert'] = [entity.dxf.insert[0],entity.dxf.insert[1]]
    mid['content'] = entity.dxf.text
    mid['height'] = entity.dxf.height
    mid['text'] = entity.dxf.text
    mid['rotation'] = entity.dxf.rotation
    return mid

def convertHatch(entity:ezdxf.entities.Hatch):
    
    S_THRESHOLD = 295509
    mid = convertEntity('hatch', entity)
    
    x1 = mid["bound"]["x1"]
    x2 = mid["bound"]["x2"]
    y1 = mid["bound"]["y1"]
    y2 = mid["bound"]["y2"]
    
    s = math.fabs((x2 - x1) * (y2 - y1))


    edges = entity.paths.paths[0]
    mid["edges"]=[]


    if isinstance(edges, ezdxf.entities.boundary_paths.EdgePath):
        edges = edges.edges
        for edge in edges:
            
            if isinstance(edge, ezdxf.entities.boundary_paths.LineEdge):
                res = {}
                res["edge_type"] = "line"
                res["coords"] = [
                    edge.start[0], 
                    edge.start[1], 
                    edge.end[0], 
                    edge.end[1]
                ]
                mid['edges'].append(res)
                

            elif isinstance(edge, ezdxf.entities.boundary_paths.EllipseEdge):

                res = {}
                res["edge_type"] = "ellipse"
                
                res['center'] = [edge.center[0],edge.center[1]]
                res['major_axis'] =[edge.major_axis[0],edge.major_axis[1]]
                res['ratio'] = edge.ratio

                res['start_param'] = edge.start_param
                res['end_param'] = edge.end_param
                # 下面自行计算起始角度和终止角度 start_theta end_theat
                start_theta = edge.start_param
                end_theta = edge.end_param

                # 如果主轴方向指向负半轴 角度 += pi 
                major_axis_rotation = vector_to_angle(res['major_axis'])
                start_theta = (start_theta + major_axis_rotation) % (PI_CIRCLE)
                end_theta = (end_theta + major_axis_rotation) % (PI_CIRCLE)
                if (start_theta > end_theta ):
                    if approximate_equal(end_theta,0,1e-2):
                        end_theta = 6.282
                    if approximate_equal(start_theta,6.282,1e-2):
                        start_theta =0
                res['calc_start_theta'] = start_theta
                res['calc_end_theta'] = end_theta

                mid["edges"].append(res)

            elif isinstance(edge, ezdxf.entities.boundary_paths.ArcEdge):
                res = {}
                res["edge_type"] = "arc"

                res["center"] = [edge.center[0], edge.center[1]]
                res['radius'] = edge.radius
                res['start_angle'] = edge.start_angle
                res['end_angle'] = edge.end_angle

                mid['edges'].append(res)
            

        
    elif isinstance(edges, ezdxf.entities.boundary_paths.PolylinePath):

        res = {}
        res['edge_type'] = "polyline"
        res["coords"] = []
        for v in edges.vertices:
            res["coords"].append([v[0], v[1]])

        mid['edges'].append(res)
    # print(type())


    if s < S_THRESHOLD:
        return None
    else:
        return mid

def convertInsert(entity:ezdxf.entities.Insert, block_list):
    if entity.dxf.name == "":
        return None, block_list
    mid = convertEntity('insert',entity)
    mid['blockName'] = entity.dxf.name
    mid['insert'] = [entity.dxf.insert[0],entity.dxf.insert[1]]
    mid['scales'] = [entity.dxf.xscale, entity.dxf.yscale]
    mid['rotation'] = entity.dxf.rotation
    
    

    clip = XClip(entity)

    if clip.has_clipping_path and clip.is_clipping_enabled:

        # print(entity.dxf.name, entity.dxf.handle, clip.get_wcs_clipping_path().vertices)

        coords = np.array(clip.get_wcs_clipping_path().vertices)
        # print(coords)

        x1 = np.min(coords[:, 0])
        x2 = np.max(coords[:, 0])
        
        x1 = max(x1, mid["bound"]["x1"])
        x2 = min(x2, mid["bound"]["x2"])

        y1 = np.min(coords[:, 1])
        y2 = np.max(coords[:, 1])

        y1 = max(y1, mid["bound"]["y1"])
        y2 = min(y2, mid["bound"]["y2"])

        mid["bound"]={
            'x1' : x1,
            'y1' : y1,
            'x2' : x2,
            'y2' : y2
        }
        # print(x1, x2, y1, y2)



    attrib_list = []
    for attrib in entity.attribs:
        print(entity.dxf.name, attrib.dxf.handle, attrib.dxf.tag, attrib.dxf.text)
        attrib_list.append({
            "attribHandle":attrib.dxf.handle, 
            "attribTag":attrib.dxf.tag,
            "attribText":attrib.dxf.text
        })
    

    # print(entity.dxf.handle, mid["bound"])



    mid["attribs"] = attrib_list

    for k in mid["bound"]:
        if np.isinf(mid["bound"][k]):
            mid = None
            break

    if entity.dxf.name not in block_list:
        block_list.append(entity.dxf.name)


    # print(mid)
    return mid, block_list

def convertMText(entity:ezdxf.entities.MText):
    if entity.dxf.width != 0:
        entity.dxf.width = 0
    mid = convertEntity('mtext',entity)
    mid['insert'] = [entity.dxf.insert[0],entity.dxf.insert[1]]
    mid['width'] = entity.dxf.width
    mid['content'] = entity.dxf.text
    return mid

def convertEllipse(entity:ezdxf.entities.Ellipse):
    mid = convertEntity('ellipse',entity)
    mid['center'] = [entity.dxf.center[0],entity.dxf.center[1]]
    mid['major_axis'] =[entity.dxf.major_axis[0],entity.dxf.major_axis[1]]
    mid['ratio'] = entity.dxf.ratio
    mid['extrusion'] = [entity.dxf.extrusion[0],entity.dxf.extrusion[1],entity.dxf.extrusion[2]]
    mid['start_param'] = entity.dxf.start_param
    mid['end_param'] = entity.dxf.end_param
    # 下面自行计算起始角度和终止角度 start_theta end_theat
    start_theta = entity.dxf.start_param
    end_theta = entity.dxf.end_param
    # 如果是翻转了的椭圆，角度 = 2pi - theta
    if (approximate_equal(mid['extrusion'][2],-1)):
        start_theta,end_theta = PI_CIRCLE -end_theta,PI_CIRCLE - start_theta 
    # 如果主轴方向指向负半轴 角度 += pi 
    major_axis_rotation = vector_to_angle(mid['major_axis'])
    start_theta = (start_theta + major_axis_rotation) % (PI_CIRCLE)
    end_theta = (end_theta + major_axis_rotation) % (PI_CIRCLE)
    if (start_theta > end_theta ):
        if approximate_equal(end_theta,0,1e-2):
            end_theta = 6.282
        if approximate_equal(start_theta,6.282,1e-2):
            start_theta =0
    mid['calc_start_theta'] = start_theta
    mid['calc_end_theta'] = end_theta

    return mid

def convertRegion(entity:ezdxf.entities.Region):
    mid = convertEntity('region',entity)
    print('In convert : 遇到Region,包围盒会变成无穷')
    return mid


def convertLeader(entity:ezdxf.entities.Leader):
    mid = convertEntity('leader',entity)
    
    mid["vertices"] = [[x[0], x[1]] for x in entity.vertices]
    return mid


def convertBlocks(doc, block_list):


    blocks = doc.blocks
    block_info = {}

    for block in blocks:    
        block_name = block.name
        res = []
        for e in block:
            if (isEntityHidden(e)):
                # print(f'块内检测到隐藏元素 {e.dxf.handle} {e.dxftype()}')
                continue
            j = None
            if (e.dxftype() == 'LINE'):
                j = convertLine(doc, e)
            elif e.dxftype() == 'CIRCLE':
                j = convertCircle(doc, e)
            elif e.dxftype() == 'TEXT':
                # 空文字直接删了
                if (e.dxf.text != ''):
                    j = convertText(e)
            elif e.dxftype() == 'ARC':
                j = convertArc(doc, e)
            elif e.dxftype() == 'POLYLINE':
                j = convertPolyLine(doc, e)
            elif e.dxftype() == 'SPLINE':
                j = convertSpline(doc, e)
            elif e.dxftype() == 'LWPOLYLINE':
                j = convertLWPolyline(doc, e)
            elif e.dxftype() == 'DIMENSION':
                j = convertDimension(e)
            elif e.dxftype() == 'ATTDEF':
                j = convertAttdef(e)
            elif e.dxftype() == 'MTEXT':
                j = convertMText(e)
            elif e.dxftype() == 'ELLIPSE':
                j = convertEllipse(e)
            elif e.dxftype() == 'REGION':
                print('region 没什么好获取的信息')
                # j = convertRegion(e)
            elif e.dxftype() == 'HATCH':
                j = convertHatch(e)             
            elif e.dxftype() == 'OLE2FRAME':
                continue                #未实现
            elif e.dxftype() == 'LEADER':
                j = convertLeader(e)          
            elif e.dxftype() == 'INSERT':
                j, _ = convertInsert(e, [])


            elif e.dxftype() == 'SOLID':
                # j = convertSolid(e)
                continue               #存在问题
            else:
                # raise NotImplementedError(f'遇到了未见过的类型 {e.dxftype()}')
                # print(f'Block中遇到了未见过的类型 {e.dxftype()} ')
                continue
            if(j is not None):
                res.append(j)
        
        block_info[block_name] = res

    return block_info

def isEntityHidden(entity):
    """
    判断给定的实体是否是隐藏的。

    参数:
    entity (DXFEntity): 要检查的实体对象。

    返回:
    bool: 如果实体隐藏返回True，否则返回False。
    """
    # 获取实体所在的图层
    doc = entity.doc
    layer = doc.layers.get(entity.dxf.layer)
    
    # 检查图层状态
    is_layer_off = layer.is_off()
    is_layer_frozen = layer.is_frozen()

    # # 检查实体自身的可见性
    # is_entity_color_invisible = entity.dxf.color == 0
    # is_entity_ltype_invisible = entity.dxf.linetype == "None"
    is_entity_invisible = entity.dxf.invisible
    # 结论
    return is_layer_off  or is_entity_invisible or is_layer_frozen


# 此函数内部与converBlocks内部有些不同，例如convertBlock还要处理Solid,ATTDEF，故暂且用到convertBlock中
def analyzeNonBlockEntity(doc, e):
    j = None
    type = e.dxftype()
    if (type == 'LINE'):
        j = convertLine(doc, e)
    elif type == 'CIRCLE':
        j = convertCircle(doc, e)
    elif type == 'TEXT':
        # 空文字直接删了
        if (e.dxf.text != ''):
            j = convertText(e)
    elif type == 'ARC':
        j = convertArc(doc, e)
    elif type == 'SOLID':
        j = convertSolid(e)  
    elif type == 'POLYLINE':
        j = convertPolyLine(doc, e)
    elif type == 'SPLINE':
        j = convertSpline(doc, e)
    elif type == 'LWPOLYLINE':
        j = convertLWPolyline(doc, e)
    elif type == 'DIMENSION':
        j = convertDimension(e)
    elif type == 'MTEXT':
        j = convertMText(e)
    elif type == 'ELLIPSE':
        j = convertEllipse(e)
    elif type == 'REGION':
        print('region 没什么好获取的信息')
        # j = convertRegion(e)
    elif type == 'HATCH':
        j = convertHatch(e)            #未实现
    # elif type == 'OLE2FRAME':
    #     pass            #未实现
    elif type == 'LEADER':
        j = convertLeader(e)           

    else:
        # raise NotImplementedError(f'遇到了未见过的类型 {type}')
        print(e.dxf.handle)
        # print(f'Nonblock中遇到了未见过的类型 {e.dxf.handle, type}')

    return j

# dxfname不需要携带.dxf后缀
def dxf2json(dxfpath,dxfname,output_folder):
    print('----dxf2json START-----')
    relpath = os.path.join(dxfpath,dxfname)
    relpath_dxf = relpath + '.dxf'
    print('dxf relative path is ' + relpath)
    doc = ezdxf.readfile(relpath_dxf,encoding='utf-8')
    msp = doc.modelspace()
    res = [] 
    block_list = []




    # print( doc.query('VIEWPORT'))

    # for viewport in doc.query('VIEWPORT'):
    #     print(viewport.dxf.handle, viewport.dxf.status, viewport.clipping_rect(), viewport.clipping_rect_corners(), viewport.get_scale())


    # assert 0 



    for e in msp:
        if e.dxf.layer == "SPLIT" or e.dxf.layer == "Split":
            continue
        
        if (isEntityHidden(e)):
            print(f'检测到隐藏元素 {e.dxf.handle} {e.dxftype()} 位置{ezdxf.bbox.extents([e])}')
            continue
        j = None
        try:
            # print(e.dxf.handle, e.dxftype())
            if (e.dxftype() == 'INSERT'):
                j, block_list = convertInsert(e, block_list)
            else:
                j = analyzeNonBlockEntity(doc, e)
        except Exception as e:
            print(e)

        if(j is not None):
            res.append(j)


    blocks = convertBlocks(doc, block_list)
    
    res = [res, blocks]
    json_str= json.dumps(res, indent=4)
    json_name = os.path.join(output_folder, dxfname) + ".json"
    # json_name = relpath + '.json'
    with open(json_name,'w',encoding='utf-8') as f:
        f.write(json_str)
        print(f'Writing to {json_name} finished!!')
        print('-----dxf2json END------')

if __name__ == "__main__":
    

    dxfpath = './测试数据'
    dxfname = 'test0717'
    dxf2json(dxfpath, dxfname, dxfpath)


    # dxfpath = './split'
    # folder_path = Path('split')
    # names = [f.stem for f in folder_path.glob('*.dxf')]
    # for name in names:
    #     dxf2json(dxfpath,name)
    
   

