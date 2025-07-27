import ezdxf
import ezdxf.bbox
import json
import os
from typing import Any, Dict, List, Optional
import math

class DXFConverterV2:
    """DXF文件转换器，将DXF文件转换为JSON格式"""
    
    PI_CIRCLE = 2 * math.pi
    
    def __init__(self, selected_layer: str):
        """
        初始化转换器
        Args:
            selected_layer: 要处理的图层名称
        """
        self.selected_layer = selected_layer
        self.entity_converters = {
            'LINE': self._convert_line,
            'CIRCLE': self._convert_circle,
            'TEXT': self._convert_text,
            'ARC': self._convert_arc,
            'POLYLINE': self._convert_polyline,
            'SPLINE': self._convert_spline,
            'LWPOLYLINE': self._convert_lwpolyline,
            'DIMENSION': self._convert_dimension,
            'MTEXT': self._convert_mtext,
            'ELLIPSE': self._convert_ellipse,
            'HATCH': self._convert_hatch,
        }
    
    def _approximate_equal(self, a: float, b: float, epsilon: float = 1e-6) -> bool:
        """比较两个浮点数是否近似相等"""
        return abs(a - b) < epsilon
    
    def _vector_to_angle(self, vector: tuple) -> float:
        """将向量转换为角度"""
        x, y = vector
        angle = math.atan2(y, x)
        return angle + self.PI_CIRCLE if angle < 0 else angle
    
    def _get_color(self, entity: Any) -> int:
        """获取实体颜色"""
        try:
            color = entity.dxf.color
            if color == 0 or color == 256:
                return entity.doc.layers.get(entity.dxf.layer).color
            return color
        except Exception:
            return 256
    
    def _get_line_type(self, doc: Any, entity: Any) -> str:
        """获取线型"""
        try:
            linetype = entity.dxf.linetype
            if linetype == "BYLAYER":
                return doc.layers.get(entity.dxf.layer).dxf.linetype
            return linetype
        except Exception:
            return "CONTINUOUS"
    
    def _convert_entity_base(self, entity_type: str, entity: Any) -> Dict:
        """转换实体的基本属性"""
        bb = ezdxf.bbox.extents([entity])
        return {
            'type': entity_type,
            'color': self._get_color(entity),
            'layerName': entity.dxf.layer,
            'handle': entity.dxf.handle,
            'bound': {
                'x1': bb.extmin[0],
                'y1': bb.extmin[1],
                'x2': bb.extmax[0],
                'y2': bb.extmax[1],
            }
        }
    
    def _convert_circle(self, doc: Any, entity: Any) -> Dict:
        """转换圆实体"""
        result = self._convert_entity_base('circle', entity)
        result.update({
            'center': [entity.dxf.center[0], entity.dxf.center[1]],
            'radius': entity.dxf.radius
        })
        return result
    
    def _convert_text(self, doc: Any, entity: Any) -> Optional[Dict]:
        """转换文本实体"""
        if not entity.dxf.text:
            return None
        result = self._convert_entity_base('text', entity)
        result.update({
            'insert': [entity.dxf.insert[0], entity.dxf.insert[1]],
            'content': entity.dxf.text,
            'height': entity.dxf.height
        })
        return result
    
    def _convert_line(self, doc: Any, entity: Any) -> Dict:
        """转换直线实体"""
        result = self._convert_entity_base('line', entity)
        result.update({
            'start': [entity.dxf.start[0], entity.dxf.start[1]],
            'end': [entity.dxf.end[0], entity.dxf.end[1]],
            'linetype': self._get_line_type(doc, entity)
        })
        return result
    
    def _convert_arc(self, doc: Any, entity: Any) -> Dict:
        """转换圆弧实体"""
        result = self._convert_entity_base('arc', entity)
        result.update({
            'linetype': self._get_line_type(doc, entity),
            'center': [entity.dxf.center[0], entity.dxf.center[1]],
            'radius': entity.dxf.radius,
            'startAngle': entity.dxf.start_angle,
            'endAngle': entity.dxf.end_angle
        })
        return result

    def _convert_polyline(self, doc: Any, entity: Any) -> Dict:
        """转换多段线实体"""
        result = self._convert_entity_base('polyline', entity)
        result.update({
            'isClosed': entity.is_closed,
            'linetype': self._get_line_type(doc, entity),
            'vertices': [[v.dxf.location[0], v.dxf.location[1]] for v in entity.vertices]
        })
        return result

    def _convert_spline(self, doc: Any, entity: Any) -> Dict:
        """转换样条曲线实体"""
        result = self._convert_entity_base('spline', entity)
        result.update({
            'linetype': self._get_line_type(doc, entity),
            'vertices': [x.tolist()[0:2] for x in entity.control_points]
        })
        return result

    def _convert_lwpolyline(self, doc: Any, entity: Any) -> Dict:
        """转换轻量多段线实体"""
        result = self._convert_entity_base('lwpolyline', entity)
        result.update({
            'isClosed': entity.is_closed,
            'linetype': self._get_line_type(doc, entity),
            'hasArc': entity.has_arc,
            'vertices': [],
            'verticesType': []
        })
        
        self._process_lwpolyline_vertices(entity, result)
        return result

    def _process_lwpolyline_vertices(self, entity: Any, result: Dict):
        """处理轻量多段线的顶点"""
        for i in range(entity.dxf.count - 1):
            self._process_vertex_pair(entity, i, i + 1, result)
        
        if entity.is_closed:
            self._process_vertex_pair(entity, entity.dxf.count - 1, 0, result)

    def _process_vertex_pair(self, entity: Any, start_idx: int, end_idx: int, result: Dict):
        """处理一对相邻顶点"""
        start_point = entity.__getitem__(start_idx)
        end_point = entity.__getitem__(end_idx)
        
        if start_point[-1] != 0.:
            arc = ezdxf.math.bulge_to_arc(start_point[:2], end_point[:2], start_point[-1])
            center, start_angle, end_angle, radius = arc
            result["vertices"].append([center[0], center[1], start_angle, end_angle, radius])
            result['verticesType'].append("arc")
        else:
            result["vertices"].append([start_point[0], start_point[1], end_point[0], end_point[1]])
            result['verticesType'].append("line")

    def _convert_dimension(self, doc: Any, entity: Any) -> Dict:
        """转换标注实体"""
        result = self._convert_entity_base('dimension', entity)
        result.update({
            'measurement': entity.dxf.actual_measurement,
            'text': entity.dxf.text,
            'dimtype': entity.dxf.dimtype,
            'textpos': [entity.dxf.text_midpoint[0], entity.dxf.text_midpoint[1]],
            'defpoint1': [entity.dxf.defpoint[0], entity.dxf.defpoint[1]],
            'defpoint2': [entity.dxf.defpoint2[0], entity.dxf.defpoint2[1]],
            'defpoint3': [entity.dxf.defpoint3[0], entity.dxf.defpoint3[1]],
            'defpoint4': [entity.dxf.defpoint4[0], entity.dxf.defpoint4[1]],
            'defpoint5': [entity.dxf.defpoint5[0], entity.dxf.defpoint5[1]]
        })
        return result

    def _convert_mtext(self, doc: Any, entity: Any) -> Dict:
        """转换多行文本实体"""
        result = self._convert_entity_base('mtext', entity)
        result.update({
            'insert': [entity.dxf.insert[0], entity.dxf.insert[1]],
            'width': entity.dxf.width,
            'text': entity.dxf.text
        })
        return result

    def _convert_ellipse(self, doc: Any, entity: Any) -> Dict:
        """转换椭圆实体"""
        result = self._convert_entity_base('ellipse', entity)
        
        # 基本属性
        result.update({
            'center': [entity.dxf.center[0], entity.dxf.center[1]],
            'major_axis': [entity.dxf.major_axis[0], entity.dxf.major_axis[1]],
            'ratio': entity.dxf.ratio,
            'extrusion': [entity.dxf.extrusion[0], entity.dxf.extrusion[1], entity.dxf.extrusion[2]],
            'start_param': entity.dxf.start_param,
            'end_param': entity.dxf.end_param,
        })
        
        # 计算角度
        start_theta = entity.dxf.start_param
        end_theta = entity.dxf.end_param
        
        if self._approximate_equal(result['extrusion'][2], -1):
            start_theta, end_theta = self.PI_CIRCLE - end_theta, self.PI_CIRCLE - start_theta
            
        major_axis_rotation = self._vector_to_angle(result['major_axis'])
        start_theta = (start_theta + major_axis_rotation) % self.PI_CIRCLE
        end_theta = (end_theta + major_axis_rotation) % self.PI_CIRCLE
        
        if start_theta > end_theta:
            if self._approximate_equal(end_theta, 0, 1e-2):
                end_theta = self.PI_CIRCLE
            if self._approximate_equal(start_theta, self.PI_CIRCLE, 1e-2):
                start_theta = 0
                
        result.update({
            'calc_start_theta': start_theta,
            'calc_end_theta': end_theta
        })
        
        return result

    def _convert_hatch(self, doc: Any, entity: Any) -> Dict:
        """转换填充实体"""
        result = self._convert_entity_base('hatch', entity)
        
        # 计算面积（与load.py保持一致）
        x1 = result["bound"]["x1"]
        x2 = result["bound"]["x2"]
        y1 = result["bound"]["y1"]
        y2 = result["bound"]["y2"]
        
        s = math.fabs((x2 - x1) * (y2 - y1))
        
        return result

    def _is_entity_hidden(self, entity: Any) -> bool:
        """检查实体是否被隐藏"""
        layer = entity.doc.layers.get(entity.dxf.layer)
        return layer.is_off() or entity.dxf.invisible

    def convert_entity(self, doc: Any, entity: Any) -> Optional[Dict]:
        """转换单个实体"""
        if (entity.dxf.layer != self.selected_layer or 
            self._is_entity_hidden(entity)):
            return None
            
        entity_type = entity.dxftype()
        converter = self.entity_converters.get(entity_type)
        
        if converter:
            try:
                return converter(doc, entity)
            except Exception as e:
                print(f"转换实体时出错: {entity_type}, {e}")
                return None
        return None
    
    def convert_hatch_entity(self, doc: Any, entity: Any) -> Optional[Dict]:
        """转换单个实体"""    
        entity_type = entity.dxftype()
        converter = self.entity_converters.get(entity_type)
        
        if converter:
            try:
                return converter(doc, entity)
            except Exception as e:
                print(f"转换实体时出错: {entity_type}, {e}")
                return None
        return None
    

    def get_bboxes(self, doc: Any) -> List[List[float]]:
        """
        Convert DXF bounding boxes and their confidence scores into a list of [x1, y1, x2, y2, conf] format
        
        Returns:
            List[List[float]]: List of bounding boxes with format [[x1, y1, x2, y2, conf], ...]
        """
        # Get all entities in modelspace
        msp = doc.modelspace()
        
        # Initialize containers for lines and texts
        lines = []
        texts = {}
        entities = []  # updated
        layer_names = set()
        # First pass: collect all lines and texts from the selected layer
        rectangles = []
        lw_rectangles = []
        for entity in msp:
            layer_names.add(entity.dxf.layer)
            # if entity.dxf.layer != self.selected_layer or self._is_entity_hidden(entity):
            if entity.dxf.layer != self.selected_layer:
                continue
            # print(dir(entity.dxf))
            # exit()
            converted = self.convert_entity(doc, entity)
            if converted:
                entities.append(converted)

            if entity.dxftype() == 'LINE':
                lines.append({
                    'start': [entity.dxf.start[0], entity.dxf.start[1]],
                    'end': [entity.dxf.end[0], entity.dxf.end[1]]
                })
            elif entity.dxftype() == 'TEXT':
                # Extract confidence value from text content
                if entity.dxf.text.startswith('Holes:'):
                    try:
                        conf = float(entity.dxf.text.split(':')[1].strip())
                        # Store text position and confidence
                        texts[entity.dxf.insert[1]] = conf
                    except (ValueError, IndexError):
                        continue
            elif entity.dxftype() == 'LWPOLYLINE':
                result = self._convert_entity_base('lwpolyline', entity)
                lw_rectangles.append(
                    [result["bound"]["x1"],
                    result["bound"]["y1"],
                    result["bound"]["x2"],
                    result["bound"]["y2"]]
                )

        for ret in lw_rectangles:
            x1, y1, x2, y2 = ret
            if texts:
                closest_y = min(texts.keys(), key=lambda y: abs(y - y2))
                conf = texts[closest_y]
            else:
                conf = 1
            rectangles.append([x1, y1, x2, y2, conf])
        print("layer_names = ", layer_names)
        # Group lines into rectangles
        used_lines = set()
        
        for i, line1 in enumerate(lines):
            if i in used_lines:
                continue
                
            # Find connected lines that form a rectangle
            connected_lines = [line1]
            current_point = line1['end']
            found_lines = {i}
            
            # Try to find 3 more connected lines
            for _ in range(3):
                found_connection = False
                for j, line2 in enumerate(lines):
                    if j in found_lines:
                        continue
                        
                    # Check if lines are connected (start or end points match)
                    if self._approximate_equal(current_point[0], line2['start'][0]) and \
                    self._approximate_equal(current_point[1], line2['start'][1]):
                        connected_lines.append(line2)
                        current_point = line2['end']
                        found_lines.add(j)
                        found_connection = True
                        break
                    elif self._approximate_equal(current_point[0], line2['end'][0]) and \
                        self._approximate_equal(current_point[1], line2['end'][1]):
                        connected_lines.append({
                            'start': line2['end'],
                            'end': line2['start']
                        })
                        current_point = line2['start']
                        found_lines.add(j)
                        found_connection = True
                        break
                        
                if not found_connection:
                    break
            
            # If we found 4 connected lines and it forms a closed shape
            if len(connected_lines) == 4 and \
            self._approximate_equal(connected_lines[0]['start'][0], connected_lines[-1]['end'][0]) and \
            self._approximate_equal(connected_lines[0]['start'][1], connected_lines[-1]['end'][1]):
                
                # Find bounding box coordinates
                xs = []
                ys = []
                for line in connected_lines:
                    xs.extend([line['start'][0], line['end'][0]])
                    ys.extend([line['start'][1], line['end'][1]])
                
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                
                # Find associated confidence score
                conf = None
                y_center = (y1 + y2) / 2
                # Find the closest text y-coordinate
                if texts:
                    closest_y = min(texts.keys(), key=lambda y: abs(y - y2))
                    conf = texts[closest_y]
                
                if conf is not None:
                    rectangles.append([x1, y1, x2, y2, conf])
                else:
                    rectangles.append([x1, y1, x2, y2, 1])
                used_lines.update(found_lines)
        
        if len(rectangles) == 0:
            for entity in entities:
                if "bound" in entity:
                    bbox = entity["bound"]
                    rectangles.append([
                        bbox["x1"], 
                        bbox["y1"], 
                        bbox["x2"], 
                        bbox["y2"], 
                        1  # 默认置信度为1
                    ])
        # print(rectangles)
        # print(len(rectangles))
        # exit()

        return rectangles


    def convert_file(self, input_path: str, output_path: str):
        """
        转换DXF文件到JSON
        Args:
            input_path: DXF文件路径
            output_path: 输出JSON文件路径
        """
        try:
            # 读取DXF文件
            doc = ezdxf.readfile(input_path, encoding='utf-8')
            bboxes = self.get_bboxes(doc)
            msp = doc.modelspace()
            
            # 转换实体
            entities = []
            hatch_entities = []
            for entity in msp:
                converted = self.convert_hatch_entity(doc, entity)
                if converted:
                    entities.append(converted)
                    # 收集hatch实体（只收集来自selected_layer的）
                    if converted.get('type') == 'hatch':
                        hatch_entities.append(converted)
            
            # 构建hatch边界框列表
            hatch_bboxes = []
            for hatch in hatch_entities:
                if 'bound' in hatch:
                    bbox = hatch['bound']
                    hatch_bboxes.append([
                        bbox['x1'],
                        bbox['y1'], 
                        bbox['x2'],
                        bbox['y2'],
                        1  # 默认置信度为1
                    ])
            
            # 写入JSON文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(entities, f, indent=4)
                
            print(f'Successfully converted {input_path} to {output_path}')
            print(f'Found {len(hatch_entities)} hatch entities in layer "{self.selected_layer}"')
            print(f'Total entities: {len(entities)}')

            return bboxes, hatch_bboxes
            
        except Exception as e:
            print(f"转换文件时出错: {e}")
            return [], []

def main():
    """主函数"""
    # dxf_path = './train.dxf'
    dxf_path = "/Users/ieellee/Documents/FDU/ship/holes_detection/shadow.dxf"
    output_path = './shadow.json'
    selected_layer = "200-1CO3"
    
    converter = DXFConverterV2(selected_layer)
    bboxes, hatch_bboxes = converter.convert_file(dxf_path, output_path)
    print(f"bboxes = {bboxes}")
    print(f"hatch_bboxes = {hatch_bboxes}")
    with open("bbox_temp.txt", "w") as f:
        for bbox in bboxes:
            f.write(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{bbox[4]}\n")
    

if __name__ == "__main__":
    main()


'''
bboxes = [[115071.0438846994, 95100.28716024933, 117877.7824240581, 97471.0789052988, 1], [118273.8566707747, 94117.39036345725, 121704.6160932686, 97220.92871731373, 1], [117810.4589573583, 91288.81903690816, 121220.186901226, 94010.55984358289, 1], [102486.991227696, 66935.28240261717, 103281.991227696, 68114.90722729189, 1], [103331.9912277074, 66932.3259494226, 104146.9912276959, 68111.95077409719, 1]]
'''