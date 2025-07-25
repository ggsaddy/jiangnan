from shapely.geometry import box, LineString
from shapely.ops import unary_union
import matplotlib.pyplot as plt

# Calculate the area covered by a bounding box
def calculate_area(bbox):
    width = bbox['x2'] - bbox['x1']
    height = bbox['y2'] - bbox['y1']
    return width * height

#并查集
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    
    def init_by_one(self):
        for i in range(len(self.parent)):
            self.parent[i] = 0
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        rootx = self.find(x)
        rooty = self.find(y)
        if rootx != rooty:
            self.parent[rooty] = rootx
    
    def remove(self, index):
        self.parent.pop(index)
    
    def __getitem__(self, index):
        return self.parent[index]
    
    def __len__(self):
        return len(self.parent)

#每一个小零件是一个component
class component:
    def __init__(self, component_type, bbox, data):
        self.type = component_type
        self.cluster_type = component_type
        self.bbox = bbox
        self.data = data #包含字典里所有属性，可使用comp.data["xxx"]进行索引
        self.need_judge = True
        self.title = None
        self.is_merged = False
        self.manual_index = 0 #人工处理的index
    
    def update_types(self, new_type):
        self.cluster_type = new_type


#一个all_componets包含多个component
class all_components:
    def __init__(self, value=None, title=None):
        if value == None:
            self.title = None
            self.sub_title = None
            self.component_list = [] #存放component
            self.groups = {}
            self.uf = None
        else:
            #拷贝初始化，并merge到一起
            self.title = title
            self.sub_title = None
            self.component_list = value
            self.groups = {}
            self.uf = UnionFind(len(value))
            self.uf.init_by_one()
            self.update_groups()
    
    def bind_uf(self, uf: UnionFind):
        self.uf = uf
    
    def push(self, component):
        if self.uf is not None:
            self.uf.parent.append(len(self.uf.parent))
        self.component_list.append(component)
        
    #额外加入一个component
    def add_one_extra(self, another: component):
        self.push(another)
        self.update_groups()
    
    #根据并查集里的关系更新包围盒
    def update_groups(self, index=None):
        # Update and create box groups
        self.groups = {}
        if index != None: #如果指定index
            for i in index:
                root = self.uf.find(i)
                if root not in self.groups:
                    self.groups[root] = self.component_list[i].bbox
                else:
                    self.groups[root] = {
                        'x1': min(self.groups[root]['x1'], self.component_list[i].bbox['x1']),
                        'y1': min(self.groups[root]['y1'], self.component_list[i].bbox['y1']),
                        'x2': max(self.groups[root]['x2'], self.component_list[i].bbox['x2']),
                        'y2': max(self.groups[root]['y2'], self.component_list[i].bbox['y2']),
                    }
        else:
            for i in range(len(self.uf)):
                root = self.uf.find(i)
                if root not in self.groups:
                    self.groups[root] = self.component_list[i].bbox
                else:
                    self.groups[root] = {
                        'x1': min(self.groups[root]['x1'], self.component_list[i].bbox['x1']),
                        'y1': min(self.groups[root]['y1'], self.component_list[i].bbox['y1']),
                        'x2': max(self.groups[root]['x2'], self.component_list[i].bbox['x2']),
                        'y2': max(self.groups[root]['y2'], self.component_list[i].bbox['y2']),
                    }
    
    def __len__(self):
        return len(self.component_list)

    def __getitem__(self, index):
        return self.component_list[index]
    
    #获取多边形包围盒
    def get_polygon_bbox(self):
        polygons = [{"x1": self.component_list[b].bbox["x1"],
                     "y1": self.component_list[b].bbox["y1"],
                     "x2": self.component_list[b].bbox["x2"],
                     "y2": self.component_list[b].bbox["y2"]} for b in range(len(self.component_list))]
        
        geometries = []
        for b in polygons:
            if b["x1"] == b["x2"] or b["y1"] == b["y2"]:
                geometries.append(LineString([(b['x1'],b['y1']), (b['x2'], b['y2'])]))
            else:
                geometries.append(box(b['x1'], b['y1'], b['x2'], b['y2']))

        merged_polygon = unary_union(geometries)
        merged_polygon = merged_polygon.convex_hull

        return merged_polygon
    
    def get_final_bbox(self):
        return list(self.groups.values())
    
    def return_result(self):
        segment_dict = {}
        for i in range(len(self.uf)):
            if self.uf[i] in segment_dict:
                segment_dict[self.uf[i]].append(self.component_list[i].data)
            else:
                segment_dict[self.uf[i]] = []
                segment_dict[self.uf[i]].append(self.component_list[i].data)
        return segment_dict.values()
    
    def return_group(self):
        segment_dict = {}
        for i in range(len(self.uf)):
            if self.uf[i] in segment_dict:
                segment_dict[self.uf[i]].push(self.component_list[i])
            else:
                segment_dict[self.uf[i]] = all_components()
                segment_dict[self.uf[i]].push(self.component_list[i])
        group_list = segment_dict.values()
        return group_list
    
    #merge两个all_components
    def merge(self, other):
        for i in range(len(other)):
            self.push(other[i])
            
     #merge另一个 comp
    def merge_without_update(self, another_comp):
        self.push(another_comp)
    
    def update_after_merge(self):
        self.uf = UnionFind(len(self.component_list))
        self.uf.init_by_one()
        self.update_groups()

class sub_components:
    def __init__(self, title: str, components: all_components):
        self.title = title
        self.components = components