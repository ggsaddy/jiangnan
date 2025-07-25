import math

class DPoint:  
    def __init__(self, x=0, y=0):  
        self.x = x
        self.y =y 
    def __eq__(self, other):  
        # 如果 other 也是 Point 实例，并且 x 和 y 坐标相等，则返回 True  
        if isinstance(other, DPoint):  
            return (round(self.x/3), round(self.y/3)) == (round(other.x/3), round(other.y/3))  
        return False  
  
    def __hash__(self):  
        # 返回 (x, y) 元组的哈希值  
        return hash((round(self.x/3), round(self.y/3)))  
    def __getitem__(self, index):  
        # 支持通过索引访问坐标  
        if index == 0:  
            return self.x  
        elif index == 1:  
            return self.y  
        else:  
            raise IndexError("Point index out of range (0 or 1 expected)")  
  
    def __setitem__(self, index, value):  
        # 支持通过索引修改坐标  
        if index == 0:  
            self.x = value
        elif index == 1:  
            self.y = value
        else:  
            raise IndexError("Point index out of range (0 or 1 expected)") 
    def __repr__(self):  
        return f"Point({self.x}, {self.y})"  
    def as_tuple(self):
        return (self.x,self.y)


class DSegment:  
    # Segment is essentially a Line with an implied direction and length  
    def __init__(self, start_point: DPoint=DPoint(0,0), end_point: DPoint=DPoint(1,0),ref=None):  
        self.start_point = start_point  
        self.end_point = end_point  
        self.ref=ref
        self.isConstraint=False
        self.isCornerhole=False
        self.StarCornerhole = None
        self.isPart=False
        self.isFb=False
    def initialize(self):
        self.isConstraint=False
        self.StarCornerhole = None
        self.isPart=False
        self.isFb=False
    def __len__(self):
        return 2

    def __eq__(self, other):  
        if isinstance(other, DSegment):  
            return (self.start_point, self.end_point) == (other.start_point, other.end_point)  
        return False  
  
    def __hash__(self):  
        return hash((self.start_point, self.end_point))
    def __getitem__(self, index):  
        # 支持通过索引访问坐标  
        if index == 0:  
            return self.start_point 
        elif index == 1:  
            return self.end_point  
        else:  
            raise IndexError("Point index out of range (0 or 1 expected)")  

    def __setitem__(self, index, value):  
        # 支持通过索引修改坐标  
        if index == 0:  
            self.start_point = value  
        elif index == 1:  
            self.end_point = value  
        else:  
            raise IndexError("Point index out of range (0 or 1 expected)") 
    def length(self):  
        return ((self.end_point.x - self.start_point.x) ** 2 +   
                (self.end_point.y - self.start_point.y) ** 2) ** 0.5  
    def mid_point(self):
        return DPoint((self.start_point.x+self.end_point.x)/2,(self.start_point.y+self.end_point.y)/2)
  
    def __repr__(self):  
        return f"Segment({self.start_point}, {self.end_point}, length={self.length()}, ref={self.ref})"  
    # def setConstraint(self,isConstraint=0):
    #     if isConstraint:
    #         self.isConstraint=2

#color=7 white color=3 green
class DElement:  
    def __init__(self):  
        pass
    def coordinatesmap(self,p:DPoint,insert,scales,rotation):
        rr=rotation/180*math.pi
        cosine=math.cos(rr)
        sine=math.sin(rr)

        # x,y=(p[0]*scales[0]+100)/200,(p[1]*scales[1]+100)/200
        x,y=(cosine*p[0]*scales[0]-sine*p[1]*scales[1])+insert[0],(sine*p[0]*scales[0]+cosine*p[1]*scales[1])+insert[1]
        return DPoint(x,y)
    def transform_point(self,point,meta):
        return self.coordinatesmap(point,meta.insert,meta.scales,meta.rotation)
class DLine(DElement):  
    def __init__(self, start_point: DPoint=DPoint(0,0), end_point: DPoint=DPoint(1,0), linetype="Continuous",color=7,handle="",meta=None):  
        super().__init__()
        self.start_point = start_point  
        self.end_point = end_point  
        self.color=color
        self.handle=handle
        self.meta=meta
        self.linetype=linetype.upper()
        # self.computeCenterCoordinateAndWeight()

    def __repr__(self):  
        return f"Line({self.start_point}, {self.end_point},handle:{self.handle})"  
    
    def __eq__(self,other):
        if not isinstance(other,DLine):
            return False
        else:
            return (self.start_point,self.end_point,self.color,self.handle)==(other.start_point,other.end_point,other.color,other.handle)

    # def computeCenterCoordinateAndWeight(self):
    #     s=DSegment(self.start_point,self.end_point,None)
    #     self.weight=s.length()
    #     self.bc=DPoint((s.start_point.x+s.end_point.x)/2,(s.start_point.y+s.end_point.y)/2)
   
    def transform(self):
        self.start_point=self.transform_point(self.start_point,self.meta)
        self.end_point=self.transform_point(self.end_point,self.meta)

class DLwpolyline(DElement):  
    def __init__(self, points: list[DPoint], linetype="Continuous",color=7,isClosed=False,handle="",isLwPolyline=False,verticesType=[],ori_vertices=[],hasArc=False,meta=None):  
        super().__init__()
        self.points = points
        self.color=color  
        self.isClosed=isClosed
        self.handle=handle
        self.isLwPolyline=isLwPolyline
        self.verticesType=verticesType
        self.ori_vertices=ori_vertices
        self.hasArc=hasArc
        self.meta=meta
        self.linetype=linetype.upper()
        # self.computeCenterCoordinateAndWeight()
  
  
    def __repr__(self):  
        return f"Lwpolyline(points:{self.points},color:{self.color},isClosed:{self.isClosed},handle:{self.handle})"  
    
    def __eq__(self,other):
        if not isinstance(other,DLwpolyline):
            return False
        else:
            if len(self.points) != len(other.points):
                return False
            else:
                for i in range(len(self.points)):
                    if self.points[i]!=other.points[i]:
                        return False
                
                return (self.isClosed,self.color,self.handle)==(other.isClosed,other.color,other.handle)
    def transform(self):
        newpoints=[]
        for p in self.points:
            newpoints.append(self.transform_point(p,self.meta))
        self.points=newpoints
    # def computeCenterCoordinateAndWeight(self):
    #     w=0
    #     x=0
    #     y=0
    #     n=len(self.points)
    #     for i in range(n-1):

    #         s=DSegment(self.points[i],self.points[i+1],None)
    #         l=s.length()
    #         w+=l
    #         x+=l*(s.start_point.x+s.end_point.x)/2
    #         y+=l*(s.start_point.y+s.end_point.y)/2
    #     if self.isClosed:
    #         s=DSegment(self.points[-1],self.points[0],None)
    #         l=s.length()
    #         w+=l
    #         x+=l*(s.start_point.x+s.end_point.x)/2
    #         y+=l*(s.start_point.y+s.end_point.y)/2
    #     self.weight=w
    #     self.bc=DPoint(x/w,y/w)


class DArc(DElement):  
    def __init__(self, center: DPoint, radius: float, start_angle: float, end_angle: float,linetype="Continuous",color=7,handle="",meta=None):  
        super().__init__()
        self.center = center  
        self.radius = radius  
        self.start_angle = start_angle  # in degrees  
        self.end_angle = end_angle      # in degrees  
        self.color=color
        self.handle=handle
        sa=start_angle/180*math.pi
        ea=end_angle/180*math.pi
        c1=math.cos(sa)
        s1=math.sin(sa)
        c2=math.cos(ea)
        s2=math.sin(ea)
        self.start_point=DPoint(center[0]+radius*c1,center[1]+radius*s1)
        self.end_point=DPoint(center[0]+radius*c2,center[1]+radius*s2)
        self.meta=meta
        self.linetype=linetype.upper()
  
    def __repr__(self):  
        return (f"Arc(center={self.center}, radius={self.radius}, "  
                f"start_angle={self.start_angle}, end_angle={self.end_angle}),handle:{self.handle}")  
    def  __eq__(self,other):
        if not isinstance(other,DArc):
            return False
        else:
            return (self.center,self.radius,self.start_angle,self.end_angle,self.color,self.handle)==(other.center,other.radius,other.start_angle,other.end_angle,other.color,other.handle)
    # def __eq__(self, other):
    #     if isinstance(other, DArc):
    #         return (self.center, self.radius, self.start_angle, self.end_angle) == (other.center, other.radius, other.start_angle, other.end_angle)
    #     return False
    def points_on_arc(self):
        # Convert start and end angles to radians
        sa = math.radians(self.start_angle)
        ea = math.radians(self.end_angle)

        # Compute start and end points
        start_point = DPoint(self.center.x + self.radius * math.cos(sa),
                             self.center.y + self.radius * math.sin(sa))
        end_point = DPoint(self.center.x + self.radius * math.cos(ea),
                           self.center.y + self.radius * math.sin(ea))

        return start_point, end_point
    def transform(self):
        self.center=self.transform_point(self.center,self.meta)
        self.radius=math.fabs(self.meta.scales[0])*self.radius
        # self.start_angle = self.start_angle+self.meta.rotation  # in degrees  
        # self.end_angle = self.end_angle+self.meta.rotation     # in degrees  
        # sa=self.start_angle/180*math.pi
        # ea=self.end_angle/180*math.pi
        # c1=math.cos(sa)
        # s1=math.sin(sa)
        # c2=math.cos(ea)
        # s2=math.sin(ea)
        # self.start_point=DPoint(self.center[0]+self.radius*c1,self.center[1]+self.radius*s1)
        # self.end_point=DPoint(self.center[0]+self.radius*c2,self.center[1]+self.radius*s2)
        self.start_point=self.transform_point(self.start_point,self.meta)
        self.end_point=self.transform_point(self.end_point,self.meta)
        self.start_angle=math.atan2(self.start_point.y-self.center.y,self.start_point.x-self.center.x)/math.pi*180
        self.end_angle=math.atan2(self.end_point.y-self.center.y,self.end_point.x-self.center.x)/math.pi*180
  

class DText(DElement):  
    def __init__(self,bound,insert=[0,0],color=7,content="",height=100,handle="",rotation=0,meta=None,is_mtext=False):  
        super().__init__()
        self.bound=bound
        self.insert=DPoint(insert[0],insert[1])
        
        self.color=color
        self.content=content.strip()
        self.height=height
        self.handle=handle
        self.textpos=False
        self.meta=meta
        self.is_mtext=is_mtext
        self.rotation=rotation
        if is_mtext:
            new_bound={"x1":self.insert.x,"x2":self.insert.x,"y1":self.insert.y,"y2":self.insert.y}
            self.bound=new_bound
  
    def __repr__(self):  
        return f"Text({self.insert}, color:{self.color},content:{self.content},height:{self.height},handle:{self.handle})"  
    def  __eq__(self,other):
        if not isinstance(other,DText):
            return False
        else:
            return (self.content,self.height,self.handle)==(other.content,other.height,other.handle)
    def __hash__(self):
        return hash((self.content,self.height,self.handle))
    def transform(self):
        self.insert=self.transform_point(self.insert,self.meta)
        self.height=math.fabs(self.meta.scales[0])*self.height
        if self.is_mtext:
            new_bound={"x1":self.insert.x,"x2":self.insert.x,"y1":self.insert.y,"y2":self.insert.y}
            self.bound=new_bound
        else:
            x1,x2,y1,y2=self.bound["x1"],self.bound["x2"],self.bound["y1"],self.bound["y2"]
            lb=DPoint(x1,y1)
            lt=DPoint(x1,y2)
            rt=DPoint(x2,y2)
            rb=DPoint(x2,y1)
            lb=self.transform_point(lb,self.meta)
            lt=self.transform_point(lt,self.meta)
            rt=self.transform_point(rt,self.meta)
            rb=self.transform_point(rb,self.meta)

            x_min,x_max,y_min,y_max=float("inf"),float("-inf"),float("inf"),float("-inf")
            x_min=min(lb.x,lt.x,rb.x,rt.x,x_min)
            x_max=max(lb.x,lt.x,rb.x,rt.x,x_max)
            y_min=min(lb.y,lt.y,rb.y,rt.y,y_min)
            y_max=max(lb.y,lt.y,rb.y,rt.y,y_max)
            new_bound={"x1":x_min,"x2":x_max,"y1":y_min,"y2":y_max}
            self.bound=new_bound

class DDimension(DElement):
    def __init__(self,textpos: DPoint=DPoint(0,0),color=7,text="",measurement=100,defpoints=[],dimtype=0,handle="",meta=None):  
        super().__init__()
        self.textpos=textpos
        self.text=text
        self.color=color
        self.measurement=measurement
        self.defpoints=defpoints
        self.handle=handle
        self.dimtype=dimtype
        self.meta=meta
        if self.text=="":
            if self.dimtype==37 or self.dimtype==34 or self.dimtype==162:
                self.text=str(round(self.measurement/math.pi*180))+"°"
            elif self.dimtype==163:
                self.text="Φ"+str(round(self.measurement))
            else:
                r_m=round(measurement)
                if r_m %10==9 or r_m %10==4:
                    r_m+=1
                err=r_m-measurement
                if err <1:
                    self.text=str(r_m)
                else:
                    self.text=str(round(self.measurement))
    def  __eq__(self,other):
        if not isinstance(other,DDimension):
            return False
        else:
            return (self.text,self.dimtype,self.handle)==(other.text,other.dimtype,other.handle)
    def __hash__(self):
        return hash((self.text,self.dimtype,self.handle))
  
    def __repr__(self):  
        return f"Dimension(pos:{self.textpos}, text:{self.text},color:{self.color},measurement:{self.measurement},defpoints:{self.defpoints},dimtype:{self.dimtype},handle:{self.handle})"  

    def transform(self):
        self.textpos=self.transform_point(self.textpos,self.meta)
        for i in range(len(self.defpoints)):
            self.defpoints[i]=self.transform_point(self.defpoints[i],self.meta)
class DCornorHole:
    cornor_hole_id=0
    def __init__(self,segments=[]):
        self.segments=segments
        self.ID=DCornorHole.cornor_hole_id
        DCornorHole.cornor_hole_id+=1

class DInsert:
    def __init__(self,blockName,scales,rotation,insert,attribs,bound):
        self.blockName=blockName
        self.scales=scales
        self.rotation=rotation
        self.insert=insert
        self.attribs=attribs
        self.bound=bound
    def mid_point(self):
        return DPoint((self.bound["x1"]+self.bound["x2"])*0.5,(self.bound["y1"]+self.bound["y2"])*0.5)



#accelerate structure

class DBlock:
    def __init__(self,segments, rect, M, N,min_cell_length,min_segments_num):
        self.rect=rect
        self.M=M
        self.N=N
        self.min_M=M
        self.min_N=N
        self.min_cell_length=min_cell_length
        self.min_segments_num=min_segments_num
        self.adapt_block()
        self.initialize(segments,rect,self.M,self.N)
    def clip_line(self,rect_x_min, rect_x_max, rect_y_min, rect_y_max, start_point, end_point):
        """
        裁剪线段使其位于包围盒内，如果线段在包围盒内，则返回裁剪后的起点和终点。
        :param rect_x_min: 包围盒最小x坐标
        :param rect_x_max: 包围盒最大x坐标
        :param rect_y_min: 包围盒最小y坐标
        :param rect_y_max: 包围盒最大y坐标
        :param start_point: 线段起点 (x1, y1)
        :param end_point: 线段终点 (x2, y2)
        :return: (裁剪后的起点, 裁剪后的终点)，如果线段完全在包围盒外，返回None
        """
        x1, y1 = start_point.x,start_point.y
        x2, y2 = end_point.x,end_point.y

        dx = x2 - x1
        dy = y2 - y1

        p = [-dx, dx, -dy, dy]
        q = [x1 - rect_x_min, rect_x_max - x1, y1 - rect_y_min, rect_y_max - y1]

        t_min = 0
        t_max = 1

        for i in range(4):
            if p[i] == 0:  # 线段平行于边界
                if q[i] < 0:
                    return None  # 完全在外部
            else:
                t = q[i] / p[i]
                if p[i] < 0:
                    t_min = max(t_min, t)  # 更新 t_min
                else:
                    t_max = min(t_max, t)  # 更新 t_max

        if t_min > t_max:
            return None  # 完全在外部

        # 计算裁剪后的起点和终点
        clipped_start = DPoint(x1 + t_min * dx, y1 + t_min * dy)
        clipped_end = DPoint(x1 + t_max * dx, y1 + t_max * dy)

        return clipped_start, clipped_end
    def adapt_block(self):
        rect_x_min, rect_x_max, rect_y_min, rect_y_max = self.rect
        xx,yy=rect_x_max-rect_x_min,rect_y_max-rect_y_min
        if xx< yy:
            ratio=int(yy/xx)
            self.N*=ratio
        else:
            ratio=int(xx/yy)
            self.M*=ratio
    def initialize(self,segments,rect,M,N):
        rect_x_min, rect_x_max, rect_y_min, rect_y_max = rect
        cell_width = (rect_x_max - rect_x_min) / M
        cell_height = (rect_y_max - rect_y_min) / N
        if cell_width<= self.min_cell_length or cell_height <=self.min_cell_length or len(segments)<=self.min_segments_num:
            self.leaf=True
            self.segments=segments
        else:
            self.leaf=False
            self.sub_blocks=[]
            grid=self.segments_in_blocks(segments)
            
            for i in range(N):
                sub_row=[]
                for j in range(M):
                    x,y=rect_x_min+j*cell_width,rect_y_min+i*cell_height
                    sub_row.append(DBlock(grid[i][j],(x-1,x+cell_width+1,y-1,y+cell_height+1),self.min_M,self.min_N,self.min_cell_length,self.min_segments_num))
                self.sub_blocks.append(sub_row)    
    def get_segment_blocks(self,segment,rect,M,N):
        rect_x_min, rect_x_max, rect_y_min, rect_y_max = rect
    
        result=self.clip_line(rect_x_min+0.1, rect_x_max-0.1, rect_y_min+0.1, rect_y_max-0.1, segment.start_point, segment.end_point)
        if result is None:
            print(segment,rect,M,N)
            clipped_start, clipped_end=segment.start_point,segment.end_point
        else:
            clipped_start, clipped_end=result
        clipped_segment=DSegment(clipped_start,clipped_end)


        cell_width = (rect_x_max - rect_x_min) / M
        cell_height = (rect_y_max - rect_y_min) / N

        # 获取线段的起点和终点
        x0, y0 = clipped_segment.start_point.x, clipped_segment.start_point.y
        x1, y1 = clipped_segment.end_point.x, clipped_segment.end_point.y

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
            if 0 <= col_index and col_index< M and 0 <= row_index  and row_index< N:
                # print(x0,y0,x1,y1,dx,dy)
                # print((x0-rect_x_min)/cell_width,(y0-rect_y_min)/cell_height)
                # print(col_index,row_index,final_col_index,final_row_index)
                grids.add((row_index, col_index))
            else:
                #assert(1==2)
                if 0 <= final_col_index and   final_col_index< M and 0 <= final_row_index  and final_row_index< N:
                    grids.add((final_row_index, final_col_index))
                break
            # 终止条件
            
            if col_index == final_col_index and row_index == final_row_index:
                break
            if col_index==final_col_index:
                if row_index==final_row_index+1 or row_index==final_row_index-1:
                    grids.add((row_index, col_index))
                    if 0 <= final_col_index and   final_col_index< M and 0 <= final_row_index  and final_row_index< N:
                        grids.add((final_row_index, final_col_index))
                    break
            if row_index == final_row_index:
                if col_index==final_col_index+1 or col_index==final_col_index-1:
                    grids.add((row_index, col_index))
                    if 0 <= final_col_index and   final_col_index< M and 0 <= final_row_index  and final_row_index< N:
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
    

    def segments_in_blocks(self,segments):
        M,N=self.M,self.N
        rect=self.rect

        grid=[]
        for i in range(N):
            row=[]
            for j in range(M):
                col=[]
                row.append(col)
            grid.append(row)
        #pbar=tqdm(total=len(segments),desc="test")
        for s in segments:
            #pbar.update()
            block_idxs=self.get_segment_blocks(s,rect,M,N)
            for idx in block_idxs:
                i,j=idx
                grid[i][j].append(s)
        #pbar.close()
        return grid


    def segments_near_segment(self,segment):
        if self.leaf:
            return self.segments
        grid_set=set()
        rect,M,N=self.rect,self.M,self.N
        g_set=self.get_segment_blocks(segment,rect,M,N)
        for g in g_set:
            grid_set.add(g)
        segments=set()
        for g in grid_set:
            i,j=g
            sub_s=self.sub_blocks[i][j].segments_near_segment(segment)
            for s in sub_s:
                segments.add(s)
        return list(segments)
    def segments_near_poly(self,poly):
        s_set=set()
        
        for s in poly:
            vs,ve=s.start_point,s.end_point
            l=s.length()
            dx,dy=ve.x-vs.x,ve.y-vs.y
            v=DPoint(dy/l*60,-dx/l*60)
            vs1,vs2,ve1,ve2=DPoint(vs.x+v.x,vs.y+v.y),DPoint(vs.x-v.x,vs.y-v.y),DPoint(ve.x+v.x,ve.y+v.y),DPoint(ve.x-v.x,ve.y-v.y)
            s1,s2=DSegment(vs1,ve1),DSegment(vs2,ve2)
            ss=self.segments_near_segment(s)
            ss1=self.segments_near_segment(s1)
            ss2=self.segments_near_segment(s2)
            for s in ss:
                s_set.add(s)
            for s in ss1:
                s_set.add(s)
            for s in ss2:
                s_set.add(s)
        return list(s_set)



def build_initial_block(segments,segmentation_config):
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


    return DBlock(segments,rect,M,N,segmentation_config.min_cell_length,segmentation_config.min_segments_num)