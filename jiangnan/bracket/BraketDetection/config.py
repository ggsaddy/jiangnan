class SegmentationConfig:
    def __init__(self):
        self.check=False
        self.verbose=True
        # self.mode="dev"
        self.mode="pro"
        self.bracket_layer=None
        self.json_path=""
        
        #grid setting
      


        self.line_expand_length=7.5
        self.arc_expand_length=5
        self.line_image_path="./output/line.png"
        self.draw_intersections=False

        self.draw_segments=True
        self.line_image_drawPolys=True
        self.draw_line_image = True
        self.draw_texts=False

        self.draw_poly_nums=1000
        self.poly_image_dir="./output"
        self.draw_polys=True
        self.draw_geometry=True

        self.segment_filter_length=0.25
        self.segment_filter_iters=100
        self.segment_remove_interval=50
        self.segment_split_epsilon=0.25

        self.intersection_epsilon=1e-9
        #包围盒
        self.bbox_min_area=5000
        self.bbox_max_area=1000*15000
        self.bbox_ratio=12


        self.remove_tolerance=1e-5

        #聚类算法合并重复路径
        self.eps=25
        self.min_samples=1

      

        self.poly_info_dir = "./output"
        self.res_image_path = f"./output/res.png"

        self.max_workers=4
        self.timeout=2

        self.type_path = "./type.json"
        self.standard_type_path = "./standard_type.json"
        self.unstandard_type_path = "./unstandard_type.json"
        self.dxf_output_folder = "./output/"

        self.json_output_path = "./output/bracket.json"   #输出解析后的肘板轮廓，便于调整匹配算法

        #肘板边数
        self.path_max_length = 30
        self.path_min_length = 3

        self.dfs_path_max_length=100
        self.dfs_path_min_length=3
        #compute cornor_hole
        self.repline_neighbor_min_length=8

        #repline邻域边最小长度
        self.check_valid_min_length=10
        #is_repline
        self.arc_repline_min_length=10
        self.arc_repline_max_length=200
        self.line_repline_min_length=10
        self.line_repline_max_length=130

        #filter cornor_hole
        self.cornor_hole_total_length=10
        self.cornor_hole_average_length=10

        #remove reference line
        self.reference_line_min_length=30
        self.reference_min_angle=30
        self.reference_max_angle=175
        self.reference_text_max_distance=350

        #dfs
        self.dfs_optional=False

        #constraint determine--parallel
        self.parallel_max_distance= 16
        self.parallel_min_distance=5

        

        self.parallel_max_distance_relax=55
        self.parallel_min_distance_relax=5
        self.contraint_factor=0.98
        self.free_edge_min_length=130
        self.constraint_min_length=52
        self.toe_length=26
        #bracket bounding box
        self.bracket_bbox_expand_length=450
        self.bracket_bbox_expand_ratio=0.25
        self.bracket_bbox_expand_is_ratio=False


        #is_parallel_tolerance
        #筛选角隅孔时的平行判断
        self.is_parallel_tolerance_neighobor=0.2
        #固定边平行判断
        self.is_parallel_tolerance=0.15
        #readJson
        self.line_type=["BYLAYER","CONTINUOUS","BYBLOCK"]
        self.color=[6,8]
        self.constraint_color=[1,3]
        self.element_type=["line","arc","lwpolyline","polyline","spline"]

        self.stiffener_name = ["Stiffener_Invisible","Stiffener_Visible"]
        # self.remove_layername=["Stiffener_Invisible","Stiffener_Visible","Plate_Invisible","Plate_Visible"]
        self.remove_linetype=[]
        self.remove_layername=["Seam","100-人孔盖","人孔盖", "开孔识别结果", "可识别贯穿孔","肘板标注","Braket"]
        self.remove_layername.extend(self.stiffener_name)
        #check is bracket
        #偏离凸多边形的程度
        self.near_convex_tolerance=0.05
        self.near_rectangle_tolerance = 0.0002
        #自由边角度下限
        self.min_angle_in_free_edge=45
        #自由边占比
        self.free_edge_ratio=0.15
        
        #约束边分割角度
        self.constraint_split_angle=30

        #加速结构
        self.min_cell_length=500
        self.min_segments_num=50
        self.M=2
        self.N=2
        self.x_padding=100
        self.y_padding=100


        #开孔所在图层
        self.hole_layer = "开孔标注"

        #剖图文件路径
        self.multi_json_path = None
        
