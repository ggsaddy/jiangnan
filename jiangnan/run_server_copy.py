from flask import Flask, request, jsonify, send_file, send_from_directory, Response
from werkzeug.exceptions import BadRequest
import zipfile
import threading
import traceback
import json, os
from waitress import serve
import argparse
import logging.handlers
import math
from datetime import datetime, timedelta
import psutil
import time
import uuid
from functools import wraps
import inspect
import tempfile
import shutil
from collections import defaultdict
import preprocess.load as load
import preprocess.convert_dwg2dxf as convert_dwg2dxf
import segment_and_multi_detection.segment_all as segment_all
import segment_and_multi_detection.multi_detection as multi_detection
import holes.extract_dimen_test as extract_dimen_test
import holes.main_test as main_test
import bracket.BraketDetection.bracket_detection as bracket_detection
import PanelInfoExtration.main_use as main_use
import multi_json_reader


        
app = Flask(__name__)

UPLOAD_FOLDER = './uploads'  # 上传文件存储目录
OUTPUT_FOLDER = './outputs'  # 输出文件存储目录
DOWNLOAD = '/download'  # 下载链接前缀
MAX_FILE_AGE_DAYS = 30  # 自动删除超过 30 天的文件
MAX_FILES_TO_KEEP = 1000  # 最多保留 1000 个文件，超出则删除最旧的
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['JSON_AS_ASCII'] = False  # 允许非ASCII字符
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True # Flask 返回的 JSON 响应是否进行美化格式化
# 全局限制配置
GLOBAL_MAX_CONCURRENT = 5  # 服务器总的最大并发数
GLOBAL_MAX_QUEUE = 10      # 服务器总的最大排队数

# 全局信号量
global_concurrent_semaphore = threading.Semaphore(GLOBAL_MAX_CONCURRENT)
global_queue_semaphore = threading.Semaphore(GLOBAL_MAX_CONCURRENT + GLOBAL_MAX_QUEUE)

# 全局统计
global_stats = {
    'active': 0,
    'queued': 0,
    'total': 0,
    'rejected': 0
}
global_stats_lock = threading.Lock()

# 端点级别统计
endpoint_stats = defaultdict(lambda: {
    'active': 0,
    'queued': 0,
    'total': 0,
    'rejected': 0
})

def not_dxf(file_name):
    '''判断文件是否为DXF格式'''
    logging.info("not_dxf() 被调用了！")
    return file_name.rsplit('.', 1)[1].lower()!="dxf"

def not_dwg(file_name):
    '''判断文件是否为DWG格式'''
    logging.info("not_dwg() 被调用了！")
    return file_name.rsplit('.', 1)[1].lower()!="dwg"

def not_json(file_name):
    '''判断文件是否为JSON格式'''
    logging.info("not_json() 被调用了！")
    return file_name.rsplit('.', 1)[1].lower()!="json"

def rename_uploaded_file(uploaded_file):
    """重命名上传的文件，避免文件名冲突"""
    # 获取文件名和扩展名
    file_name, file_ext = os.path.splitext(uploaded_file.filename)
    # 生成新的文件名
    new_file_name = f"{uuid.uuid4().hex[:8]}_{int(time.time())}{file_ext}"
    
    return new_file_name

# 临时文件夹管理
class TempFolderManager:
    def __init__(self, base_dir=None, cleanup_delay=300):  # 5分钟后清理
        self.base_dir = base_dir
        self.cleanup_delay = cleanup_delay
        self.pending_cleanup = {}  # {folder_path: cleanup_time}
        self.cleanup_lock = threading.Lock()
        
        # 启动清理线程
        self.cleanup_thread = threading.Thread(target=self.cleanup_worker, daemon=True)
        self.cleanup_thread.start()

    def create_temp_folder(self, service_name=None):
        """创建临时文件夹,并返回路径"""
        folder_name = f"task_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        # 使用服务名称创建专门的工作目录
        service_dir = os.path.join(".", service_name)
        os.makedirs(service_dir, exist_ok=True)
        temp_path = os.path.join(service_dir, "server_temp", folder_name)
        os.makedirs(temp_path, exist_ok=True)
        
        print(f"创建临时文件夹: {temp_path}")
        return temp_path
    
    def schedule_cleanup(self, folder_path):
        """安排延迟清理"""
        cleanup_time = time.time() + self.cleanup_delay
        
        with self.cleanup_lock:
            self.pending_cleanup[folder_path] = cleanup_time
        
        print(f"安排清理: {folder_path} (延迟 {self.cleanup_delay}s)")
    
    def cleanup_worker(self):
        """后台清理工作线程"""
        while True:
            time.sleep(60)  # 每分钟检查一次
            current_time = time.time()
            # 需要清理的文件夹列表
            to_cleanup = []
            with self.cleanup_lock:
                for folder_path, cleanup_time in list(self.pending_cleanup.items()):
                    if current_time >= cleanup_time:
                        to_cleanup.append(folder_path)
                        del self.pending_cleanup[folder_path]
            
            # 执行清理
            for folder_path in to_cleanup:
                self._cleanup_folder(folder_path)
    
    def _cleanup_folder(self, folder_path):
        """清理文件夹"""
        try:
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path, ignore_errors=True)
                print(f"已清理: {folder_path}")
        except Exception as e:
            print(f"清理失败 {folder_path}: {e}")

# 全局管理器实例
temp_manager = TempFolderManager(cleanup_delay=60)

def with_temp_folder(service_name=None):
    """临时文件夹装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            actual_service_name = service_name

            # 创建基于服务名的临时文件夹
            temp_folder = temp_manager.create_temp_folder(service_name=actual_service_name)
            
            try:
                kwargs['temp_folder'] = temp_folder
                result = func(*args, **kwargs)
                return result
            finally:
                temp_manager.schedule_cleanup(temp_folder)
        
        return wrapper
    return decorator

def limit_concurrent_requests(max_concurrent=2, max_queue=5, timeout=300):
    """双重并发限制装饰器：全局限制 + 端点限制"""
    def decorator(func):
        endpoint_name = func.__name__
        
        # 端点级别的信号量
        endpoint_semaphore = threading.Semaphore(max_concurrent)
        endpoint_queue_semaphore = threading.Semaphore(max_concurrent + max_queue)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            with global_stats_lock:
                global_stats['total'] += 1
                
            endpoint_stats[endpoint_name]['total'] += 1
            
            # 第一层检查：全局队列限制
            if not global_queue_semaphore.acquire(blocking=False):
                with global_stats_lock:
                    global_stats['rejected'] += 1
                endpoint_stats[endpoint_name]['rejected'] += 1
                
                return jsonify({
                    'error': '服务器总体负载过高，请稍后重试',
                    'code': 'GLOBAL_TOO_MANY_REQUESTS',
                    'global_active': global_stats['active'],
                    'global_queued': global_stats['queued'],
                    'global_limit': GLOBAL_MAX_CONCURRENT
                }), 503  # 503 Service Unavailable
            
            # 第二层检查：端点队列限制
            if not endpoint_queue_semaphore.acquire(blocking=False):
                endpoint_stats[endpoint_name]['rejected'] += 1
                global_queue_semaphore.release()  # 释放全局队列位置
                
                return jsonify({
                    'error': f'{endpoint_name}端点请求过多，请稍后重试',
                    'code': 'ENDPOINT_TOO_MANY_REQUESTS',
                    'endpoint': endpoint_name,
                    'endpoint_active': endpoint_stats[endpoint_name]['active'],
                    'endpoint_queued': endpoint_stats[endpoint_name]['queued'],
                    'endpoint_limit': max_concurrent
                }), 429  # 429 Too Many Requests
            
            try:
                # 进入全局排队
                with global_stats_lock:
                    global_stats['queued'] += 1
                
                # 进入端点排队
                endpoint_stats[endpoint_name]['queued'] += 1
                
                print(f"[{endpoint_name}] 全局排队: {global_stats['queued']}, "
                      f"端点排队: {endpoint_stats[endpoint_name]['queued']}")
                
                # 获取全局执行权限
                global_acquired = global_concurrent_semaphore.acquire(timeout=timeout)
                
                with global_stats_lock:
                    global_stats['queued'] -= 1
                
                if not global_acquired:
                    endpoint_stats[endpoint_name]['queued'] -= 1
                    endpoint_queue_semaphore.release()
                    global_queue_semaphore.release()
                    
                    return jsonify({
                        'error': '全局处理超时，服务器繁忙',
                        'code': 'GLOBAL_TIMEOUT',
                        'timeout': timeout
                    }), 408
                
                try:
                    # 获取端点执行权限
                    endpoint_acquired = endpoint_semaphore.acquire(timeout=timeout)
                    endpoint_stats[endpoint_name]['queued'] -= 1
                    
                    if not endpoint_acquired:
                        global_concurrent_semaphore.release()
                        endpoint_queue_semaphore.release()
                        global_queue_semaphore.release()
                        
                        return jsonify({
                            'error': f'{endpoint_name}端点处理超时',
                            'code': 'ENDPOINT_TIMEOUT',
                            'timeout': timeout
                        }), 408
                    
                    try:
                        # 开始实际处理
                        with global_stats_lock:
                            global_stats['active'] += 1
                        endpoint_stats[endpoint_name]['active'] += 1
                        
                        print(f"[{endpoint_name}] 开始处理 - "
                              f"全局活跃: {global_stats['active']}/{GLOBAL_MAX_CONCURRENT}, "
                              f"端点活跃: {endpoint_stats[endpoint_name]['active']}/{max_concurrent}")
                        
                        # 执行业务逻辑
                        result = func(*args, **kwargs)
                        
                        print(f"[{endpoint_name}] 处理完成")
                        return result
                        
                    finally:
                        # 清理端点资源
                        endpoint_stats[endpoint_name]['active'] -= 1
                        endpoint_semaphore.release()
                        
                finally:
                    # 清理全局资源
                    with global_stats_lock:
                        global_stats['active'] -= 1
                    global_concurrent_semaphore.release()
                    
            finally:
                # 清理队列资源
                endpoint_queue_semaphore.release()
                global_queue_semaphore.release()
        
        return wrapper
    return decorator

@app.route('/download/<file_name>')
def download_file(file_name):
    """提供文件下载功能"""
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], file_name)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    # return send_file(file_path, as_attachment=True)
    return send_from_directory(app.config['OUTPUT_FOLDER'], file_name, as_attachment=True)

@app.route("/dxf2json", methods=['POST'])
def dxf2json():
    """处理DXF文件上传并转换为JSON格式,并传出下载链接"""
    # 检查请求中是否包含文件
    if 'dxf' not in request.files:
        return jsonify({'error': 'No file in the request'}), 400
    
    file = request.files['dxf']
    # 创建文件目录（如果不存在）
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    # 判断文件传入格式是否正确
    if not_dxf(file.filename):
        app.logger.error(f"Invalid file type: {file.filename}")
        return jsonify({
            'error': 'Invalid file type',
            'message': 'Only DXF files are accepted',
            'received': file.filename
        }), 415 
    # 按时间戳给文件重新命名
    file_name = rename_uploaded_file(file)
    # 生成文件存储路径
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)

    # 保存文件
    file.save(file_path)

    file_name = os.path.splitext(file_name)[0]

    # 调用转换函数
    load.dxf2json(app.config['UPLOAD_FOLDER'], file_name, app.config['OUTPUT_FOLDER'])

    return jsonify({
            'status': 'success',
            'converted_file': f'{DOWNLOAD}/{file_name}.json',  # 提供下载链接
    }), 200

@app.route("/dwg2dxf", methods=['POST'])
def dwg2dxf():
    """处理DWG文件上传并转换为DXF格式,并传出下载链接"""
    # 检查请求中是否包含文件
    if 'dwg' not in request.files:
        return jsonify({'error': 'No file in the request'}), 400

    file = request.files['dwg']
    # 创建文件目录（如果不存在）
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    # 判断文件传入格式是否正确
    if not_dwg(file.filename):
        app.logger.error(f"Invalid file type: {file.filename}")
        return jsonify({
            'error': 'Invalid file type',
            'message': 'Only DWG files are accepted',
            'received': file.filename
        }), 415 
    # 按时间戳给文件重新命名
    file_name = rename_uploaded_file(file)
    # 生成文件存储路径
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)

    # 保存文件
    file.save(file_path)

    file_name = os.path.splitext(file_name)[0]
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{file_name}.dxf")
    convert_dwg2dxf.dwg2dxf(file_path, output_path)
    
    return jsonify({
            'status': 'success',
            'converted_file': f'{DOWNLOAD}/{file_name}.dxf',  # 提供下载链接
    }), 200

@app.route("/dwg2segment", methods=['POST'])
@limit_concurrent_requests(max_concurrent=1,  max_queue=1, timeout=1200)
@with_temp_folder('dwg2segment')
def dwg2segment(temp_folder):
    '''处理DWG文件上传并转换为DXF格式,并进行曲面分割处理,输出json流文件'''
    # 检查请求中是否包含文件
    if 'dwg' not in request.files:
        return jsonify({'error': 'No file in the request'}), 400

    file = request.files['dwg']
    # 判断文件传入格式是否正确
    if not_dwg(file.filename):
        app.logger.error(f"Invalid file type: {file.filename}")
        return jsonify({
            'error': 'Invalid file type',
            'message': 'Only DWG files are accepted',
            'received': file.filename
        }), 415 
    # 按时间戳给文件重新命名
    file_name = rename_uploaded_file(file)
    # 生成文件存储路径
    file_path = os.path.join(temp_folder, file_name)
    # 保存文件
    file.save(file_path)

    file_name = os.path.splitext(file_name)[0]
    output_path = os.path.join(temp_folder, f"{file_name}.dxf")
    convert_dwg2dxf.dwg2dxf(file_path, output_path)
    output_folder = os.path.dirname(output_path) 
    result_mergerd_bbox = segment_all.segment_all_main(output_path, output_folder, output_folder)

    # 设置分页参数（默认每页1个肘板）
    per_page = 1
    total_pages = math.ceil(len(result_mergerd_bbox) / per_page)
    
    def generate():
        """生成器函数，用于分页流式输出"""
        # 分页流式输出
        for page in range(total_pages):
            start_idx = page * per_page
            end_idx = start_idx + per_page
            # current_page_data = processed_data[start_idx:end_idx]
            current_page_data = []
            for i in range(start_idx, end_idx):
                if i < len(result_mergerd_bbox):
                    current_page_data.append(result_mergerd_bbox[i])
            
            yield json.dumps({
                "code": 200,
                "message": "success",
                "data": {
                    "items": current_page_data,
                    "pagination": {
                        "current_page": page + 1,
                        "page_size": per_page,
                        "total_items": len(result_mergerd_bbox),
                        "total_pages": total_pages
                    }
                }
            },
            default=str, 
            ensure_ascii=False, 
            indent=4) + "\n"

    return Response(
        generate(),
        mimetype='application/json-stream',
        headers={
            'Content-Type': 'application/json-stream; charset=utf-8',
            "X-Accel-Buffering": "no",  # 禁用Nginx缓冲
            "Cache-Control": "no-cache"
        }
    )

@app.route("/dwg2detection", methods=['POST'])
@limit_concurrent_requests(max_concurrent=1,  max_queue=1, timeout=1200)
@with_temp_folder('dwg2detection')
def dwg2detection(temp_folder):
    '''处理DWG文件上传并转换为DXF格式,并进行多级剖图处理,输出json流文件'''
    # 检查请求中是否包含文件
    if 'dwg' not in request.files:
        return jsonify({'error': 'No file in the request'}), 400

    file = request.files['dwg']
    # 判断文件传入格式是否正确
    if not_dwg(file.filename):
        app.logger.error(f"Invalid file type: {file.filename}")
        return jsonify({
            'error': 'Invalid file type',
            'message': 'Only DWG files are accepted',
            'received': file.filename
        }), 415 
    # 按时间戳给文件重新命名
    file_name = rename_uploaded_file(file)
    # 生成文件存储路径
    file_path = os.path.join(temp_folder, file_name)
    # 保存文件
    file.save(file_path)

    file_name = os.path.splitext(file_name)[0]
    output_path = os.path.join(temp_folder, f"{file_name}.dxf")
    convert_dwg2dxf.dwg2dxf(file_path, output_path)
    output_folder = os.path.dirname(output_path)
    multi_detection_result_list = multi_detection.multi_detection_main(output_path, output_folder, output_folder)
    
    # 设置分页参数（默认每页1个肘板）
    per_page = 1
    total_pages = math.ceil(len(multi_detection_result_list) / per_page)
    
    def generate():
        """生成器函数，用于分页流式输出"""
        # 分页流式输出
        for page in range(total_pages):
            start_idx = page * per_page
            end_idx = start_idx + per_page
            # current_page_data = processed_data[start_idx:end_idx]
            current_page_data = []
            for i in range(start_idx, end_idx):
                if i < len(multi_detection_result_list):
                    current_page_data.append(multi_detection_result_list[i])
            
            yield json.dumps({
                "code": 200,
                "message": "success",
                "data": {
                    "items": current_page_data,
                    "pagination": {
                        "current_page": page + 1,
                        "page_size": per_page,
                        "total_items": len(multi_detection_result_list),
                        "total_pages": total_pages
                    }
                }
            },
            default=str, 
            ensure_ascii=False, 
            indent=4) + "\n"

    return Response(
        generate(),
        mimetype='application/json-stream',
        content_type='application/json; charset=utf-8',  # 确保返回 UTF-8 编码的 JSON
        headers={
            "X-Accel-Buffering": "no",  # 禁用Nginx缓冲
            "Cache-Control": "no-cache"
        }
    )

@app.route("/dimension_test", methods=['POST'])
@limit_concurrent_requests(max_concurrent=1,  max_queue=1, timeout=1200)
@with_temp_folder('extract_dimension')
def extract_dimension(temp_folder):
    '''处理DWG文件上传并转换为DXF格式,并进行孔洞尺寸标注信息提取处理,输出json流文件'''
    # 检查请求中是否包含文件
    if 'dwg' not in request.files:
        return jsonify({'error': 'No file in the request'}), 400

    file = request.files['dwg']
    # 判断文件传入格式是否正确
    if not_dwg(file.filename):
        app.logger.error(f"Invalid file type: {file.filename}")
        return jsonify({
            'error': 'Invalid file type',
            'message': 'Only DWG files are accepted',
            'received': file.filename
        }), 415 
    # 按时间戳给文件重新命名
    file_name = rename_uploaded_file(file)
    # 生成文件存储路径
    file_path = os.path.join(temp_folder, file_name)
    # 保存文件
    file.save(file_path)

    file_name = os.path.splitext(file_name)[0]
    output_path = os.path.join(temp_folder, f"{file_name}.dxf")
    convert_dwg2dxf.dwg2dxf(file_path, output_path)
    results = extract_dimen_test.extract_dimen_test(output_path)
    
    # 设置分页参数（默认每页1个肘板）
    per_page = 1
    total_pages = math.ceil(len(results) / per_page)
    
    def generate():
        """生成器函数，用于分页流式输出"""
        # 分页流式输出
        for page in range(total_pages):
            start_idx = page * per_page
            end_idx = start_idx + per_page
            # current_page_data = processed_data[start_idx:end_idx]
            current_page_data = []
            for i in range(start_idx, end_idx):
                if i < len(results) and i < len(results):
                    current_page_data.append(results[i])
            
            yield json.dumps({
                "code": 200,
                "message": "success",
                "data": {
                    "items": current_page_data,
                    "pagination": {
                        "current_page": page + 1,
                        "page_size": per_page,
                        "total_items": len(results),
                        "total_pages": total_pages
                    }
                }
            },
            default=str, 
            ensure_ascii=False, 
            indent=4) + "\n"

    return Response(
        generate(),
        mimetype='application/json-stream',
        headers={
            "X-Accel-Buffering": "no",  # 禁用Nginx缓冲
            "Cache-Control": "no-cache"
        }
    )

@app.route("/main_test", methods=['POST'])
@limit_concurrent_requests(max_concurrent=1,  max_queue=1, timeout=1200)
@with_temp_folder('main')
def main(temp_folder):
    '''处理DWG文件上传并转换为DXF格式,并进行孔洞尺寸标注信息提取处理,输出json流文件'''
    # 检查请求中是否包含文件
    if 'dwg' not in request.files:
        return jsonify({'error': 'No file in the request'}), 400

    file = request.files['dwg']
    # 判断文件传入格式是否正确
    if not_dwg(file.filename):
        app.logger.error(f"Invalid file type: {file.filename}")
        return jsonify({
            'error': 'Invalid file type',
            'message': 'Only DWG files are accepted',
            'received': file.filename
        }), 415 
    # 按时间戳给文件重新命名
    file_name = rename_uploaded_file(file)
    # 生成文件存储路径
    file_path = os.path.join(temp_folder, file_name)
    # 保存文件
    file.save(file_path)

    file_name = os.path.splitext(file_name)[0]
    output_path = os.path.join(temp_folder, f"{file_name}.dxf")

    convert_dwg2dxf.dwg2dxf(file_path, output_path)
    data = {
        "dxfpath": temp_folder,
        "dxfname": f"{file_name}.dxf",
        "json_path": f"{file_name}.json"
    }
    args = main_test.merge_args(data)
    results = main_test.main_test(args)

    # 设置分页参数（默认每页1个肘板）
    per_page = 1
    total_pages = math.ceil(len(results) / per_page)
    
    def generate():
        """生成器函数，用于分页流式输出"""
        # 分页流式输出
        for page in range(total_pages):
            start_idx = page * per_page
            end_idx = start_idx + per_page
            # current_page_data = processed_data[start_idx:end_idx]
            current_page_data = []
            for i in range(start_idx, end_idx):
                if i < len(results) and i < len(results):
                    current_page_data.append(results[i])
            
            yield json.dumps({
                "code": 200,
                "message": "success",
                "data": {
                    "items": current_page_data,
                    "pagination": {
                        "current_page": page + 1,
                        "page_size": per_page,
                        "total_items": len(results),
                        "total_pages": total_pages
                    }
                }
            },
            default=str, 
            ensure_ascii=False, 
            indent=4) + "\n"

    return Response(
        generate(),
        mimetype='application/json-stream',
        content_type='application/json; charset=utf-8',  # 确保返回 UTF-8 编码的 JSON
        headers={
            "X-Accel-Buffering": "no",  # 禁用Nginx缓冲
            "Cache-Control": "no-cache"
        }
    )

@app.route("/dwg_bracket_withmutijson", methods=['POST'])
@limit_concurrent_requests(max_concurrent=1,  max_queue=1, timeout=1200)
@with_temp_folder('dwg_bracket_withmutijson')
def dwg_bracket_withmutijson(temp_folder):
    '''处理DWG文件上传并转换为DXF格式并接收外部传入的多级剖图json文件,进行肘板检测,输出json流文件'''
    # 检查请求中是否包含文件
    if 'dwg' not in request.files or 'multi_json' not in request.files:
        return jsonify({'error': 'No file in the request'}), 400

    file = request.files['dwg']
    multi_json = request.files['multi_json']
    # 创建文件目录（如果不存在）
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    # 判断文件传入格式是否正确
    if not_dwg(file.filename):
        app.logger.error(f"Invalid file type: {file.filename}")
        return jsonify({
            'error': 'Invalid file type',
            'message': 'Only DWG files are accepted',
            'received': file.filename
        }), 415 
    if not_json(multi_json.filename):
        app.logger.error(f"Invalid file type: {multi_json.filename}")
        return jsonify({
            'error': 'Invalid file type',
            'message': 'Only JSON files are accepted',
            'received': multi_json.filename
        }), 415
    # 按时间戳给文件重新命名
    file_name = rename_uploaded_file(file)
    multi_json_name = rename_uploaded_file(multi_json)
    # 生成文件存储路径
    file_path = os.path.join(temp_folder, file_name)
    multi_json_path = os.path.join(temp_folder, multi_json_name)
    # 保存文件
    file.save(file_path)
    multi_json.save(multi_json_path)
    # 对传入的multi_json文件进行格式复原
    # multi_json_reader.multi_json_reader(multi_json_path)

    file_name = os.path.splitext(file_name)[0]
    output_path = os.path.join(temp_folder, f"{file_name}.dxf")
    convert_dwg2dxf.dwg2dxf(file_path, output_path)
    
    output_folder = os.path.join(temp_folder, 'dwg_output')  # 假设输出目录为'temp_folder/dwg_output'
    config_path = None
    bbox,all_json_data = bracket_detection.bracket_detection_withmutijson(file_path, output_folder, multi_json_path, config_path)
    # 设置分页参数（默认每页1个肘板）
    per_page = 1
    total_pages = math.ceil(len(all_json_data) / per_page)
    
    def generate():
        """生成器函数，用于分页流式输出"""
        # 分页流式输出
        for page in range(total_pages):
            start_idx = page * per_page
            end_idx = start_idx + per_page
            # current_page_data = processed_data[start_idx:end_idx]
            current_page_data = []
            for i in range(start_idx, end_idx):
                if i < len(all_json_data) and i < len(bbox):
                    current_page_data.append(all_json_data[i])
            
            yield json.dumps({
                "code": 200,
                "message": "success",
                "data": {
                    "items": current_page_data,
                    "pagination": {
                        "current_page": page + 1,
                        "page_size": per_page,
                        "total_items": len(all_json_data),
                        "total_pages": total_pages
                    }
                }
            },
            default=str, 
            ensure_ascii=False, 
            indent=4) + "\n"

    return Response(
        generate(),
        mimetype='application/json-stream',
        headers={
            "X-Accel-Buffering": "no",  # 禁用Nginx缓冲
            "Cache-Control": "no-cache"
        }
    )

@app.route("/dwg_bracket_add_withmutijson", methods=['POST'])
@limit_concurrent_requests(max_concurrent=1,  max_queue=1, timeout=1200)
@with_temp_folder('dwg_bracket_add_withmutijson')
def dwg_bracket_add_withmutijson(temp_folder):
    '''处理DWG文件上传并转换为DXF格式并接收外部传入的多级剖图json文件,进行肘板检测,输出json流文件'''
    # 检查请求中是否包含文件
    if 'dwg' not in request.files or 'multi_json' not in request.files:
        return jsonify({'error': 'No file in the request'}), 400

    file = request.files['dwg']
    multi_json = request.files['multi_json']
    # 判断文件传入格式是否正确
    if not_dwg(file.filename):
        app.logger.error(f"Invalid file type: {file.filename}")
        return jsonify({
            'error': 'Invalid file type',
            'message': 'Only DWG files are accepted',
            'received': file.filename
        }), 415 
    if not_json(multi_json.filename):
        app.logger.error(f"Invalid file type: {multi_json.filename}")
        return jsonify({
            'error': 'Invalid file type',
            'message': 'Only JSON files are accepted',
            'received': multi_json.filename
        }), 415
    # 按时间戳给文件重新命名
    file_name = rename_uploaded_file(file)
    multi_json_name = rename_uploaded_file(multi_json)
    # 生成文件存储路径
    file_path = os.path.join(temp_folder, file_name)
    multi_json_path = os.path.join(temp_folder, multi_json_name)
    # 保存文件
    file.save(file_path)
    multi_json.save(multi_json_path)
    # 对传入的multi_json文件进行格式复原
    multi_json_reader.multi_json_reader(multi_json_path)

    file_name = os.path.splitext(file_name)[0]
    output_path = os.path.join(temp_folder, f"{file_name}.dxf")
    convert_dwg2dxf.dwg2dxf(file_path, output_path)
    
    output_folder = 'dwg_output'  # 假设输出目录为'dwg_output'
    config_path = None
    bbox,all_json_data = bracket_detection.bracket_detection_add_withmutijson(file_path, output_folder, multi_json_path, config_path)

    # 设置分页参数（默认每页1个肘板）
    per_page = 1
    total_pages = math.ceil(len(all_json_data) / per_page)
    
    def generate():
        """生成器函数，用于分页流式输出"""
        # 分页流式输出
        for page in range(total_pages):
            start_idx = page * per_page
            end_idx = start_idx + per_page
            # current_page_data = processed_data[start_idx:end_idx]
            current_page_data = []
            for i in range(start_idx, end_idx):
                if i < len(all_json_data) and i < len(bbox):
                    current_page_data.append(all_json_data[i])
            
            yield json.dumps({
                "code": 200,
                "message": "success",
                "data": {
                    "items": current_page_data,
                    "pagination": {
                        "current_page": page + 1,
                        "page_size": per_page,
                        "total_items": len(all_json_data),
                        "total_pages": total_pages
                    }
                }
            },
            default=str, 
            ensure_ascii=False, 
            indent=4) + "\n"

    return Response(
        generate(),
        mimetype='application/json-stream',
        headers={
            "X-Accel-Buffering": "no",  # 禁用Nginx缓冲
            "Cache-Control": "no-cache"
        }
    )

@app.route("/dwg_bracket_inbbox_withmutijson", methods=['POST'])
@limit_concurrent_requests(max_concurrent=1,  max_queue=1, timeout=1200)
@with_temp_folder('dwg_bracket_inbbox_withmutijson')
def dwg_bracket_inbbox_withmutijson(temp_folder):
    '''处理DWG文件上传并转换为DXF格式并接收外部传入的多级剖图json文件,进行肘板检测,输出json流文件'''
    # 检查请求中是否包含文件
    if 'dwg' not in request.files or 'multi_json' not in request.files or 'bbox' not in request.form:
        return jsonify({'error': 'No file in the request'}), 400

    file = request.files['dwg']
    multi_json = request.files['multi_json']
    bbox = request.form['bbox']
    # 判断文件传入格式是否正确
    if not_dwg(file.filename):
        app.logger.error(f"Invalid file type: {file.filename}")
        return jsonify({
            'error': 'Invalid file type',
            'message': 'Only DWG files are accepted',
            'received': file.filename
        }), 415 
    if not_json(multi_json.filename):
        app.logger.error(f"Invalid file type: {multi_json.filename}")
        return jsonify({
            'error': 'Invalid file type',
            'message': 'Only JSON files are accepted',
            'received': multi_json.filename
        }), 415
    # 按时间戳给文件重新命名
    file_name = rename_uploaded_file(file)
    multi_json_name = rename_uploaded_file(multi_json)
    # 生成文件存储路径
    file_path = os.path.join(temp_folder, file_name)
    multi_json_path = os.path.join(temp_folder, multi_json_name)
    # 保存文件
    file.save(file_path)
    multi_json.save(multi_json_path)
    # 对传入的multi_json文件进行格式复原
    multi_json_reader.multi_json_reader(multi_json_path)
    
    file_name = os.path.splitext(file_name)[0]
    output_path = os.path.join(temp_folder, f"{file_name}.dxf")
    convert_dwg2dxf.dwg2dxf(file_path, output_path)
    
    output_folder = 'dwg_output'  # 假设输出目录为'dwg_output'
    config_path = None
    bbox,all_json_data = bracket_detection.bracket_detection_inbbox_withmutijson(file_path, output_folder, bbox, multi_json_path, config_path)

    # 设置分页参数（默认每页1个肘板）
    per_page = 1
    total_pages = math.ceil(len(all_json_data) / per_page)
    
    def generate():
        """生成器函数，用于分页流式输出"""
        # 分页流式输出
        for page in range(total_pages):
            start_idx = page * per_page
            end_idx = start_idx + per_page
            # current_page_data = processed_data[start_idx:end_idx]
            current_page_data = []
            for i in range(start_idx, end_idx):
                if i < len(all_json_data) and i < len(bbox):
                    current_page_data.append({
                        "bbox": bbox[i],
                        "data": all_json_data[i]
                    })
            
            yield json.dumps({
                "code": 200,
                "message": "success",
                "data": {
                    "items": current_page_data,
                    "pagination": {
                        "current_page": page + 1,
                        "page_size": per_page,
                        "total_items": len(all_json_data),
                        "total_pages": total_pages
                    }
                }
            },
            default=str, 
            ensure_ascii=False, 
            indent=4) + "\n"

    return Response(
        generate(),
        mimetype='application/json-stream',
        headers={
            "X-Accel-Buffering": "no",  # 禁用Nginx缓冲
            "Cache-Control": "no-cache"
        }
    )

@app.route("/panel_info", methods=['POST'])
@limit_concurrent_requests(max_concurrent=1,  max_queue=1, timeout=1200)
@with_temp_folder('panel_info')
def panel_info(temp_folder):
    '''处理DWG文件上传并转换为DXF格式并接收外部传入的excel文件,输出json流文件'''
    # 检查请求中是否包含文件
    if 'dwg' not in request.files or 'excel_1' not in request.files or 'excel_2' not in request.files:
        return jsonify({'error': 'No file in the request'}), 400

    file = request.files['dwg']
    excel_1 = request.files['excel_1']
    excel_2 = request.files['excel_2']
    # 判断文件传入格式是否正确
    if not_dwg(file.filename):
        app.logger.error(f"Invalid file type: {file.filename}")
        return jsonify({
            'error': 'Invalid file type',
            'message': 'Only DWG files are accepted',
            'received': file.filename
        }), 415 
    # 按时间戳给文件重新命名
    file_name = rename_uploaded_file(file)
    excel_1_name = rename_uploaded_file(excel_1)
    excel_2_name = rename_uploaded_file(excel_2)
    # 生成文件存储路径
    file_path = os.path.join(temp_folder, file_name)
    excel_1_path = os.path.join(temp_folder, excel_1_name)
    excel_2_path = os.path.join(temp_folder, excel_2_name)
    # 保存文件
    file.save(file_path)
    excel_1.save(excel_1_path)
    excel_2.save(excel_2_path)
    
    file_name = os.path.splitext(file_name)[0]
    output_path = os.path.join(temp_folder, f"{file_name}.dxf")
    convert_dwg2dxf.dwg2dxf(file_path, output_path)

    jsonlist = main_use.main(temp_folder, os.path.basename(output_path), excel_1_name, excel_2_name)

    # 设置分页参数（默认每页1个肘板）
    per_page = 1
    total_pages = math.ceil(len(jsonlist) / per_page)

    def generate():
        """生成器函数，用于分页流式输出"""
        # 分页流式输出
        for page in range(total_pages):
            start_idx = page * per_page
            end_idx = start_idx + per_page
            # current_page_data = processed_data[start_idx:end_idx]
            current_page_data = []
            for i in range(start_idx, end_idx):
                if i < len(jsonlist):
                    current_page_data.append(jsonlist[i])

            yield json.dumps({
                "code": 200,
                "message": "success",
                "data": {
                    "items": current_page_data,
                    "pagination": {
                        "current_page": page + 1,
                        "page_size": per_page,
                        "total_items": len(jsonlist),
                        "total_pages": total_pages
                    }
                }
            },
            default=str, 
            ensure_ascii=False, 
            indent=4) + "\n"

    return Response(
        generate(),
        mimetype='application/json-stream',
        headers={
            "X-Accel-Buffering": "no",  # 禁用Nginx缓冲
            "Cache-Control": "no-cache"
        }
    )

if __name__ == "__main__":
    # 使用Waitress多线程服务器
    # app.run(host='0.0.0.0', port=1180, threaded=False)
    serve(
        app,
        host='0.0.0.0',
        port=1180,
        threads=10,               # 设置线程池大小为10
        connection_limit=1000,    # 最大连接数
        channel_timeout=600,      # 通道超时600秒
        asyncore_use_poll=True,   # 使用更高效的poll机制
        ident="server"        # 服务标识
    )
