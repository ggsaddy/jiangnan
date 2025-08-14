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
import preprocess.load as load
import preprocess.convert_dwg2dxf as convert_dwg2dxf
import segment_and_multi_detection.segment_all as segment_all
import segment_and_multi_detection.multi_detection as multi_detection
import holes.extract_dimen_test as extract_dimen_test
import holes.main_test as main_test
import bracket.BraketDetection.bracket_detection as bracket_detection
from flask.json.provider import JSONProvider


        
app = Flask(__name__)

UPLOAD_FOLDER = './uploads'  # 上传文件存储目录
OUTPUT_FOLDER = './outputs'  # 输出文件存储目录
DOWNLOAD = '/download'  # 下载链接前缀
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER



def not_dxf(file_name):
    '''判断文件是否为DXF格式'''
    logging.info("not_dxf() 被调用了！")
    return file_name.rsplit('.', 1)[1].lower()!="dxf"

def not_dwg(file_name):
    '''判断文件是否为DWG格式'''
    logging.info("not_dwg() 被调用了！")
    return file_name.rsplit('.', 1)[1].lower()!="dwg"

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
    # 生成文件存储路径
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        
    # 保存文件
    file.save(file_path)

    file_name = os.path.splitext(file.filename)[0]
    '''data = request.get_json()  # 解析JSON数据
    dxfpath = data["dxfpath"]
    dxfname = data["dxfname"]'''
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
    # 生成文件存储路径
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        
    # 保存文件
    file.save(file_path)

    file_name = os.path.splitext(file.filename)[0]
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{file_name}.dxf")
    convert_dwg2dxf.dwg2dxf(file_path, output_path)
    
    return jsonify({
            'status': 'success',
            'converted_file': f'{DOWNLOAD}/{file_name}.dxf',  # 提供下载链接
    }), 200
    '''data = request.get_json()  # 解析JSON数据
    dxfpath = request.args.get("dxfpath")
    dxfname = request.args.get("dxfname")
    dwg_path = data["dwg_path"]
    dxf_path = data["dxf_path"]
    convert_dwg2dxf.dwg2dxf(dwg_path, dxf_path)
    # convert_dwg2dxf.dwg2dxf(**data)
    # print(dwg_path, dxf_path)
    return "<p>success!</p>"'''

@app.route("/segment", methods=['POST'])
def segment():
    data = request.get_json()
    input_file = data["input_file"]
    input_folder = data["input_folder"]
    output_folder = data["output_folder"]
    segment_all.segment_all_main(input_file, input_folder, output_folder)
    return "<p>success!</p>"

@app.route("/detection", methods=['POST'])
def detection():
    data = request.get_json()
    input_file = data["input_file"]
    input_folder = data["input_folder"]
    output_folder = data["output_folder"]
    multi_detection.multi_detection_main(input_file, input_folder, output_folder)
    return "<p>success!</p>"

@app.route("/dimension_test", methods=['POST'])
def extract_dimension():
    data = request.get_json()
    dxf_path = data["dxf_path"]
    print(dxf_path)
    extract_dimen_test.extract_dimen_test(dxf_path)
    return "<p>success!</p>"

@app.route("/main_test", methods=['POST'])
def main():
    data = request.get_json()
    args = main_test.merge_args(data)
    main_test.main_test(args)
    return "<p>success!</p>"

@app.route("/bracket", methods=['POST'])
def bracket():
    data = request.get_json()
    input_path = data["input_path"]
    output_folder = data["output_folder"]
    config_path = data["config_path"]
    bracket_detection.bracket_detection(input_path, output_folder, config_path)
    return "<p>success!</p>"

@app.route("/bracket_add", methods=['POST'])
def bracket_add():
    data = request.get_json()
    input_path = data["input_path"]
    polys_path = data["polys_path"]
    output_folder = data["output_folder"]
    config_path = data["config_path"]
    bracket_detection.bracket_detection_add(input_path, polys_path, output_folder, config_path)
    return "<p>success!</p>"

@app.route("/dwg_bracket", methods=['POST'])
def dwg_bracket():
    data = request.get_json()
    input_path = data["input_path"] # "./data/test.dwg"
    output_path = data["output_path"] # "./data/test.dxf"
    output_folder_1 = data["output_folder_1"] # "./data/test.dxf"
    convert_dwg2dxf.dwg2dxf(input_path, output_path)
    output_folder = os.path.dirname(output_path) # "./data"
    multi_detection.multi_detection_main(output_path, output_folder, output_folder)
    config_path = None
    bbox,all_json_data = bracket_detection.bracket_detection(output_path, output_folder_1, config_path)

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

    '''response_data = {
        "bbox": bbox,
        "all_json_data": all_json_data
    }

    json_str = json.dumps(response_data,default=str, ensure_ascii=False, indent=4)
    return jsonify({
            'len_bbox': len(bbox),
            'len_all_json_data': len(all_json_data),
    }), 200 
    return Response(json_str,
    status=200,
    mimetype='application/json')'''
    # return "<p>success!</p>"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=1180)
