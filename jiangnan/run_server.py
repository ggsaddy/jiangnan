from flask import Flask, request, jsonify, send_file
from werkzeug.exceptions import BadRequest
import zipfile
import threading
import traceback
import json, os
from waitress import serve
import argparse
import logging.handlers

import preprocess.load as load
import preprocess.convert_dwg2dxf as convert_dwg2dxf
import segment_and_multi_detection.segment_all as segment_all
import segment_and_multi_detection.multi_detection as multi_detection
import holes.extract_dimen_test as extract_dimen_test
import holes.main_test as main_test
import bracket.BraketDetection.bracket_detection as bracket_detection

app = Flask(__name__)

@app.route("/dxf2json", methods=['POST'])
def dxf2json():
    data = request.get_json()  # 解析JSON数据
    '''dxfpath = request.args.get("dxfpath")
    dxfname = request.args.get("dxfname")'''
    dxfpath = data["dxfpath"]
    dxfname = data["dxfname"]
    load.dxf2json(dxfpath, dxfname, dxfpath)
    # print(dxfpath,dxfname)
    return "<p>success!</p>"

@app.route("/dwg2dxf", methods=['POST'])
def dwg2dxf():
    data = request.get_json()  # 解析JSON数据
    '''dxfpath = request.args.get("dxfpath")
    dxfname = request.args.get("dxfname")'''
    dwg_path = data["dwg_path"]
    dxf_path = data["dxf_path"]
    convert_dwg2dxf.dwg2dxf(dwg_path, dxf_path)
    # convert_dwg2dxf.dwg2dxf(**data)
    # print(dwg_path, dxf_path)
    return "<p>success!</p>"

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
    return bbox,all_json_data
    # return "<p>success!</p>"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=1180)
