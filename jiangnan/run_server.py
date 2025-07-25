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

app = Flask(__name__)

@app.route("/dxf2json", methods=['POST'])
def dxf2json():
    data = request.get_json()  # 解析JSON数据
    '''dxfpath = request.args.get("dxfpath")
    dxfname = request.args.get("dxfname")'''
    dxfpath = data["dxfpath"]
    dxfname = data["dxfname"]
    load.dxf2json(dxfpath, dxfname, dxfpath)
    print(dxfpath,dxfname)
    return "<p>success!</p>"

@app.route("/dwg2dxf", methods=['POST'])
def dwg2dxf():
    data = request.get_json()  # 解析JSON数据
    '''dxfpath = request.args.get("dxfpath")
    dxfname = request.args.get("dxfname")'''
    dwg_path = data["dwg_path"]
    dxf_path = data["dxf_path"]
    convert_dwg2dxf.dwg2dxf(dwg_path, dxf_path)
    print(dwg_path, dxf_path)
    return "<p>success!</p>"

app.run(host='0.0.0.0', port=1180)
