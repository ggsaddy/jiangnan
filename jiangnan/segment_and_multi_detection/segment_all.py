import argparse
import os

import ezdxf.colors
from load import dxf2json
from split.main import segment_v4, multi_detect_local
import ezdxf
import logging
import json
import multiprocessing
from glob import glob
from split.main import segment_v0725
def segment_all_main(input_file,input_folder,output_folder):
    file_name = os.path.basename(input_file)[:-4]
    input_dxf_file = os.path.join(input_folder, file_name + ".dxf")
    dxf2json(input_folder, file_name, output_folder)
    input_json_file = os.path.join(output_folder, file_name + ".json")
    segment_v0725(input_json_file,output_folder)

if __name__ == "__main__":
    input_file = "/disk1/user4/work/造船厂/结构AI/qzr/output/test_0725v6.dxf"
    input_folder = "/disk1/user4/work/造船厂/结构AI/qzr/output"
    output_folder = "/disk1/user4/work/造船厂/结构AI/qzr/output"
    segment_all_main(input_file, input_folder, output_folder)