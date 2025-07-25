'''
1. 官网下载对应平台版本 https://www.opendesign.com/guestfiles/oda_file_converter
2. 如果是linux或者macos 使用 chmod a+x ~/Apps/ODAFileConverter_QT5_lnxX64_8.3dll_23.9.AppImage
3. windows修改win_oda_path,  macos修改mac_oda_path
4. python convert_dwg2dxf.py --dwg_path dwg2dxf.dwg
5. dxf文件在当前目录下
'''
import os 
import argparse
import ezdxf 
from ezdxf.addons import odafc 





def dwg2dxf(dwg_path, dxf_path):



    win_oda_path = r"C:\Program Files\ODA\ODAFileConverter\ODAFileConverter.exe"
    mac_oda_path = "/Applications/ODAFileConverter.app/Contents/MacOS/ODAFileConverter"

    sys_name = os.name 
    if sys_name == "posix":
        ezdxf.options.set("odafc-addon", "unix_exec_path", mac_oda_path)
    elif sys_name == "nt":
        ezdxf.options.set("odafc-addon", "win_exec_path", win_oda_path)
    else:
        raise Exception("Unsupported system")
    


    odafc.convert(dwg_path, dxf_path, version="R2018")




if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dwg_path', type=str, default="dwg2dxf.dwg", help="dwg path")
    # args = parser.parse_args()
    # odafc.convert(args.dwg_path, args.dwg_path.replace(".dwg", ".dxf"), version="R2018")
    
    dwg_path = r"D:\desktop\结构AI建模\test_dwg.dwg"
    dxf_path = r"D:\desktop\结构AI建模\test_dwg.dxf"

    dwg2dxf(dwg_path, dxf_path)