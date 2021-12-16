import json
import time
import os.path
import requests

with open('./assets_text.json', 'r') as file:
    data = file.read()
    y = json.loads(data)
    for key in y.keys():
        print("key = " + key)
        fname_diff = key + "_diff_1k.jpg"
        fname_disp = key + "_disp_1k.jpg"
        fname_norm = key + "_nor_gl_1k.jpg"
        fname_roug = key + "_rough_1k.jpg"
        oname_base = "./texts/" + key
        oname_diff = oname_base + "/" + key + "_diff_1k.jpg"
        oname_disp = oname_base + "/" + key + "_disp_1k.jpg"
        oname_norm = oname_base + "/" + key + "_norm_1k.jpg"
        oname_roug = oname_base + "/" + key + "_roug_1k.jpg"
        url_diff = "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/" + key + "/" + fname_diff
        url_disp = "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/" + key + "/" + fname_disp
        url_norm = "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/" + key + "/" + fname_norm
        url_roug = "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/" + key + "/" + fname_roug
        print("url_diff = " + url_diff)
        print("url_disp = " + url_disp)
        print("url_norm = " + url_norm)
        print("url_roug = " + url_roug)
        # https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/rocks_ground_05/rocks_ground_05_diff_1k.jpg
        # https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/rocks_ground_05/rocks_ground_05_disp_1k.jpg
        # https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/rocks_ground_05/rocks_ground_05_nor_gl_1k.jpg
        # https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/rocks_ground_05/rocks_ground_05_rough_1k.jpg
        if os.path.isfile(oname_roug):
            print("exists, skipping...")
        else:
            res_diff = requests.get(url_diff)
            res_disp = requests.get(url_disp)
            res_norm = requests.get(url_norm)
            res_roug = requests.get(url_roug)
            if res_diff.status_code == 200 and \
               res_disp.status_code == 200 and \
               res_norm.status_code == 200 and \
               res_roug.status_code == 200:
                print("downloaded...")
                os.makedirs(oname_base)
                with open(oname_diff, 'wb') as outfile:
                    outfile.write(res_diff.content)
                with open(oname_disp, 'wb') as outfile:
                    outfile.write(res_disp.content)
                with open(oname_norm, 'wb') as outfile:
                    outfile.write(res_norm.content)
                with open(oname_roug, 'wb') as outfile:
                    outfile.write(res_roug.content)
                print("written...")
            else:
                print("download failed")
            time.sleep(1)
