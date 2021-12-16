import json
import time
import os.path
import requests

with open('./assets_hdri.json', 'r') as file:
    data = file.read()
    y = json.loads(data)
    for key in y.keys():
        print("key = " + key)
        fname = key + "_2k.hdr"
        oname = "./hdris/" + fname
        url = "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/" + fname
        print("url = " + url)
        if os.path.isfile(oname):
            print("exists, skipping...")
        else:
            res = requests.get(url)
            print("downloaded...")
            with open(oname, 'wb') as outfile:
                outfile.write(res.content)
            print("written...")
            time.sleep(1)
