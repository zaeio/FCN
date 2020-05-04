from cv2 import cv2
from PIL import Image
import numpy as np
import os
import glob
import config as cfg
np.set_printoptions(threshold=np.inf)
# import ImageProcessLib as imgLib

def get_filelist(path):
    """遍历文件和子文件夹"""
    Filelist = []
    Dirlist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            # 文件名列表，包含完整路径
            Filelist.append(os.path.join(home, filename))
            # # 文件名列表，只包含文件名
            # Filelist.append( filename)
        for dir in dirs:
            Dirlist.append(os.path.join(home, dir))
    return Filelist, Dirlist

def generate_tfrecord(dirlist):
    for dir in dirlist:
        for index,png in enumerate(glob.glob(dir + '\*.png')):
            if index == 0:
                srcIMG = Image.open(png)
                srcIMG.show()
                print('srcIMG')
            if index == 1:
                labelIMG = Image.open(png)
                labelIMG.show()
                print('labelIMG')
                smallIMG = labelIMG.resize((cfg.IMG_W, cfg.IMG_H))
                lable_np = np.array(smallIMG)
                print(lable_np)
                

if __name__ =="__main__":
    path = r'E:\BK\Program\TensorFlow\samples\stick samples\segment\labelme_json'
    Filelist, Dirlist = get_filelist(path)
    generate_tfrecord(Dirlist)
