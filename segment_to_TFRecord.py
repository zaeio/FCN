from cv2 import cv2
from PIL import Image
import tensorflow as tf
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
    writer = tf.python_io.TFRecordWriter('StickSegment.tfrecords') #输出成tfrecord文件
    for i, dir in enumerate(dirlist):
        srcIMG_byte = []
        labelIMG_byte = []
        for index, png in enumerate(glob.glob(dir + '\*.png')):
            if index == 0:
                srcIMG = Image.open(png)
                src_smallIMG = srcIMG.resize((cfg.IMG_W, cfg.IMG_H))
                # srcNP = np.array(src_smallIMG)
                srcIMG_byte = src_smallIMG.tobytes()
            if index == 1:
                labelIMG = Image.open(png).convert('L')
                label_smallIMG = labelIMG.resize((cfg.IMG_W, cfg.IMG_H))
                # labelNP = np.array(label_smallIMG)
                labelIMG_byte = label_smallIMG.tobytes()

        example = tf.train.Example(features = tf.train.Features(feature = {
            "img_raw": tf.train.Feature(bytes_list = tf.train.BytesList(value = [srcIMG_byte])),
            "img_mask": tf.train.Feature(bytes_list = tf.train.BytesList(value = [labelIMG_byte])),
        }))
        writer.write(example.SerializeToString())  #序列化为字符串
        print('NO.{} done'.format(i))

if __name__ =="__main__":
    path = r'E:\BK\Program\TensorFlow\samples\stick samples\segment\labelme_json'
    Filelist, Dirlist = get_filelist(path)
    generate_tfrecord(Dirlist)