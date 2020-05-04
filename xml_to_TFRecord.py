import os 
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import glob  # 用于遍历文件夹内的xml文件
import xml.etree.ElementTree as ET  # 用于解析xml文件
from cv2 import cv2
from config import *

writer = tf.python_io.TFRecordWriter('stickobject.tfrecords') #输出成tfrecord文件
# 遍历文件夹内的全部xml文件，1个xml文件描述1个图像文件的标注信息
for index,f in enumerate(glob.glob(r"E:\BK\Program\TensorFlow\samples\stick samples\xml\*.xml")):
    # 解析xml文件
    try:
        tree = ET.parse(f)
    except FileNotFoundError:
        print("无法找到xml文件: "+f)
    except ET.ParseError:
        print("无法解析xml文件: "+f)
    else:  # ET.parse()运行正确
        # 取得xml根节点
        root = tree.getroot()

        # 取得图像路径和文件名
        # print(root.find("filename").text)
        img_path = root.find("path").text
        # print(image_path)
        # srcIMG = cv2.imread(image_path)
        
        # 取得图像宽高
        img_width = int(root.find("size")[0].text)
        img_height = int(root.find("size")[1].text)
        # print("size: [{}, {}]".format(img_width, img_height))  # width节点的序号为[0], height节点的序号为[1]

        # 取得bbox
        # 查找根节点下全部名为object的节点, 每个object包含一个bndbox和一个name
        stick_box_n = [0,0,0,0]
        bottom_box_n = [0,0,0,0]
        for obj in root.findall("object"):
            for name in obj.iter('name'):
                xmin = int(obj[4][0].text)
                ymin = int(obj[4][1].text)
                xmax = int(obj[4][2].text)
                ymax = int(obj[4][3].text)
                
                if obj.find("name").text == 'stick':
                    stick_box_n[0] = ((xmin + xmax)/2.) / float(SRC_IMG_W) # xc
                    stick_box_n[1] = ((ymin + ymax)/2.) / float(SRC_IMG_H) # yc
                    stick_box_n[2] = (xmax - xmin) / float(SRC_IMG_W) # w
                    stick_box_n[3] = (ymax - ymin) / float(SRC_IMG_H) # h
                elif obj.find("name").text =='bottom':
                    
                    bottom_box_n[0] = ((xmin + xmax)/2.) / float(SRC_IMG_W) # xc
                    bottom_box_n[1] = ((ymin + ymax)/2.) / float(SRC_IMG_H) # yc
                    bottom_box_n[2] = (xmax - xmin) / float(SRC_IMG_W) # w
                    bottom_box_n[3] = (ymax - ymin) / float(SRC_IMG_H) # h

        srcIMG = Image.open(img_path)
        # grayIMG = srcIMG.convert('L') # 转为灰度
        smallIMG = srcIMG.resize((IMG_W, IMG_H))
        byteIMG = smallIMG.tobytes()  #将图片转化为二进制格式
        example = tf.train.Example(features = tf.train.Features(feature = {
            "img_raw": tf.train.Feature(bytes_list = tf.train.BytesList(value = [byteIMG])),
            "stick_xc": tf.train.Feature(float_list = tf.train.FloatList(value = [stick_box_n[0]])),
            "stick_yc": tf.train.Feature(float_list = tf.train.FloatList(value = [stick_box_n[1]])),
            "stick_w": tf.train.Feature(float_list = tf.train.FloatList(value = [stick_box_n[2]])),
            "stick_h": tf.train.Feature(float_list = tf.train.FloatList(value = [stick_box_n[3]])),
            "bottom_xc": tf.train.Feature(float_list = tf.train.FloatList(value = [bottom_box_n[0]])),
            "bottom_yc": tf.train.Feature(float_list = tf.train.FloatList(value = [bottom_box_n[1]])),
            "bottom_w": tf.train.Feature(float_list = tf.train.FloatList(value = [bottom_box_n[2]])),
            "bottom_h": tf.train.Feature(float_list = tf.train.FloatList(value = [bottom_box_n[3]])),
        }))
        writer.write(example.SerializeToString())  #序列化为字符串
        print('NO.{} done'.format(index))
writer.close()

