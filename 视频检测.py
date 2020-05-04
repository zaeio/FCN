import tensorflow as tf
import numpy as np
import os
from cv2 import cv2
from PIL import Image
from RPN import RPN
import selectivesearch.selectivesearch as ss
from config import*
import ImageProcessLib as imgLib

rpn = RPN()
ppt = imgLib.PPT()
cap = cv2.VideoCapture(0+ cv2.CAP_DSHOW)

def calibrate(event, x, y, flags, param):
    """取点的顺序为: 左上,右上,左下,右下"""
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(ppt.points1) < 4:
            ppt.points1.append([x, y])

with tf.Session() as sess:
    """读取模型"""
    saver = tf.train.import_meta_graph('./My_Model/CNN_Model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./My_Model'))
    graph = tf.get_default_graph()
    # for tensor_name in tf.contrib.graph_editor.get_tensors(tf.get_default_graph()):
    #     print(tensor_name)
    img_PH = graph.get_tensor_by_name("img_PH:0")
    drop_PH = graph.get_tensor_by_name("drop_PH:0")
    cls_output = graph.get_tensor_by_name("cls_fc2_2/cls_output:0")
    reg_output = graph.get_tensor_by_name("reg_fco_4/reg_output:0")

    """标定"""
    cv2.namedWindow("calibration")
    cv2.setMouseCallback("calibration", calibrate)
    while(len(ppt.points1) < 4):
        ret, frame = cap.read()
        srcIMG = cv2.resize(frame, (SRC_IMG_W, SRC_IMG_H))
        dstIMG = imgLib.draw_points(srcIMG, ppt.points1)
        cv2.imshow("calibration", dstIMG)
        cv2.waitKey(1)
    cv2.destroyWindow('calibration')
    ppt.get_tr_mat() # 计算变换矩阵

    cv2.namedWindow("table")
    ret, frame = cap.read()
    srcIMG = cv2.resize(frame, (SRC_IMG_W, SRC_IMG_H))
    tableIMG = ppt.do_perspective(srcIMG)
    cv2.imshow("table", tableIMG)
    """识别和定位"""
    while(1):
        ret, frame = cap.read()
        bigIMG = cv2.resize(frame, (SRC_IMG_W, SRC_IMG_H))
        # 图像预处理: 交换通道，缩小，扩展维度，归一化
        rgbIMG = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        smallIMG = cv2.resize(rgbIMG, (IMG_W,IMG_H))
        image_4 = smallIMG[np.newaxis,:,:]
        image_n = imgLib.image_normalize(image_4)
        # 前向传播获得预测值
        cls_pres, reg_pres = sess.run([cls_output, reg_output], feed_dict={img_PH:image_n, drop_PH:1})
        chosen_index = np.argmax(cls_pres, axis=0)[1] # 类别最有可能的index

        # if cls_pres[chosen_index][1] > 0.99:
        p_text = str(cls_pres[chosen_index][1])

        # 由预测的偏移量和anchor计算最终的prebox
        pre_box = imgLib.pre_2_prebox(reg_pres[chosen_index], rpn.anchor_list[chosen_index])
        # 基准anchor，用于绘制和调试
        choosen_anchor = rpn.anchor_list[chosen_index]
        anchor_xyxy = imgLib.xywh_2_xyxy(choosen_anchor)
        anchor_box = imgLib.normxy_2_realxy(anchor_xyxy)

        cv2.rectangle(bigIMG, (pre_box[0], pre_box[1]), (pre_box[2], pre_box[3]), (0,0,255), 3)
        cv2.putText(bigIMG, p_text, (pre_box[0]+5,pre_box[1]+25), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1, 8, 0)
        cv2.rectangle(bigIMG, (anchor_box[0], anchor_box[1]), (anchor_box[2], anchor_box[3]), (0,200,0), 2)

        bottom_left_point = [pre_box[0], pre_box[3]]
        p2, real_coord = ppt.get_coordinate(bottom_left_point)
        cube_box = [p2[0], p2[1]-CUBE_IMG_SIZE, p2[0]+CUBE_IMG_SIZE, p2[1]]

        # 在tableIMG的基础上画cube，得到cubeIMG
        cubeIMG = tableIMG.copy()
        cv2.rectangle(cubeIMG, (cube_box[0], cube_box[1]), (cube_box[2], cube_box[3]), (0,0,0), 2)
        center_point = [p2[0] + CUBE_IMG_SIZE/2, p2[1] - CUBE_IMG_SIZE/2] # cube中心点
        center_point = np.array(center_point).astype(np.int32)
        cv2.line(cubeIMG, (center_point[0]-5, center_point[1]), (center_point[0]+5, center_point[1]), (0,0,0), 1) # 画中心十字
        cv2.line(cubeIMG, (center_point[0], center_point[1]-5), (center_point[0], center_point[1]+5), (0,0,0), 1)
        real_p_text = "%d,%d mm" % (real_coord[0] + CUBE_SIZE/2, real_coord[1] + CUBE_SIZE/2) # 中心点世界坐标文本
        cv2.putText(cubeIMG, real_p_text, (cube_box[0], cube_box[1]-5), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 1)
        cv2.imshow("table", cubeIMG)


        cv2.imshow("object", bigIMG)
        cv2.waitKey(1)




    