import tensorflow as tf
import numpy as np
from cv2 import cv2
import network as net
import config as cfg
import ImageProcessLib as imgLib

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./My_Model/FCN_Model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./My_Model'))
    graph = tf.get_default_graph()
    # for tensor_name in tf.contrib.graph_editor.get_tensors(tf.get_default_graph()):
    #     print(tensor_name)
    img_PH = graph.get_tensor_by_name("img_PH:0")
    batch_PH = graph.get_tensor_by_name("batch_PH:0")
    drop_PH = graph.get_tensor_by_name("drop_PH:0")
    output = graph.get_tensor_by_name("output:0")

    srcIMG = cv2.imread("./testimg/test (1).jpg")
    rgbIMG = cv2.cvtColor(srcIMG, cv2.COLOR_BGR2RGB)
    smallIMG = cv2.resize(rgbIMG, (cfg.IMG_W,cfg.IMG_H)) # 网络输入图像大小224*160
    image_4 = smallIMG[np.newaxis,:,:] # 扩展维度
    image_n = imgLib.image_normalize(image_4) # 标准化
    
    prediction = sess.run(output, feed_dict={img_PH:image_n, batch_PH:1, drop_PH:1})
    labelMAP4 = np.argmax(prediction, axis=3).astype(np.uint8)
    labelMAP2 = np.reshape(labelMAP4, (cfg.IMG_H, cfg.IMG_W))

    # 转cv2 mat，二值化
    _,label_binIMG = cv2.threshold(labelMAP2,0.5,255,cv2.THRESH_BINARY)
    # 添加2个全0通道，制作红色蒙版
    zero_channel = np.zeros(label_binIMG.shape, dtype = "uint8")
    mask_IMG = cv2.merge([zero_channel, label_binIMG, zero_channel])
    big_mask_IMG = cv2.resize(mask_IMG, (cfg.SRC_IMG_W,cfg.SRC_IMG_H))
    # 叠加
    overlappingIMG = cv2.addWeighted(srcIMG, 0.7, big_mask_IMG, 0.3, 0)
    cv2.imshow("object", overlappingIMG)
    cv2.waitKey(0)


    
