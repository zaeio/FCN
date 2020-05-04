import os 
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2
import config as cfg


def segment_tfrecord_decode(filename, batch_size):
    filename_queue = tf.train.string_input_producer([filename])# create a queue

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#return file_name and file
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                           'img_mask': tf.FixedLenFeature([], tf.string),
                                       })#return image and label

    srcIMG = tf.decode_raw(features['img_raw'], tf.uint8)
    srcIMG = tf.reshape(srcIMG, [cfg.IMG_H, cfg.IMG_W, 3])
    maskIMG = tf.decode_raw(features['img_mask'], tf.uint8)
    maskIMG = tf.reshape(maskIMG, [cfg.IMG_H, cfg.IMG_W])

    img_batch, mask_batch = tf.train.shuffle_batch([srcIMG, maskIMG],   batch_size= batch_size, 
                                                                        num_threads=64,
                                                                        capacity=2000,
                                                                        min_after_dequeue=1500,
                                                                        )
    
    return img_batch, mask_batch

def cube_tfrecord_decode(filename, batch_size): # read train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])# create a queue

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#return file_name and file
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                           'xc_n': tf.FixedLenFeature([], tf.float32),
                                           'yc_n': tf.FixedLenFeature([], tf.float32),
                                           'w_n': tf.FixedLenFeature([], tf.float32),
                                           'h_n': tf.FixedLenFeature([], tf.float32),
                                       })#return image and label

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [cfg.IMG_H, cfg.IMG_W, 3])  #reshape image to 208*208*3
    xc = tf.cast(features['xc_n'], tf.float32)
    yc = tf.cast(features['yc_n'], tf.float32)
    w = tf.cast(features['w_n'], tf.float32)
    h = tf.cast(features['h_n'], tf.float32)

    img_batch, xc_batch, yc_batch, w_batch, h_batch = tf.train.shuffle_batch([img, xc, yc, w, h],
                                                                                        batch_size= batch_size,
                                                                                        num_threads=64,
                                                                                        capacity=2000,
                                                                                        min_after_dequeue=1500,
                                                                                        )
    
    xc_batch = tf.reshape(xc_batch,[batch_size,1])
    yc_batch = tf.reshape(yc_batch,[batch_size,1])
    w_batch = tf.reshape(w_batch,[batch_size,1])
    h_batch = tf.reshape(h_batch,[batch_size,1])

    box_batch = tf.concat([xc_batch, yc_batch, w_batch, h_batch], 1)

    return img_batch, box_batch

def stick_tfrecord_decode(filename, batch_size):
    filename_queue = tf.train.string_input_producer([filename])# create a queue
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#return file_name and file
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                           'stick_xc': tf.FixedLenFeature([], tf.float32),
                                           'stick_yc': tf.FixedLenFeature([], tf.float32),
                                           'stick_w': tf.FixedLenFeature([], tf.float32),
                                           'stick_h': tf.FixedLenFeature([], tf.float32),
                                           'bottom_xc': tf.FixedLenFeature([], tf.float32),
                                           'bottom_yc': tf.FixedLenFeature([], tf.float32),
                                           'bottom_w': tf.FixedLenFeature([], tf.float32),
                                           'bottom_h': tf.FixedLenFeature([], tf.float32),
                                       })#return image and label

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [cfg.IMG_H, cfg.IMG_W, 3])  #reshape image to 208*208*3
    stick_xc = tf.cast(features['stick_xc'], tf.float32)
    stick_yc = tf.cast(features['stick_yc'], tf.float32)
    stick_w = tf.cast(features['stick_w'], tf.float32)
    stick_h = tf.cast(features['stick_h'], tf.float32)
    bottom_xc = tf.cast(features['bottom_xc'], tf.float32)
    bottom_yc = tf.cast(features['bottom_yc'], tf.float32)
    bottom_w = tf.cast(features['bottom_w'], tf.float32)
    bottom_h = tf.cast(features['bottom_h'], tf.float32)

    img_batch, stick_xc_batch, stick_yc_batch, stick_w_batch, stick_h_batch, bottom_xc_batch, bottom_yc_batch, bottom_w_batch, bottom_h_batch \
        = tf.train.shuffle_batch([img, stick_xc, stick_yc, stick_w, stick_h, bottom_xc, bottom_yc, bottom_w, bottom_h], 
        batch_size= batch_size,
        num_threads=64,
        capacity=2000,
        min_after_dequeue=1500,
        )
    
    stick_xc_batch = tf.reshape(stick_xc_batch,[batch_size,1])
    stick_yc_batch = tf.reshape(stick_yc_batch,[batch_size,1])
    stick_w_batch = tf.reshape(stick_w_batch,[batch_size,1])
    stick_h_batch = tf.reshape(stick_h_batch,[batch_size,1])

    bottom_xc_batch = tf.reshape(bottom_xc_batch,[batch_size,1])
    bottom_yc_batch = tf.reshape(bottom_yc_batch,[batch_size,1])
    bottom_w_batch = tf.reshape(bottom_w_batch,[batch_size,1])
    bottom_h_batch = tf.reshape(bottom_h_batch,[batch_size,1])

    stick_box_batch = tf.concat([stick_xc_batch, stick_yc_batch, stick_w_batch, stick_h_batch], 1)
    bottom_box_batch = tf.concat([bottom_xc_batch, bottom_yc_batch, bottom_w_batch, bottom_h_batch], 1)

    return img_batch, stick_box_batch, bottom_box_batch

def image_normalize(images):
    norm_IMG = images.astype(np.float32)
    for IMG in norm_IMG:
        cv2.normalize(IMG, IMG, 1, 0, cv2.NORM_MINMAX) 
    return norm_IMG

def image_balance(images):
    balanced_IMG = images.astype(np.float32)
    for IMG in balanced_IMG:
        for channel in IMG:
            channel -= np.mean(channel)
    return balanced_IMG

def normxy_2_realxy(box):
    """将归一化的坐标转换为真实坐标"""
    x1 = int(box[0] * cfg.SRC_IMG_W)
    x2 = int(box[2] * cfg.SRC_IMG_W)
    y1 = int(box[1] * cfg.SRC_IMG_H) # 720
    y2 = int(box[3] * cfg.SRC_IMG_H)
    
    x1 = max(x1, 0)
    x2 = min(x2, cfg.SRC_IMG_W)
    y1 = max(y1, 0)
    y2 = min(y2, cfg.SRC_IMG_H)
    return np.array([x1, y1, x2, y2])

def xywh_2_xyxy(box):
    """xywh格式坐标转换为xyxy格式"""
    xmin = box[0] - box[2]/2
    xmax = box[0] + box[2]/2
    ymin = box[1] - box[3]/2
    ymax = box[1] + box[3]/2
    return np.array([xmin, ymin, xmax, ymax])

def pre_2_xywh(pre, anchor):
    """由预测偏移量和anchor得到最终box真实值"""
    pre_x = anchor[0] + pre[0]*anchor[2]
    # pre_y = anchor[1] + pre[1]*anchor[3]
    pre_y = anchor[1] + pre[1]*anchor[2]
    pre_w = anchor[2] * np.exp(pre[2])
    pre_h = anchor[3] * np.exp(pre[3])
    # pre_w = anchor[2] * pre[2]
    # pre_h = anchor[3] * pre[3]
    
    return [pre_x, pre_y, pre_w, pre_h]

def pre_2_prebox(pre, anchor):
    """由预测偏移量和anchor得到最终box真实值"""
    pre_xywh = pre_2_xywh(pre, anchor)
    pre_xyxy_n = xywh_2_xyxy(pre_xywh)
    pre_box = normxy_2_realxy(pre_xyxy_n)
    return pre_box

def draw_points(srcIMG, points):
    """将鼠标绘制的点添加到图像上，用于标定"""
    for _, point in enumerate(points):
        xy_text = "%d,%d" % (point[0], point[1])
        cv2.circle(srcIMG, (point[0], point[1]), 1, (0, 255, 0), thickness = -1)
        cv2.putText(srcIMG, xy_text, (point[0], point[1]), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), thickness = 1)
    return srcIMG

class PPT():
    """透视变换类"""
    def __init__(self):
        self.points1 = []
        self.points2 = np.float32([[0,0], [cfg.TABLE_IMG_W, 0], [0, cfg.TABLE_IMG_H], [cfg.TABLE_IMG_W, cfg.TABLE_IMG_H]])
        self.M = []

    def get_tr_mat(self):
        self.points1 = np.array(self.points1).astype(np.float32)
        self.M = cv2.getPerspectiveTransform(self.points1, self.points2)

    def do_perspective(self, srcIMG):
        dstIMG = cv2.warpPerspective(srcIMG,self.M,(cfg.TABLE_IMG_W,cfg.TABLE_IMG_H))
        return dstIMG

    def get_coordinate(self, raw_point):
        """计算目标点的透视坐标和世界坐标，目标点为图像上的真实值"""
        raw_point_mat = [[raw_point[0]], [raw_point[1]], [1]]
        p2 = np.reshape(np.dot(self.M, raw_point_mat), [3]) # 乘透视变换矩阵
        p2[0] = p2[0]/p2[2]
        p2[1] = p2[1]/p2[2] # 至此获得透视变换后的平面坐标

        bottom_Left_p = [p2[0], cfg.TABLE_IMG_H-p2[1]] # 以左下角为原点的坐标
        real_x = bottom_Left_p[0] / cfg.TABLE_IMG_W * cfg.TABLE_W # 换算为真实世界坐标
        real_y = bottom_Left_p[1] / cfg.TABLE_IMG_H * cfg.TABLE_H
        return p2[0:2].astype(np.int32), [real_x, real_y]

def main1():
    """读取object tfrecords"""
    tfrecords_file = './cubeobject.tfrecords'
    BATCH_SIZE = 100
    img_batch, box_batch = cube_tfrecord_decode(tfrecords_file, BATCH_SIZE)

    with tf.Session()  as sess:

        i = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop() and i<1:
                # just plot one batch size
                images, boxes = sess.run([img_batch, box_batch])

                for j in np.arange(100):
                    
                    srcIMG = images[j,:,:,:]
                    bgrIMG = cv2.cvtColor(srcIMG, cv2.COLOR_RGB2BGR)
                    srcIMG = cv2.resize(bgrIMG,(1280, 720))
                    x1 = int((boxes[j][0] - boxes[j][2]/2) * 1280)
                    x2 = int((boxes[j][0] + boxes[j][2]/2) * 1280)
                    y1 = int((boxes[j][1] - boxes[j][3]/2) * 720)
                    y2 = int((boxes[j][1] + boxes[j][3]/2) * 720)

                    # print(x1, x2, y1, y2)
                    cv2.rectangle(srcIMG, (x1, y1), (x2, y2), (0,0,255), 3)
                    cv2.imshow("src", srcIMG)
                    cv2.waitKey (0)

                i+=1
        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)
        
if __name__ == '__main__':
    tfrecords_file = './StickSegment.tfrecords'
    BATCH_SIZE = 20
    image_batch, mask_batch = segment_tfrecord_decode(tfrecords_file, BATCH_SIZE)

    with tf.Session()  as sess:
        flag = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop() and flag<1:

                srcIMG, maskIMG = sess.run([image_batch, mask_batch])
                

                # for j in np.arange(10):
                #     plt.subplot(121) 
                #     plt.imshow(srcIMG[j,:,:,:])
                #     plt.subplot(122)
                #     plt.imshow(maskIMG[j,:,:,:].squeeze(), cmap='gray')
                #     plt.show()

                flag+=1
        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)