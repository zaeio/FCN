import tensorflow as tf
import numpy as np
from cv2 import cv2
import os
import network as net
import ImageProcessLib as imgLib
import config as cfg

np.set_printoptions(threshold=np.inf)

def show_data(pres, masks, step, feed_dict):
    """打印训练信息: 步数、准确率、误差、学习率"""
    accuracy = compute_accuracy(pres, masks)
    loss_, LR = sess.run([loss, decayed_lr], feed_dict)
    print("step {}: accuracy={:.2}, loss={:.2}, LearningRate={:.3}, ".format(step, accuracy, loss_, LR))
    return accuracy

def compute_accuracy(pres, masks):
    """计算准确度"""
    # 计算分类准确率
    difference = np.abs(np.argmax(pres, axis=3) - masks)
    accuracy = 1 - np.mean(difference)
    return accuracy

def compute_loss():
    with tf.variable_scope('loss'):
        one_hot_label = tf.one_hot(mask_PH, 2, name='one_hot_label')
        cross_entropy = -tf.reduce_sum(one_hot_label * tf.log(tf.clip_by_value(segment_pre,1e-10,1.0)), name='cls_loss')
        tf.summary.scalar('loss', cross_entropy)
        return cross_entropy

""" define placeholder """
img_PH = tf.placeholder(tf.float32, [None, cfg.IMG_H, cfg.IMG_W, 3], name='img_PH')
mask_PH = tf.placeholder(tf.uint8, [None, cfg.IMG_H, cfg.IMG_W], name='img_PH')
batch_PH = tf.placeholder(tf.int32, name='batch_PH') # tf.nn.conv2d_transpose需要指定outputsize，训练时为batchsize，使用时为1
step_PH = tf.placeholder(tf.float32, name='step_PH')
drop_PH = tf.placeholder(tf.float32, name='drop_PH')

""" def net """
with tf.variable_scope('conv11_32'):
    Wcv_11 = tf.Variable(tf.truncated_normal([3,3,3,32], stddev=0.1), name='Wcv_11')
    bcv_11 = tf.Variable(tf.constant(0.1, shape=[32]), name='bcv_11')
    tf.summary.histogram('conv11/weights', Wcv_11)
    tf.summary.histogram('conv11/bias', bcv_11)
    conv_11 = net.conv_2d(img_PH, Wcv_11, bcv_11, outputName='conv_11') # output size 28x28x32
with tf.variable_scope('conv12_32'):
    Wcv_12 = tf.Variable(tf.truncated_normal([3,3,32,32], stddev=0.1), name='Wcv_12')
    bcv_12 = tf.Variable(tf.constant(0.1, shape=[32]), name='bcv_12')
    tf.summary.histogram('conv12/weights', Wcv_12)
    tf.summary.histogram('conv12/bias', bcv_12)
    conv_12 = net.conv_2d(conv_11, Wcv_12, bcv_12, outputName='conv_12') # output size 28x28x32
pool_1 = net.max_pool_2d(conv_12, layerName='pool_1') # output size n/2 x n/2 x 32
with tf.variable_scope('conv21_64'):
    Wcv_21 = tf.Variable(tf.truncated_normal([3,3,32,64], stddev=0.1), name='Wcv_21')
    bcv_21 = tf.Variable(tf.constant(0.1, shape=[64]), name='bcv_21')
    tf.summary.histogram('conv21/weights', Wcv_21)
    tf.summary.histogram('conv21/bias', bcv_21)
    conv_21 = net.conv_2d(pool_1, Wcv_21, bcv_21, outputName='conv_21') # output size n/2 x n/2 x 64
with tf.variable_scope('conv22_64'):
    Wcv_22 = tf.Variable(tf.truncated_normal([3,3,64,64], stddev=0.1), name='Wcv_22')
    bcv_22 = tf.Variable(tf.constant(0.1, shape=[64]), name='bcv_22')
    tf.summary.histogram('conv22/weights', Wcv_22)
    tf.summary.histogram('conv22/bias', bcv_22)
    conv_22 = net.conv_2d(conv_21, Wcv_22, bcv_22, outputName='conv_22') # output size n/2 x n/2 x 64
pool_2 = net.max_pool_2d(conv_22, layerName='pool_2') # output size n/4 x n/4 x 64
with tf.variable_scope('conv31_128'):
    Wcv_31 = tf.Variable(tf.truncated_normal([3,3,64,128], stddev=0.1), name='Wcv_31')
    bcv_31 = tf.Variable(tf.constant(0.1, shape=[128]), name='bcv_31')
    tf.summary.histogram('conv31/weights', Wcv_31)
    tf.summary.histogram('conv31/bias', bcv_31)
    conv_31 = net.conv_2d(pool_2, Wcv_31, bcv_31, outputName='conv_31') # output size n/4 x n/4 x 128
with tf.variable_scope('conv32_128'):
    Wcv_32 = tf.Variable(tf.truncated_normal([3,3,128,128], stddev=0.1), name='Wcv_32')
    bcv_32 = tf.Variable(tf.constant(0.1, shape=[128]), name='bcv_32')
    tf.summary.histogram('conv32/weights', Wcv_32)
    tf.summary.histogram('conv32/bias', bcv_32)
    conv_32 = net.conv_2d(conv_31, Wcv_32, bcv_32, outputName='conv_32') # output size n/4 x n/4 x 128
pool_3 = net.max_pool_2d(conv_32, layerName='pool_3') # output size n/8 x n/8 x 128
with tf.variable_scope('conv41_256'):
    Wcv_41 = tf.Variable(tf.truncated_normal([3,3,128,256], stddev=0.1), name='Wcv_41')
    bcv_41 = tf.Variable(tf.constant(0.1, shape=[256]), name='bcv_41')
    tf.summary.histogram('conv41/weights', Wcv_41)
    tf.summary.histogram('conv41/bias', bcv_41)
    conv_41 = net.conv_2d(pool_3, Wcv_41, bcv_41, outputName='conv_41') # output size n/8 x n/8 x 256
with tf.variable_scope('conv42_256'):
    Wcv_42 = tf.Variable(tf.truncated_normal([3,3,256,256], stddev=0.1), name='Wcv_42')
    bcv_42 = tf.Variable(tf.constant(0.1, shape=[256]), name='bcv_42')
    tf.summary.histogram('conv42/weights', Wcv_42)
    tf.summary.histogram('conv42/bias', bcv_42)
    feature_map = net.conv_2d(conv_41, Wcv_42, bcv_42, outputName='feature') # output size 20 x 28 x 256
# 反卷积
with tf.variable_scope('deconv_81'):
    kernel_81 = tf.Variable(tf.truncated_normal([3,3,128,256], stddev=0.1), name='kernel_81')
    deconv_81 = tf.nn.conv2d_transpose(feature_map, kernel_81, output_shape=[batch_PH,cfg.FEATURE_H*2,cfg.FEATURE_W*2,128], strides=[1,2,2,1], name='deconv_81')
    deconv_act_81 = tf.nn.leaky_relu(deconv_81)
with tf.variable_scope('deconv_82'):
    kernel_82 = tf.Variable(tf.truncated_normal([3,3,64,128], stddev=0.1), name='kernel_81')
    deconv_82 = tf.nn.conv2d_transpose(deconv_act_81, kernel_82, output_shape=[batch_PH,cfg.FEATURE_H*4,cfg.FEATURE_W*4,64], strides=[1,2,2,1], name='deconv_81')
    deconv_act_82 = tf.nn.leaky_relu(deconv_82)
with tf.variable_scope('deconv_83'):
    kernel_83 = tf.Variable(tf.truncated_normal([3,3,2,64], stddev=0.1), name='kernel_81')
    deconv_83 = tf.nn.conv2d_transpose(deconv_act_82, kernel_83, output_shape=[batch_PH,cfg.IMG_H,cfg.IMG_W,2], strides=[1,2,2,1], name='deconv_81')
    segment_pre = tf.nn.softmax(deconv_83, name='output')

saver = tf.train.Saver()
image_batch, mask_batch = imgLib.segment_tfrecord_decode(cfg.tfrecords_file, cfg.BATCH_SIZE)
# image_test, box_test = imgLib.read_and_decode(tfrecords_file, 20)
loss = compute_loss()
decayed_lr = tf.train.exponential_decay(0.001, step_PH, 100, 0.6, staircase=False)
optimizer = tf.train.AdamOptimizer(decayed_lr).minimize(loss)

with tf.Session()  as sess:
    merged = tf.summary.merge_all()#合并可视化数据
    writer = tf.summary.FileWriter("./My_Model",sess.graph)
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    k = 0
    try:
        while not coord.should_stop() and k < 1:
            accuracy = 0
            step = 0
            while accuracy < 0.99:
            # just plot one batch size
                images, masks = sess.run([image_batch, mask_batch])
                images_f = imgLib.image_normalize(images) # 图像归一化

                feed_dict={img_PH:images_f, mask_PH:masks, batch_PH:cfg.BATCH_SIZE, step_PH:step, drop_PH:0.7}
                pres = sess.run(segment_pre, feed_dict)

                if step % 10 == 0:
                    # print('imgf\n', np.around(images_f[0,0:10,0:10,0],3))
                    accuracy = show_data(pres, masks, step, feed_dict)

                    summary = sess.run(merged, feed_dict) # 可视化数据
                    writer.add_summary(summary, step)
                    # 保存数据
                    saver.save(sess, cfg.model_save_path, global_step=step, write_meta_graph=False)
                    # 保存图
                    if not os.path.exists('./My_Model/FCN_Model.meta'):
                        saver.export_meta_graph('./My_Model/FCN_Model.meta')
                
                sess.run(optimizer, feed_dict)
                step += 1
            k += 1
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)



