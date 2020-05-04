import tensorflow as tf
import numpy as np

def fc_layer(inputs, W, b, activation_function=tf.nn.relu, outputName='fc_output'):
    """添加全连接层"""
    if activation_function == None:
        outputs = tf.nn.bias_add(tf.matmul(inputs, W), b, name=outputName)
    else:
        Wx_plus_b = tf.nn.bias_add(tf.matmul(inputs, W), b)
        outputs = activation_function(Wx_plus_b, name=outputName)

    return outputs

def conv_2d(inputs, W, b, stride=[1,1], activation_function=tf.nn.leaky_relu, outputName='conv_output'):
    """添加卷积层"""
    conv = tf.nn.conv2d(inputs, W, strides=[1,stride[0],stride[1],1], padding='SAME')
    outputs = activation_function(tf.nn.bias_add(conv, b), name=outputName)

    return outputs

def max_pool_2d(inputs, k=[2,2], layerName='pool_layer'):
    """添加池化层"""
        # [1,2,2,1]: 图像个数为1，窗口大小2x2，图像深度为1
    outputs = tf.nn.max_pool(inputs, ksize=[1,k[0],k[1],1], strides=[1,k[0],k[1],1], padding='SAME' ,name=layerName)
    return outputs

def SPP(inputs, levels=[3,1], layerName='SPP'):
    """魔改版的SPP, 纵向多分段, 横向3 1分, 用于棒子"""
    # 添加Spatial Pyramid Pooling.   Works for levels=[1, 2, 3, 6, ···]
    with tf.name_scope(layerName):
        input_shape = inputs.get_shape().as_list() # get input size
        # print (input_shape)
        pool_outputs = [] # 保存各level的Tensor
        for i in levels:
            if i !=1:
                k_h = input_shape[1] / 10
            else:
                k_h = input_shape[1] / i
            k_w = input_shape[2] / i
            pool = tf.nn.max_pool(inputs, ksize=[1, k_h, k_w, 1], strides=[1, k_h, k_w, 1], padding='SAME')
            pool_outputs.append(tf.reshape(pool, [tf.shape(inputs)[0], -1])) # 将各level输出的Tensor放入list

        spp_pool = tf.concat(pool_outputs, 1) # 拼接Tensor
        # print("SPPool output shape {:}".format(spp_pool.get_shape().as_list()))
        return spp_pool

"""
def SPP(inputs, levels=[4,2,1], layerName='SPP'):
    # 添加Spatial Pyramid Pooling.   Works for levels=[1, 2, 3, 6, ···]
    with tf.name_scope(layerName):
        input_shape = inputs.get_shape().as_list() # get input size
        # print (input_shape)
        pool_outputs = [] # 保存各level的Tensor
        for i in levels:
            k_h = input_shape[1] / i
            k_w = input_shape[2] / i
            pool = tf.nn.max_pool(inputs, ksize=[1, k_h, k_w, 1], strides=[1, k_h, k_w, 1], padding='SAME')
            pool_outputs.append(tf.reshape(pool, [tf.shape(inputs)[0], -1])) # 将各level输出的Tensor放入list

        spp_pool = tf.concat(pool_outputs, 1) # 拼接Tensor
        # print("SPPool output shape {:}".format(spp_pool.get_shape().as_list()))
        return spp_pool
"""
