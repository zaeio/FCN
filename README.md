V3.1
使用了用高级的损失函数
V3.2
加入透视变换，实现定位
输入图像大小改为224-160，特征图大小改为28-20

V4.0
用于识别棒状物体
将anchor结构改为长条形
用整棒进行分类，用棒底进行回归
network.py: 用tf.nn.bias_add代替以前的'+'，解决了卷积输出全为负的bug
