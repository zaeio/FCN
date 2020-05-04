SRC_IMG_W = 640
SRC_IMG_H = 480
IMG_W = 224 # 224
IMG_H = 160 # 128
FEATURE_W = 28 # 28
FEATURE_H = 20 # 16
BATCH_SIZE = 20
SAMPLE_SIZE = 6

# model_save_path = r'E:\BK\Program\TensorFlow\RCNN2\My Model\CNN_Model'
# tfrecords_file = r'E:\BK\Program\TensorFlow\RCNN2\cubeobject.tfrecords'
# test_file = r'E:\BK\Program\TensorFlow\RCNN2\object_test.tfrecords'

model_save_path = './My_Model/FCN_Model'
tfrecords_file = './StickSegment.tfrecords'

TABLE_IMG_W = 640 # 透视变换后的图像宽高
TABLE_IMG_H = 502
TABLE_W = 395 # 桌面的真实宽高，单位mm
TABLE_H = 310
CUBE_IMG_SIZE = 92 # cube在变换后的图像上的边长 = 57mm/310mm*502
CUBE_SIZE = 58 # 魔方的真实边长，单位mm