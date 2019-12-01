import os
import tensorflow as tf
from PIL import Image
import shutil
# 第几个TFRecord文件
recordfilenum = 0

# 第几个图片
num = 0

# tfrecord文件保存路径
folder = 'data/tfrecord/'

# 每个tfrecord存放图片个数
bestnum = 500

# train集路径
train_path = 'data/train_augmentation/'

# 验证集路径
validation_sets = 'data/validation_set/'

# tfrecord 文件格式名字
tfrecordfilename = ('train_tfrecord.tfrecords-%d' % recordfilenum)

# 判断是否有folder目录
if not os.path.exists(folder):
    os.makedirs(folder)

# 拿出train里的所有文件名字
classes = []
for i in os.listdir(train_path):
    classes.append(i)
print(classes)

# 生成的文件
writer = tf.python_io.TFRecordWriter(os.path.join(folder, tfrecordfilename))

# 拿出文件名字和index
for index, name in enumerate(classes):
    # print(index, name)
    class_path = os.path.join(train_path, name)
    print(class_path)
    for img_name in os.listdir(class_path):
        num += 1
        if num > bestnum: # 超过500，写入下一个tfrecord文件
            num = 1
            recordfilenum += 1
            tfrecordfilename = ('train_tfrecord.tfrecords-%d' % recordfilenum)
            writer = tf.python_io.TFRecordWriter(os.path.join(folder, tfrecordfilename))# 生成tfrecord文件

        img_path = os.path.join(class_path, img_name)
        val_path = os.path.join(validation_sets, name)
        if not os.path.exists(val_path):
            os.makedirs(val_path)
        if num % 8 == 0:
            print(num)
            shutil.move(img_path, os.path.join(val_path, img_name))  # 将img_path路径下的图片移动到 os.path.join(val_path,img_name)
        else:
            img = Image.open(img_path)  # 加载路径下的图片
            img = img.tobytes()  # 将图片转化为二进制格式
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img]))
            }))  # example对象对label和image数据进行封装；Int64List存放图像的标签对应的整数；BytesList存放图像数据
            writer.write(example.SerializeToString())  # 序列化为字符串
writer.close()



