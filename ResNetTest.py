import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
from PIL import Image
import os
import csv
from time import *

dic_data = {}
with open('data/test_submission.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        # print(line[1]+line[2])
        dic_data[line[1]] = line[2]
        # print(dic_data)
# 训练文件路径
img_train_data = 'data/resize_train_data'

classes = []
for i in os.listdir(img_train_data):
    classes.append(i)



# 测试文件路径
img_test_data = 'data/test_final/'

#模型地址
model_path = 'model/model-30000'

# 生成输出结果
# result_dir = './result.txt'
# if not os.path.exists(result_dir):
#     os.mkdir(result_dir)

x = tf.placeholder(tf.float32, [None, 224, 224, 3])
pre, end_points = nets.resnet_v2.resnet_v2_50(x, num_classes=50, is_training=True) # resnet函数中存在softmax函数，pre已经
                                                                                            # 归一化了 50个位数加起来和是1
pre = tf.reshape(pre, shape=[-1, 50]) # 调整pre形状（厚度）
a = tf.argmax(pre, 1) # 拿出值最大所在的位置就是类别
saver = tf.train.Saver() # 保存模型语句
with tf.Session() as sess:
    begin_time = time()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_path)
    num = 0
    acc_num = 0
    for pic in os.listdir(img_test_data):
        class_path = os.path.join(img_test_data, pic)
        for img_file_name in os.listdir(class_path):
            num += 1
            if '.jpg' in img_file_name:
                img_path = os.path.join(class_path, img_file_name)
                img = Image.open(img_path)
                img = tf.reshape(img, [1, 224, 224, 3])
                # img1 = tf.reshape(img, [1, 224, 224, 3])
                img = tf.cast(img, tf.float32) / 255.0  # 使矩阵中的pixel在0~1之间
                b_image, b_image_raw = sess.run([img, img])
                t_label = sess.run(a, feed_dict={x: b_image})
                index_ = t_label[0]
                print(img_path, t_label, classes[index_], dic_data[img_file_name]) # dic_data[img_file_name]返回照片名对应的类名字
                if dic_data[img_file_name] == classes[index_]:  # dic_data[pic]是测试集表格中的类名字，classes[index_]是预测输出的类名
                    acc_num += 1
                # predict = classes[index_]
                # with open(result_dir, 'a') as f1:
                #     if pic == predict:
                #         print(pic, img_file_name, predict, file=f1)
    acc = acc_num / num
    print(acc_num)
    print(num)
    print("acc:" + str(acc))
    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', run_time)