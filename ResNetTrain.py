import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import os

def read_and_decode_tfrecord(filename):
    # 定义数据流队列和读取器
    filename_queue = tf.train.string_input_producer(filename)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img': tf.FixedLenFeature([], tf.string),
                                       })  # 取出包含image和label的features对象，拆解为图像数据和标签数据
    label = tf.cast(features['label'], tf.int32)
    img = tf.decode_raw(features['img'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    img = tf.cast(img, tf.float32)/255   # 将矩阵归一化
    return img, label

save_model_dir = 'model/'
if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)

# 定义批次大小，x，y占位符，学习率
batch_size_ = 2
lr = tf.Variable(0.0001, dtype=tf.float32)
x = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = tf.placeholder(tf.float32, [None])

train_list = []
class_path = 'data/tfrecord/'
for dir_name in  os.listdir(class_path):
    if 'train' in dir_name:
        train_list.append(os.path.join(class_path, dir_name))

img, label = read_and_decode_tfrecord(train_list)

# 将队列中的数据顺序打乱
img_batch, label_batch = tf.train.shuffle_batch([img, label], num_threads=2, batch_size=batch_size_, capacity=100,
                                                min_after_dequeue=95)
# 将label值进行onehot编码
one_hot_labels = tf.one_hot(indices=tf.cast(y, tf.int32), depth=50)

# 采用resnet进行训练返回50维数的向量，和网络对应的激活函数的成分的字典
pre, end_points = nets.resnet_v2.resnet_v2_50(x, num_classes=50, is_training=True)
pre = tf.reshape(pre, shape=[-1, 50])

# 定义损失函数和优化器
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pre, labels=one_hot_labels))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# 准确度
a = tf.argmax(pre, 1)
b = tf.argmax(one_hot_labels, 1)
correct_pre = tf.equal(a, b)
accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 创建一个协调器，管理线程
    coord = tf.train.Coordinator()
    # 启动QueueRunner,此时文件名队列已经进队
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 0
    while True:
        i += 1
        b_image, b_label = sess.run([img_batch, label_batch])
        if len(b_label) == 50:
            acc_train, loss_, y_t, y_p, a_, b_ = sess.run([accuracy, optimizer, loss, one_hot_labels, pre, a, b], feed_dict={x: b_image,
                                                                                                           y: b_label})
            print('step: {}, train_loss: {}, acc_train: {}'.format(i, loss_, acc_train))
            if i == 500:
                saver.save(sess, save_model_dir, global_step=i)
            elif i == 800:
                saver.save(sess, save_model_dir, global_step=i)
            elif i == 1000:
                saver.save(sess, save_model_dir, global_step=i)
        coord.request_stop()
        # 其他所有线程关闭之后，这一函数才能返回
        coord.join(threads)