from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os

# 数据增强增加学习样本
datagen = ImageDataGenerator(
    rotation_range=40, # 角度值，0~180，图像旋转
    width_shift_range=0.2, # 水平平移，相对总宽度的比例
    height_shift_range=0.2, # 垂直平移，相对总高度的比例
    shear_range=0.2, # 随机错切换角度
    zoom_range=0.2, # 随机缩放范围
    horizontal_flip=True, # 一半图像水平翻转
    fill_mode='nearest' # 填充新创建像素的方法
)
data_dir = 'data/resize_train_data'

if __name__ == '__main__':
    for file in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, file)
        # print(class_dir)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = load_img(img_path)
                save_path = 'data/train_augmentation/' + file
                if not os.path.exists(save_path):
                    # 不存在创建目录
                    # 创建目录函数
                    os.makedirs(save_path)
                x = img_to_array(img)
                print(x.shape)
                x = x.reshape((1,) + x.shape)
                print(x.shape)
                n = 1
                for bath in datagen.flow(x, batch_size=1, save_to_dir=save_path, save_prefix='train',
                                         save_format='jpeg'):
                    n += 1
                    if n > 6:
                        break

