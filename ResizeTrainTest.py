from PIL import Image
import os

# resize_train
# folder = 'data/train/'
# resize_path = 'data/resize_train_data/'
# for file in os.listdir(folder):
#     class_dir = folder + file
#     print(class_dir)
#     if os.path.isdir(class_dir):
#         for img_name in os.listdir(class_dir):
#             img = Image.open(os.path.join(class_dir, img_name))
#             out = img.resize((224, 224))
#             if not os.path.exists(os.path.join(resize_path, file)):
#                 os.makedirs(os.path.join(resize_path, file))
#             out.save(os.path.join(resize_path, file, img_name))

# resize_test
folder = 'data/test/'
resize_path = 'data/resize_test_data'
for file in os.listdir(folder):
    class_dir = folder + file
    print(class_dir)
    img = Image.open(class_dir)
    out = img.resize((224, 224))
    if not os.path.exists(os.path.join(resize_path)):
        os.makedirs(os.path.join(resize_path))
    out.save(os.path.join(resize_path, file))
