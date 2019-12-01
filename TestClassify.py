import os
import csv
import shutil
import numpy as np
from PIL import Image
import cv2
from PIL import Image
import os.path
import glob
def convertjpg(jpgfile,outdir,width=224,height=224):
    img = Image.open(jpgfile)
    try:
        new_img = img.resize((width, height), Image.BILINEAR)
        new_img.save(os.path.join(outdir, os.path.basename(jpgfile)))
    except Exception as e:
        print(e)

dic_data = {}
with open("data/test_submission.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        print(line[1]+line[2])
        dic_data[line[1]] = line[2]

test_dir = 'data/resize_test_data/'
test_file = 'data/test_final'
for img_file_name in os.listdir(test_dir):
    class_name = dic_data[img_file_name]
    class_path = os.path.join(test_file, class_name)
    if not os.path.exists(class_path):
        os.makedirs(class_path)
    img_path = os.path.join(test_dir, img_file_name)
    for jpgfile in glob.glob(img_path):
        convertjpg(jpgfile, class_path)
