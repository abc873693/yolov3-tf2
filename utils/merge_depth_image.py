
import os
import shutil
import numpy as np
from cv2 import cv2 as cv

source_path = '/home/MIT_lab/bts/workspace/bts/tensorflow/shrimp_v7_test/rgb/'
depth_path = '/home/MIT_lab/bts/workspace/bts/tensorflow/shrimp_v7_test/cmap/'
out_path = '/home/MIT_lab/yolov3-tf2/dataset/shrimp_v9/test/'

if not os.path.exists(out_path):
    os.makedirs(out_path)


def resizeAndSaveToDir(source_path, target_path):
    pic = cv.imread(source_path)
    height, width = pic.shape[:2]
    frame = cv.resize(pic, (416, 416), interpolation=cv.INTER_CUBIC)
    cv.imwrite(
        target_path, frame)
    cv.waitKey(1)


width = 416
height = 416

for file_name in os.listdir(source_path):
    if ".jpg" in file_name:
        print(file_name)
        depth_file_name = file_name.replace('.jpg', '.png')
        label_file_name = file_name.replace('.jpg', '.txt')
        label_path = source_path + label_file_name
        image = cv.imread(source_path + file_name)
        image = cv.resize(image, (height, width),
                          interpolation=cv.INTER_CUBIC)
        rgbd = np.full((height, width, 4), 255, dtype=int)
        if os.path.exists(depth_path + depth_file_name):
            depth = cv.imread(depth_path + depth_file_name)
            depth = cv.resize(depth, (height, width),
                              interpolation=cv.INTER_CUBIC)
            rgbd[:, :, -1] = depth[:, :, 0]
        rgbd[:, :, 0:3] = image[:, :, :]
        shutil.copy(label_path, out_path + label_file_name)
        cv.imwrite(out_path + depth_file_name, rgbd)
