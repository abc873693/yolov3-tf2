
import os
import shutil
from cv2 import cv2 as cv

file_type = ".png"

_type = 'test'

out_path = './dataset/mm_v2_depth_crop/{}/'.format(_type)
source_path = './dataset/microfield_monocular_v2_depth/416X416/{}/'.format(_type)
if not os.path.exists(out_path):
    os.makedirs(out_path)
image_width = 416.
image_height = 416.

crop_height = 104
crop_width = 104

cfg = open('{}.txt'.format(_type), "w")

for file_name in os.listdir(source_path):
    if file_type in file_name:
        # , cv.IMREAD_UNCHANGED
        image = cv.imread(source_path + file_name, cv.IMREAD_UNCHANGED)
        print(file_name)
        image_path = source_path + file_name
        label_path = image_path.replace(file_type, '.txt')
        label_name = label_path.split('/')[-1]
        label_text = open(label_path, "r")
        bboxes = label_text.read().splitlines()
        i = 0
        for text in bboxes:
            array = text.split(' ')
            if len(array) != 0 and array[0] != '':
                train = open(out_path + label_name.replace('.txt',
                                                           '_' + str(i) + '.txt'), "w")
                class_id = int(array[0])
                center_x = float(array[1]) * image_width
                center_y = float(array[2]) * image_height
                width = float(array[3]) * image_width
                height = float(array[4]) * image_height
                size = array[5]
                distance = array[6]
                x_min = int(center_x - width / 2.0)  # xmin
                x_max = int(center_x + width / 2.0)  # xmax
                y_min = int(center_y - height / 2.0)  # ymin
                y_max = int(center_y + height / 2.0)  # ymax
                train.write(text)
                train.close()

                crop = image[y_min:y_max, x_min:x_max]
                crop = cv.resize(crop, (crop_height, crop_width),
                                 interpolation=cv.INTER_CUBIC)
                crop_image_path = out_path + \
                    file_name.replace('.png', '_' + str(i) + '.png')
                cv.imwrite(crop_image_path, crop)
                cfg.write(crop_image_path + '\n')
                i += 1
