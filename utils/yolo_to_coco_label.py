
import os
import shutil

file_type = ".jpg"

label_base_path = './'
label_name = 'dataset.txt'
source_path = './dataset/microfield_monocular_v2_depth/416X416/train/'
if not os.path.exists(label_base_path):
    os.makedirs(label_base_path)
train = open(label_base_path + label_name, "w")
image_width = 416.
image_height = 416.

for file_name in os.listdir(source_path):
    if file_type in file_name:
        print(file_name)
        image_path = source_path + file_name
        label_path = image_path.replace(file_type, '.txt')
        label_text = open(label_path, "r")
        bboxes = label_text.read().splitlines()
        label = image_path + ''
        for text in bboxes:
            array = text.split(' ')
            if len(array) != 0 and array[0] != '':
                class_id = int(array[0])
                center_x = float(array[1]) * image_width
                center_y = float(array[2]) * image_height
                width = float(array[3]) * image_width
                height = float(array[4]) * image_height
                x_min = center_x - width / 2.0  # xmin
                x_max = center_x + width / 2.0  # xmax
                y_min = center_y - height / 2.0  # ymin
                y_max = center_y + height / 2.0  # ymax
                label += ' {},{},{},{},{}'.format(x_min,
                                                  y_min, x_max, y_max, class_id)
        train.write(label + "\n")
train.close()
