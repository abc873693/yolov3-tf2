
import os
import shutil

file_type = ".jpg"

label_base_path = './'
source_path = './dataset/microfield_monocular_v2/416X416/valid/'
if not os.path.exists(label_base_path):
    os.makedirs(label_base_path)
image_width = 416.
image_height = 416.

for file_name in os.listdir(source_path):
    if file_type in file_name:
        print(file_name)
        image_path = source_path + file_name
        label_path = image_path.replace(file_type, '.txt')
        label_text = open(label_path, "r")
        bboxes = label_text.read().splitlines()
        # out = open('tmp/' + label_path.split('/')[-1], "w")
        label = ''
        for text in bboxes:
            array = text.split(' ')
            if len(array) != 0 and array[0] != '':
                class_id = int(array[0])
                center_x = float(array[1])
                center_y = float(array[2])
                width = float(array[3])
                height = float(array[4])
                label += '{} {} {} {} {}\n'.format(class_id, center_x,
                                                  center_y, width, height)
        label_text.close()
        out = open(label_path, "w")
        out.write(label + "\n")
        out.close()
