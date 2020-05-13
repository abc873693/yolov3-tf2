
import os
import shutil

# label_base_path = './'
# label_name = 'shrimp_test_files_with_gt.txt'
# source_path = '/home/MIT_lab/yolov3-tf2/dataset/microfield_monocular_v2_depth/416X416/test/'
# if not os.path.exists(label_base_path):
#     os.makedirs(label_base_path)
# train = open(label_base_path + label_name, "a")

# for filepname in os.listdir(source_path):
#     if ".jpg" in filepname or ".png" in filepname:
#         print(filepname)
#         #text_filename = filepname.replace('.jpg','.txt')
#         #tmp = open(source_path + text_filename, "w")
#         #tmp.close()
#         train.write(label_base_path + filepname + ' study_room/sync_depth_00272.png 518.8579' +  "\n")
# train.close()

if __name__ == '__main__':
    dataset_name = 'dataset/shrimp_v5'
    datasets = ['1929', 'clear999', 'clear1000', 'clear1005', 'clear1000_2', 'clear1001', 'green859', 'noShrimp100', 'noShrimp200_1', 'noShrimp200_2', 'noShrimp200_3',
                'noShrimp200_4', 'noShrimp200_5', 'noShrimp200_6', 'noShrimp200_7', 'noShrimp200_8', 'noShrimp200_9', 'thickBrown1227', 'thinGreen1068', 'yellowGreen1052',
                'ray1000_1', 'ray1000_2', 'ray1000_3', 'ray1000_4', 'green1000', 'microfield_1', 'microfield_2', 'microfield_3', 'microfield_4', 'microfield_5_v2']
    train_out = open('train.txt', "w")
    valid_out = open('valid.txt', "w")
    test_out = open('test.txt', "w")
    append_text = ' dataset/study_room/sync_depth_00272.png 518.8579'
    for dataset in datasets:
        root_path = 'dataset/{0}/416X416'.format(dataset)
        train_path = '{0}/cfg/train.txt'.format(root_path)
        valid_path = '{0}/cfg/valid.txt'.format(root_path)
        test_path = '{0}/cfg/test.txt'.format(root_path)
        train = open(train_path, "r")
        for filePath in train.read().splitlines():
            fileName = filePath.split('/')[-1]
            labelPath = filePath.replace('.jpg', '.txt')
            labelName = fileName.replace('.jpg', '.txt')
            fileNewPath = '{}/train/{}_'.format(dataset_name, dataset)+ fileName
            train_out.write(fileNewPath + append_text)
            train_out.write('\n')
            shutil.copy(filePath, fileNewPath)
            shutil.copy(labelPath, '{}/train/{}_'.format(dataset_name, dataset)+ labelName)
        if not('noShrimp' in dataset):
            valid = open(valid_path, "r")
            test = open(test_path, "r")
            for filePath in valid.read().splitlines():
                fileName = filePath.split('/')[-1]
                labelPath = filePath.replace('.jpg', '.txt')
                labelName = fileName.replace('.jpg', '.txt')
                fileNewPath = '{}/valid/{}_'.format(dataset_name, dataset)+ fileName
                shutil.copy(filePath, fileNewPath)
                shutil.copy(labelPath, '{}/valid/{}_'.format(dataset_name, dataset) + labelName)
                valid_out.write(fileNewPath + append_text)
                valid_out.write('\n')
            for filePath in test.read().splitlines():
                fileName = filePath.split('/')[-1]
                labelPath = filePath.replace('.jpg', '.txt')
                labelName = fileName.replace('.jpg', '.txt')
                fileNewPath = '{}/test/{}_'.format(dataset_name, dataset)+ fileName
                shutil.copy(filePath, fileNewPath)
                shutil.copy(labelPath, '{}/test/{}_'.format(dataset_name, dataset) + labelName)
                test_out.write(fileNewPath + append_text)
                test_out.write('\n')
    train_out.close()
    valid_out.close()
    test_out.close()
