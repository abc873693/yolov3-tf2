import os
import shutil

if __name__ == '__main__':
    datasets = ['1929', 'clear999', 'clear1000', 'clear1005', 'clear1000_2', 'clear1001', 'green859', 'noShrimp100', 'noShrimp200_1', 'noShrimp200_2', 'noShrimp200_3',
                'noShrimp200_4', 'noShrimp200_5', 'noShrimp200_6', 'noShrimp200_7', 'noShrimp200_8', 'noShrimp200_9', 'thickBrown1227', 'thinGreen1068', 'yellowGreen1052',
                'ray1000_1', 'ray1000_2', 'ray1000_3', 'ray1000_4', 'green1000', 'microfield_1', 'microfield_2', 'microfield_3', 'microfield_4', 'microfield_5_v2']
    train_out = open('shrimp.v3/train.txt', "w")
    valid_out = open('shrimp.v3/valid.txt', "w")
    for dataset in datasets:
        root_path = 'dataset/{0}/416X416'.format(dataset)
        train_path = '{0}/cfg/train.txt'.format(root_path)
        valid_path = '{0}/cfg/valid.txt'.format(root_path)
        train = open(train_path, "r")
        for filePath in train.read().splitlines():
            train_out.write(filePath)
            train_out.write('\n')
        if not('noShrimp' in dataset):
            valid = open(valid_path, "r")
            for filePath in valid.read().splitlines():
                valid_out.write(filePath)
                valid_out.write('\n')
    train_out.close()
    valid_out.close()
