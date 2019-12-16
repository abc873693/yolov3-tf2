"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python yolo_to_tfrecord.py --text_input_path=data/train_labels.text  --output_path=train.tfrecord
  # Create test data:
  python yolo_to_tfrecord.py --text_input_path=data/test_labels.text  --output_path=test.tfrecord
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
# import pandas as pd
import tensorflow as tf

from PIL import Image
# from tensorflow.contrib.object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict
import hashlib


from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_string('text_input_path', 'train.txt',
                    'Path to the yolo label path list input')
flags.DEFINE_string('output_path', 'train.tfrecord', 'Path to output TFRecord')

# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'raccoon':
        return 1
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(imagePath):
    with tf.io.gfile.GFile(imagePath, 'rb') as fid:
        encoded_jpg = fid.read()
    labelPath = imagePath.replace('.jpg', '.txt')
    labelText = open(labelPath, "r")
    bboxes = labelText.read().splitlines()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = imagePath.rsplit('/', 1)[1]
    # print(filename)
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    sizees = []

    for text in bboxes:
        array = text.split(' ')
        if len(array) != 0:
            class_id = int(array[0])
            center_x = float(array[1])
            center_y = float(array[2])
            width = float(array[3])
            height = float(array[4])
            if len(array >= 5):
                size = float(array[5])
            xmins.append(center_x - width / 2.0)  # xmin
            xmaxs.append(center_x + width / 2.0)  # xmax
            ymins.append(center_y - height / 2.0)  # ymin
            ymaxs.append(center_y + height / 2.0)  # ymax
            sizees.append(size)
            classes_text.append('shrimp'.encode('utf8'))
            classes.append(class_id)
            print(size)
    key = hashlib.sha256(encoded_jpg).hexdigest()
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(416),
        'image/width': _int64_feature(416),
        'image/filename': _bytes_feature(filename.encode('utf8')),
        'image/source_id': _bytes_feature(filename.encode('utf8')),
        'image/encoded': _bytes_feature(encoded_jpg),
        'image/format': _bytes_feature(image_format),
        'image/key/sha256': _bytes_feature(key.encode('utf8')),
        'image/object/bbox/xmin': _float_list_feature(xmins),
        'image/object/bbox/xmax': _float_list_feature(xmaxs),
        'image/object/bbox/ymin': _float_list_feature(ymins),
        'image/object/bbox/ymax': _float_list_feature(ymaxs),
        'image/object/bbox/size': _float_list_feature(sizees),
        'image/object/class/text': _bytes_list_feature(classes_text),
        'image/object/class/label': _int64_list_feature(classes),
    }))
    # print(tf_example)
    return tf_example


def _bytes_list_feature(values):
    """Returns a bytes_list from a string / byte."""
    if isinstance(values, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        values = values.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_list_feature(values):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_list_feature(values):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def main(_argv):
    text = open(FLAGS.text_input_path, "r")
    output_path = FLAGS.output_path
    writer = tf.io.TFRecordWriter(output_path)
    for filePath in text.read().splitlines():
        tf_example = create_tf_example(filePath)
        writer.write(tf_example.SerializeToString())
        output_path = os.path.join(os.getcwd(), output_path)

    print('Successfully created the TFRecords: {}'.format(output_path))
    writer.close()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
