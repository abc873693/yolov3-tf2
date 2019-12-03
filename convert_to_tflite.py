from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
#from yolov3_tf2.models import YoloV3, YoloV3Tiny


def main(_argv):
    #, input_shapes={'input_1': [1, 416, 416, 3]}
    converter = tf.lite.TFLiteConverter.from_keras_model(
        'checkpoints/yolov3_tiny_20190822.h5')
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)


if __name__ == '__main__':
    app.run(main)
