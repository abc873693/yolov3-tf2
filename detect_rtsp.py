import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs
import os
import numpy as np


flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('url', 'rtsp://192.168.100.10/h264/ch1/main/av_stream', 'rtsp url')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video.mp4',
                    'path to video file or number for webcam)')


def main(_argv):
    #%%
    if FLAGS.tiny:
        yolo = YoloV3Tiny()
    else:
        yolo = YoloV3()

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    times = []

    cap = cv2.VideoCapture(FLAGS.url)

    out = cv2.VideoWriter('appsrc ! videoconvert ! '
                          'x264enc noise-reduction=10000 speed-preset=ultrafast tune=zerolatency ! '
                          'rtph264pay config-interval=1 pt=96 !'
                          'tcpserversink host=140.117.169.194 port=5000 sync=false',
                          0, 25, (640, 480))

    out_path = './out/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    #%%
    while(cap.isOpened()):
        ret, img = cap.read()

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

        img_in = tf.expand_dims(img, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        t2 = time.time()
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2) 
        # if(nums > 0):
        #     cv2.imwrite(out_path + 'frame{0}.jpg'.format(index), img)
        frameOfWindows = cv2.resize(
            img, (800, 600), interpolation=cv2.INTER_CUBIC)
        out.write(frameOfWindows)
        cv2.imshow('output', frameOfWindows)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
