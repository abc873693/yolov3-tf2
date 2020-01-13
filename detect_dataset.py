import time
import os
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import (draw_outputs, read_yolo_labels, yolo_extend_evaluate)

flags.DEFINE_string('classes', './data/shrimp.names', 'path to classes file')
flags.DEFINE_string('weights_postfix', 'last', 'path to weights postfix')
flags.DEFINE_boolean('tiny', True, 'yolov3 or yolov3-tiny')
flags.DEFINE_boolean('map', False, 'calculate')
flags.DEFINE_float('iou_trethold', 0.5, 'iou_trethold')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('experiment', '20200109_2', 'path to dataset')
flags.DEFINE_string('dataset', 'microfield_5_v2_test', 'path to dataset')
# flags.DEFINE_string('output_path', 'output/', 'path to output path')
flags.DEFINE_integer('num_classes', 1, 'number of classes in the model')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    weights_path = 'checkpoints/{}/yolov3_train_{}.tf'.format(FLAGS.experiment, FLAGS.weights_postfix)

    yolo.load_weights(weights_path)
    logging.info('weights loaded')
    yolo.summary()

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    experiment_path = 'results/{}-{}/'.format(FLAGS.experiment, FLAGS.weights_postfix)
    experiment_dataset_path = '{}/{}/'.format(experiment_path, FLAGS.dataset)
    
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    if not os.path.exists(experiment_dataset_path):
        os.makedirs(experiment_dataset_path)

    input_path = 'data/{}/'.format(FLAGS.dataset)

    image_count = 0
    label_count = 0

    REs = []

    TP_total, FP_total, FN_total = 0, 0, 0

    for file_name in os.listdir(input_path):
        logging.info(file_name)
        if file_name.find('.jpg') == -1:
            continue
        image_path = input_path + file_name

        image_count += 1

        img = tf.image.decode_image(open(image_path, 'rb').read(), channels=3)
        img = tf.expand_dims(img, 0)
        img = transform_images(img, FLAGS.size)

        t1 = time.time()
        boxes, sizes, scores, classes , nums = yolo(img)
        t2 = time.time()
        logging.info('time: {}'.format(t2 - t1))

        # logging.info('detections:{}'.format(nums[0]))
        # for i in range(nums[0]):
        #     logging.info('\t{}, {}, {} size = {}'.format(class_names[int(classes[0][i])],
        #                                     np.array(scores[0][i]),
        #                                     np.array(boxes[0][i]),
        #                                     sizes[0][i]))       
                                            

        img = cv2.imread(image_path)
        img = draw_outputs(img, (boxes, sizes , scores, classes, nums), class_names)
        output_path = '{}/{}'.format(experiment_dataset_path , file_name)
        cv2.imwrite(output_path , img)
        logging.info('output saved to: {}'.format(output_path))

        if FLAGS.map:
            label_filename = file_name.replace('.jpg','.txt')
            label_path = input_path + label_filename
            if os.path.exists(label_path):
                label_count += 1
                # labels np array [class_id, center_x, center_y, width, height, size]
                labels, has_size_label, labels_nums = read_yolo_labels(label_path)
                classes_true = labels[ : , 0]
                boxes_ture = labels[ : , 1:5]
                size_ture = np.array([])

                if has_size_label:
                    size_ture = labels[: , 5]
                outputs = (boxes, sizes , scores, classes, nums)
                grund_truth = (boxes_ture, size_ture, classes_true, labels_nums)
                TP, FP, FN, RE = yolo_extend_evaluate(outputs , grund_truth , FLAGS.iou_trethold)
                TP_total += TP
                FP_total += FP
                FN_total += FN
                REs.extend(RE)

    are = 0
    if len(REs) != 0:
        are = sum(REs) / len(REs)
    precision = TP_total / (TP_total + FP_total)
    recall = TP_total / (TP_total + FN_total)
    results_text = open('records.txt', "a")
    results_text.write('{} {} {} {:.2f}% {:.2f}% {:.2f}%\n'.format(TP_total, FP_total, FN_total, precision * 100.0, recall * 100.0, are * 100.0))
    logging.info('TP = {} FP = {} FN = {}'.format(TP_total, FP_total, FN_total))
    logging.info('precision = {:.2f} recall = {:.2f}, size average ralative error = {:.2f}'.format(precision, recall, are * 100.0))
    results_text.close()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
