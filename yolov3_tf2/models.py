from absl import flags
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)
from .batch_norm import BatchNormalization
from .utils import (
    broadcast_iou,
    read_yolo_labels,
    yolo_extend_evaluate,
    single_value_evaluate
)
from .dataset import transform_images

import os

flags.DEFINE_float('yolo_iou_threshold', 0.5, 'iou threshold')
flags.DEFINE_float('yolo_score_threshold', 0.25, 'score threshold')

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58),
                              (81, 82), (135, 169),  (344, 319)],
                             np.float32) / 416
yolo_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])


def DarknetConv(x, filters, size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x


def DarknetResidual(x, filters):
    prev = x
    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)
    x = Add()([prev, x])
    return x


def DarknetBlock(x, filters, blocks):
    x = DarknetConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)
    return x


def Darknet(name=None, channels=3):
    x = inputs = Input([None, None, channels])
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)  # skip connection
    x = x_36 = DarknetBlock(x, 256, 8)  # skip connection
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)


def DarknetTiny(name=None, channels=3):
    x = inputs = Input([None, None, channels])
    x = DarknetConv(x, 16, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 32, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 64, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 128, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = x_8 = DarknetConv(x, 256, 3)  # skip connection
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 512, 3)
    x = MaxPool2D(2, 1, 'same')(x)
    x = DarknetConv(x, 1024, 3)
    return tf.keras.Model(inputs, (x_8, x), name=name)


def YoloConv(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)
    return yolo_conv


def YoloConvTiny(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])
            x = DarknetConv(x, filters, 1)

        return Model(inputs, x, name=name)(x_in)
    return yolo_conv


def YoloOutput(filters, anchors, classes, name=None):
    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, anchors * (classes + 6), 1, batch_norm=False)
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                            anchors, classes + 6)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)
    return yolo_output


def yolo_boxes(pred, anchors, classes):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h , obj, ...classes, size))
    grid_size = tf.shape(pred)[1]
    box_xy, box_wh, objectness,  class_probs, size = tf.split(
        pred, (2, 2, 1, classes, 1), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    size = tf.sigmoid(size)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
        tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, size, pred_box


def yolo_nms(outputs, anchors, masks, classes):
    # boxes, conf, type, size 
    b, s , c, t = [], [], [], []
    # print('out shape = {}'.format(outputs.shape))
    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))
        s.append(tf.reshape(o[3], (tf.shape(o[3])[0], -1, tf.shape(o[3])[-1])))
    
    # for i in range(0,len(b)):
    #     print('{} {} {} {}'.format(b[i].shape,s[i].shape , c[i].shape,t[i].shape))
    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)
    size = tf.concat(s, axis=1)

    scores = confidence * class_probs
    boxes, scores_nms, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(
            scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=100,
        max_total_size=100,
        iou_threshold=FLAGS.yolo_iou_threshold,
        score_threshold=FLAGS.yolo_score_threshold
    )

    selected_indices = tf.image.non_max_suppression(
        boxes=tf.reshape(bbox, [-1, 4]),
        scores=tf.reshape(scores, [-1]),
        max_output_size=100,
        iou_threshold=FLAGS.yolo_iou_threshold,
        score_threshold=FLAGS.yolo_score_threshold
    )

    bbox_size = bbox.shape[1]
        # print(tf.slice(i,[0,0,0],[0,-1,4]))
    picked_boxes, picked_score, picked_size = [], [], []
    if bbox_size is not None:
        np_bbox = bbox.numpy()[0, :, :]
        np_score = scores.numpy()[0, :, 0]
        np_size = size.numpy()[0, :, 0]
        np_selected_indices = selected_indices.numpy()
        for index in np_selected_indices:
            picked_boxes.append(np_bbox[index])
            picked_score.append(np_score[index])
            picked_size.append(np_size[index])

    valid_detections = tf.convert_to_tensor([len(picked_boxes)], tf.float32)
    picked_boxes = tf.reshape(tf.convert_to_tensor(picked_boxes, tf.float32),(1,-1,4))
    picked_size = tf.reshape(tf.convert_to_tensor(picked_size, tf.float32),(1,-1))
    picked_score = tf.reshape(tf.convert_to_tensor(picked_score, tf.float32),(1,-1))
       
    return picked_boxes, picked_size , picked_score, classes, valid_detections

def YoloV3(size=None, channels=3, anchors=yolo_anchors,
           masks=yolo_anchor_masks, classes=80, training=False):
    x = inputs = Input([size, size, channels], name='input')

    x_36, x_61, x = Darknet(name='yolo_darknet', channels=channels)(x)

    x = YoloConv(512, name='yolo_conv_0')(x)
    output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)

    x = YoloConv(256, name='yolo_conv_1')((x, x_61))
    output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

    x = YoloConv(128, name='yolo_conv_2')((x, x_36))
    output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)

    if training:
        return Model(inputs, (output_0, output_1, output_2), name='yolov3')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes),
                     name='yolo_boxes_2')(output_2)

    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='yolov3')


def YoloV3Tiny(size=None, channels=3, anchors=yolo_tiny_anchors,
               masks=yolo_tiny_anchor_masks, classes=80, training=False):
    x = inputs = Input([size, size, channels], name='input')

    x_8, x = DarknetTiny(name='yolo_darknet', channels=channels)(x)

    x = YoloConvTiny(256, name='yolo_conv_0')(x)
    output_0 = YoloOutput(256, len(masks[0]), classes, name='yolo_output_0')(x)

    x = YoloConvTiny(128, name='yolo_conv_1')((x, x_8))
    output_1 = YoloOutput(128, len(masks[1]), classes, name='yolo_output_1')(x)

    if training:
        return Model(inputs, (output_0, output_1), name='yolov3_tiny')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:4], boxes_1[:4]))
    return Model(inputs, outputs, name='yolov3_tiny')

def SingleOutput(size=None, channels=3, training=False):
    x = inputs = Input([size, size, channels])

    x = DarknetConv(x, 16, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 64, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 128, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 64, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 16, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 4, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 1, 1)
    x = MaxPool2D(2, 2, 'same')(x)
    outputs = x

    if hasattr(x, 'numpy'):
        outputs = x.numpy()
    return Model(inputs, outputs, name='single_output')

def calculateMap(yolo, model, channels, input_path):

    image_count = 0
    label_count = 0

    REs = []

    TP_total, FP_total, FN_total = 0, 0, 0

    if channels == 4:
        file_type = '.png'
    elif channels == 3:
        file_type = '.jpg'
    else :
        raise 'can\'t use other channels'

    for file_name in os.listdir(input_path):
        if file_name.find(file_type) == -1:
            continue
        image_path = input_path + file_name

        image_count += 1

        img = tf.image.decode_png(open(image_path, 'rb').read(), channels=4)
        img = tf.expand_dims(img, 0)
        img = transform_images(img, FLAGS.size)

        if model == 'single-output':
            sizes = yolo(img)
        else:
            out1, out2 = yolo(img)
            boxes, sizes, scores, classes , nums = tinyEnd(out1, out2)
        
        label_filename = file_name.replace(file_type,'.txt')
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
            if model == 'single-output':
                RE = single_value_evaluate(sizes, labels[0 , 5])
                REs.append(RE)
            else:
                outputs = (boxes, sizes , scores, classes, nums)
                grund_truth = (boxes_ture, size_ture, classes_true, labels_nums)
                TP, FP, FN, RE = yolo_extend_evaluate(outputs , grund_truth, has_size_label, FLAGS.yolo_iou_threshold)
                TP_total += TP
                FP_total += FP
                FN_total += FN
                REs.extend(RE)

    are = 1.0
    if len(REs) != 0:
        are = sum(REs) / len(REs)
    if((TP_total + FP_total) != 0):
        precision = TP_total / (TP_total + FP_total)
    else :
        precision = 0.0
    if((TP_total + FN_total) != 0):
        recall = TP_total / (TP_total + FN_total)
    else :
        recall = 0.0
    return precision, recall, are

def tinyEnd(output_0, output_1 ,anchors=yolo_tiny_anchors,
               masks=yolo_tiny_anchor_masks, classes=1):
    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:4], boxes_1[:4]))
    return outputs

@tf.function
def minus_if_zero(a, b):
    c = tf.math.subtract(a,b)
    c = c * b
    c = tf.math.divide_no_nan(c, b)
    return c

def YoloLoss(anchors, classes=80, ignore_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, size, obj, ...cls))
        # bbox, size, objectness, class_probs, pred_box
        pred_box, pred_obj, pred_class, pre_size, pred_xywh = yolo_boxes(
            y_pred, anchors, classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, size, obj, cls))
        true_box, true_sizes, true_obj, true_class_idx = tf.split(
            y_true, (4, 1, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - \
            tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh),
                           tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        true_box_flat = tf.boolean_mask(true_box, tf.cast(obj_mask, tf.bool))
        best_iou = tf.reduce_max(broadcast_iou(
            pred_box, true_box_flat), axis=-1)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # 5. calculate all losses
        xy_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        size_loss = obj_mask * box_loss_scale * \
                tf.reduce_sum(tf.square(minus_if_zero(pre_size , true_sizes)), axis=-1)            
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + \
            (1 - obj_mask) * ignore_mask * obj_loss
        # TODO: use binary_crossentropy instead
        class_loss = obj_mask * sparse_categorical_crossentropy(
            true_class_idx, pred_class)

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        size_loss = tf.reduce_sum(size_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + size_loss + obj_loss + class_loss
    return yolo_loss

def SingleSizeLoss():
    def size_loss(y_true, y_pred):
        # x = y_true[:,0]
        # y = x.reshape([8, 1, 1, 1])
        #TODO simplfy
        size_loss = tf.reduce_sum(tf.square(minus_if_zero(y_pred , y_true)), axis=-1) 
        # a = y_pred.numpy()
        # b = y_true.numpy()
        # c = size_loss.numpy()
        # print(a)
        # print(b)
        # print(c)
        # size_loss = tf.reduce_sum(size_loss, axis=(1, 2, 3))
        return size_loss
    return size_loss
