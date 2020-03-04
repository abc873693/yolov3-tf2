from absl import logging
import numpy as np
import tensorflow as tf
import cv2
import random as random
import matplotlib.pyplot as plt
import yolov3_tf2.dataset as dataset

YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]

YOLOV3_TINY_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
]


def load_darknet_weights(model, weights_file, tiny=False):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    if tiny:
        layers = YOLOV3_TINY_LAYER_LIST
    else:
        layers = YOLOV3_LAYER_LIST

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
                    sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            logging.info("{}/{} {}".format(
                sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.input_shape[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
        (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
        (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


def draw_outputs(img, outputs, class_names):
    boxes, sizes, objectness, classes, nums = outputs
    boxes, sizes, objectness, classes, nums = boxes[0], sizes[0] ,  objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        rectangleColor = [0, 255, 0]
        end = (x1y1[0], x1y1[1] - 20)
        img = cv2.rectangle(img, x1y1, x2y2, rectangleColor, 1)
        # img = cv2.rectangle(img, x1y1, end, rectangleColor, cv2.FILLED)
        # img = cv2.putText(img, '{} {:.4f}'.format(
        #     class_names[int(classes[i])], objectness[i]),
        #     x1y1, cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(img,'{} {:.4f}%'.format(class_names[int(classes[i])], objectness[i]),
                    end, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
        cv2.putText(img,'size = {:.4f}'.format(sizes[i]),
                    x1y1, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img


def draw_labels(x, y, class_names):
    img = x.numpy()
    boxes, classes = tf.split(y, (4, 1), axis=-1)
    classes = classes[..., 0]
    wh = np.flip(img.shape[0:2])
    for i in range(len(boxes)):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, class_names[classes[i]],
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                          1, (0, 0, 255), 2)
    return img


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)

def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect))*1.0

def read_yolo_labels(label_path):
    label_text = open(label_path, "r")
    labels = []
    label_text = open(label_path, "r")
    lines = label_text.read().splitlines()
    has_size_label = True
    for text in lines:
        if text == '':
            continue
        array = text.split(' ')
        if len(array) != 0:
            class_id = int(array[0])
            center_x = float(array[1])
            center_y = float(array[2])
            width = float(array[3])
            height = float(array[4])
            if len(array) >= 6:
                size = float(array[5])
            else:
                has_size_label = False
                size = float(0.5)
            x1 = center_x - width / 2.0  # xmin
            x2 = center_x + width / 2.0  # xmax
            y1 = center_y - height / 2.0  # ymin
            y2 = center_y + height / 2.0  # ymax
            # [y1, x1, y2, x2]
            label = [class_id, x1, y1, x2, y2, size]
            labels.append(label)
    return np.array(labels), has_size_label, len(labels)

def yolo_extend_evaluate(outputs , grund_truths, iou_trethold):
    boxes, sizes, objectness, classes, nums = outputs
    boxes_ture, sizes_ture, classes_ture, nums_ture = grund_truths
    boxes, sizes, objectness, classes, nums = boxes[0].numpy(), sizes[0].numpy() , objectness[0].numpy(), classes[0].numpy(), nums[0]
    TP , FP = 0, 0
    size_errors = []
    for i in range(nums):
        max_iou = 0
        index = -1
        for j in range(nums_ture):
            iou = compute_iou(boxes[i], boxes_ture[j])
            if iou > max_iou and iou >= iou_trethold and classes[i] == classes_ture[j]:
                max_iou = iou
                index = j
        if index == -1:
            FP += 1
        else:
            TP += 1
            if(len(sizes_ture) == 0):
                size_ralative_error = 1.0
            else:
                size_ralative_error = abs(sizes[i] - sizes_ture[index]) / sizes_ture[index]
            # logging.info('iou = {}  boxes_pre = {} boxes_ture = {}'.format(max_iou, boxes[i],boxes_ture[index]))
            # logging.info('size_pre = {} size_ture = {} size_ralative_error = {}'.format(sizes[i],sizes_ture[index], size_ralative_error))
            size_errors.append(size_ralative_error)
    are = 0
    if len(size_errors) != 0:
        are = sum(size_errors)/ len(size_errors)
    # logging.info('TP = {} FP = {}, size average ralative error = {}'.format(TP, FP, are))
    return TP, FP, (nums_ture - TP), size_errors

def save_loss_plot(train_loss = [], val_loss = [], precision = [],save_path = './loss.png'):
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(save_path, dpi=100)
    plt.clf()
        
def randomHSV(images, saturation, exposure, hue):
    hsv_images = tf.image.rgb_to_hsv(images)
    size = hsv_images.shape[0]
    rand_list = []
    for i in range(0, size):
        d_saturation = random.uniform( 1/saturation, saturation)
        d_exposure = random.uniform( 1/exposure, exposure)
        rand_list.append([1.0, d_saturation, d_exposure])
    rand_tensor = tf.convert_to_tensor(rand_list)
    hsv_images = tf.einsum('ijkl,il->ijkl', images, rand_tensor)
    images = tf.image.hsv_to_rgb(hsv_images)
    images = tf.image.random_hue(image = images, max_delta = hue)
    return images

def cropImages(images, labels, jitter, anchors, anchor_masks):
    ow = images.shape[2]
    oh = images.shape[1]
    dw = (ow*jitter)
    dh = (oh*jitter)
    
    pleft  = random.uniform(-dw, dw)
    pright = random.uniform(-dw, dw)
    ptop   = random.uniform(-dh, dh)
    pbot   = random.uniform(-dh, dh)
    
    swidth =  ow - pleft - pright
    sheight = oh - ptop - pbot
    
    sx = uniform_size(pleft, ow)
    sy = uniform_size(ptop, oh)
    ex = uniform_size(swidth + pleft, ow)
    ey = uniform_size(sheight + ptop, oh)

    images = tf.image.crop_to_bounding_box(
        images,
        int(sx),
        int(sy),
        int(ex) - int(sx),
        int(ey) - int(sy)
    )
    
    flip = random.random() %2

    labels = fill_truth_detection(labels, sx, sy, ex, ey, ow, oh, flip, anchors, anchor_masks)

    images = tf.image.resize(
        images,
        (oh,ow)
    )

    if(flip == 1):
        images = tf.image.flip_left_right(images)

    return images

def uniform_size(target, origin):
    if(target > origin):
        return origin
    if(target < 0):
        return 0
    return target

def fill_truth_detection(labels, sx, sy, ex, ey, ow, oh, flip, anchors, anchor_masks):
    nw = ex - sx
    nh = ey - sy
    
    asw = nw / ow
    ash = nh / oh

    sx_nor = sx / ow
    sy_nor = sy / oh

    label1 , label2 = labels

    num_max = tf.constant(0.999)
    num_min = tf.constant(0.)

    batch_data = []
    for batch in range(0, label1.shape[0]): #batch
        boxes = []
        for i in range(0, label1.shape[1]): #h
            for j in range(0, label1.shape[2]): #w
                for k in range(0, label1.shape[3]):
                    box = label1[batch][i][j][k]
                    nps = box.numpy()
                    is_all_zero = not np.any(nps)
                    if not is_all_zero:
                        x1 = box[0].numpy()
                        y1 = box[1].numpy()
                        x2 = box[2].numpy()
                        y2 = box[3].numpy()
                        x1 = min(0.999, max(0, x1 * asw - sx_nor)) 
                        y1 = min(0.999, max(0, y1 * ash - sy_nor)) 
                        x2 = min(0.999, max(0, x2 * asw - sx_nor))
                        y2 = min(0.999, max(0, y2 * ash - sy_nor))
                        # x1_tf = tf.math.minimum(num_max, tf.math.maximum(num_min, box[0] * asw - sx_nor))
                        # y1_tf = tf.math.minimum(num_max, tf.math.maximum(num_min, box[1] * ash - sy_nor))
                        # x2_tf = tf.math.minimum(num_max, tf.math.maximum(num_min, box[2] * asw - sx_nor))
                        # y2_tf = tf.math.minimum(num_max, tf.math.maximum(num_min, box[3] * ash - sy_nor))
                        w = float(x2 - x1)
                        h = float(y2 - y1)
                        if w < 0.001 or h < 0.001:
                            continue
                        else:
                            boxes.append([x1, y1, x2, y2, box[4].numpy(), box[5].numpy()])
                            # print(box)
        for i in range(0, label2.shape[1]): #h
            for j in range(0, label2.shape[2]): #w
                for k in range(0, label2.shape[3]):
                    box = label2[batch][i][j][k]
                    nps = box.numpy()
                    is_all_zero = not np.any(nps)
                    if not is_all_zero:
                        x1 = box[0].numpy()
                        y1 = box[1].numpy()
                        x2 = box[2].numpy()
                        y2 = box[3].numpy()
                        x1 = min(0.999, max(0, x1 * asw - sx_nor)) 
                        y1 = min(0.999, max(0, y1 * ash - sy_nor)) 
                        x2 = min(0.999, max(0, x2 * asw - sx_nor))
                        y2 = min(0.999, max(0, y2 * ash - sy_nor))
                        w = float(x2 - x1)
                        h = float(y2 - y1)
                        if w < 0.001 or h < 0.001:
                            continue
                        else:
                            if flip % 2 == 0:
                                x1 = 0.999 - x1
                                x2 = 0.999 - x2
                            boxes.append([x1, y1, x2, y2, box[4].numpy(), box[5].numpy()])
                            #print(box)
        batch_data.append(boxes)
    batch_data_pad = []
    for batch in batch_data:
        if len(batch) == 0:
            batch.append([0, 0, 0, 0, 0, 0])
        paddings = [[0, 100 - len(batch)], [0, 0]]
        batch_tf = tf.convert_to_tensor(batch)
        batch_tf = tf.pad(batch_tf, paddings)
        boxes = batch_tf.numpy()
        batch_data_pad.append(boxes)
    labels_np = np.array(batch_data_pad)
    labels_tf = tf.convert_to_tensor(labels_np, dtype = tf.float32)
    labels = dataset.transform_targets(labels_tf, anchors, anchor_masks, 1)
    return labels