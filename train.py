from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import cv2
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks, calculateMap

)
from yolov3_tf2.utils import (
    freeze_all,
    save_loss_plot,
    randomHSV,
    cropImages
)
import yolov3_tf2.dataset as dataset

import os

flags.DEFINE_string('dataset', './data/shrimp_v6/train.tfrecord', 'path to dataset')
flags.DEFINE_string('val_dataset', './data/shrimp_v6/valid.tfrecord', 'path to validation dataset')
flags.DEFINE_string('test_dataset_path', './data/microfield_monocular_test/', 'path to test dataset')
flags.DEFINE_boolean('tiny', True, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('name', '20200304_1',
                    'path to weights name')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_string('classes', './data/shrimp.names', 'path to classes file')
flags.DEFINE_enum('mode', 'eager_tf', ['fit', 'eager_fit', 'eager_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')
flags.DEFINE_enum('transfer', 'none',
                  ['none', 'darknet', 'no_output', 'frozen', 'fine_tune'],
                  'none: Training from scratch, '
                  'darknet: Transfer darknet, '
                  'no_output: Transfer all but output, '
                  'frozen: Transfer and freeze all, '
                  'fine_tune: Transfer all and freeze darknet only')
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('epochs', 100, 'number of epochs')
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_integer('num_classes', 1, 'number of classes in the model')
flags.DEFINE_integer('weights_num_classes', None, 'specify num class for `weights` file if different, '
                     'useful in transfer learning with different number of classes')

#python3 train.py --batch_size 8 --dataset ./microfield_4_train.tfrecord --val_dataset ~/Data/voc2012_val.tfrecord --epochs 10 --mode eager_fit --transfer fine_tune --weights ./checkpoints/yolov3-tiny.tf --tiny
def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    weight_base_dir = 'checkpoints/{}/'.format(FLAGS.name)
    if not os.path.exists(weight_base_dir):
        os.makedirs(weight_base_dir)

    if FLAGS.tiny:
        model = YoloV3Tiny(FLAGS.size, training=True,
                           classes=FLAGS.num_classes)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        model = YoloV3(FLAGS.size, training=True, classes=FLAGS.num_classes)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks

    # train_dataset = dataset.load_fake_dataset()
    # train_dataset = dataset.load_text_dataset('dataset/microfield_4/416X416/cfg/train.txt')
    if FLAGS.dataset:
        train_dataset = dataset.load_tfrecord_dataset(
            FLAGS.dataset, FLAGS.classes)
    train_dataset = train_dataset.shuffle(buffer_size=1024)  # TODO: not 1024
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, 1)))
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    # val_dataset = dataset.load_fake_dataset()
    # val_dataset = dataset.load_text_dataset('dataset/microfield_4/416X416/cfg/valid.txt')
    if FLAGS.val_dataset:
        val_dataset = dataset.load_tfrecord_dataset(
            FLAGS.val_dataset, FLAGS.classes)
    val_dataset = val_dataset.batch(FLAGS.batch_size)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, 1)))

        # Configure the model for transfer learning
    if FLAGS.transfer == 'none':
        pass  # Nothing to do
    elif FLAGS.transfer in ['darknet', 'no_output']:
        # Darknet transfer is a special case that works
        # with incompatible number of classes

        # reset top layers
        if FLAGS.tiny:
            model_pretrained = YoloV3Tiny(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
        else:
            model_pretrained = YoloV3(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
        model_pretrained.load_weights(FLAGS.weights)

        if FLAGS.transfer == 'darknet':
            model.get_layer('yolo_darknet').set_weights(
                model_pretrained.get_layer('yolo_darknet').get_weights())
            freeze_all(model.get_layer('yolo_darknet'))

        elif FLAGS.transfer == 'no_output':
            for l in model.layers:
                if not l.name.startswith('yolo_output'):
                    l.set_weights(model_pretrained.get_layer(
                        l.name).get_weights())
                    freeze_all(l)

    else:
        # All other transfer require matching classes
        model.load_weights(FLAGS.weights)
        if FLAGS.transfer == 'fine_tune':
            # freeze darknet and fine tune other layers
            darknet = model.get_layer('yolo_darknet')
            freeze_all(darknet)
        elif FLAGS.transfer == 'frozen':
            # freeze everything
            freeze_all(model)

    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    # step = tf.Variable(0, trainable=False)
    # schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
    #     [100000, 112500], [1e-0, 1e-1, 1e-2])
    # # lr and wd can be a function or a tensor
    # lr = 1e-4  * schedule(step)
    # wd = lambda: 5e-4 * schedule(step)

    # optimizer = tfa.optimizers.SGDW(
    #     learning_rate=lr, weight_decay=wd)
        
    loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes)
            for mask in anchor_masks]

    if FLAGS.mode == 'eager_tf':
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

        loss_records = []
        loss_val_records = []
        precision_records = []
        recall_records = []
        relative_error_records = []
        
        loss_best = 10000000.0
        loss_best_epochs = 0
        loss_val_best = 10000000.0
        loss_val_best_epochs = 0
        
        for epoch in range(1, FLAGS.epochs + 1):
            for batch, (images, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    # images = cropImages(images, labels, 0.3, anchors, anchor_masks)
                    # images = tf.image.random_brightness(images, 0.1)  # 隨機亮度
                    # images = tf.image.random_saturation(images, 0.7, 1.3)  # 隨機飽和度
                    # images = tf.image.random_contrast(images, 0.6, 1.5)  # 隨機對比度
                    images = tfa.image.random_hsv_in_yiq(
                        images,
                        max_delta_hue=0.1,
                        lower_saturation=1/1.5,
                        upper_saturation=1.5,
                        lower_value=1/1.5,
                        upper_value=1.5,
                    )
                    # images = randomHSV(images, 1.5, 1.5, 0.3)
                    outputs = model(images, training=True)
                    regularization_loss = tf.reduce_sum(model.losses)
                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))

                logging.info("{}_train_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_loss.update_state(total_loss)

            for batch, (images, labels) in enumerate(val_dataset):
                # images = cropImages(images, labels, 0.3, anchors, anchor_masks)
                # images = tf.image.random_brightness(images, 0.1)  # 隨機亮度
                # images = tf.image.random_saturation(images, 0.7, 1.3)  # 隨機飽和度
                # images = tf.image.random_contrast(images, 0.6, 1.5)  # 隨機對比度
                images = tfa.image.random_hsv_in_yiq(
                    images,
                    max_delta_hue=0.1,
                    lower_saturation=1/1.5,
                    upper_saturation=1.5,
                    lower_value=1/1.5,
                    upper_value=1.5,
                )
                # images = randomHSV(images, 1.5, 1.5, 0.3)
                outputs = model(images)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                for output, label, loss_fn in zip(outputs, labels, loss):
                    pred_loss.append(loss_fn(label, output))
                total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                logging.info("{}_val_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_val_loss.update_state(total_loss)

            train_loss = avg_loss.result().numpy()
            val_loss = avg_val_loss.result().numpy()

            loss_records.append(train_loss)
            loss_val_records.append(val_loss)

            precision, recall, ARE = calculateMap(model, FLAGS.test_dataset_path)
            
            precision_records.append(precision)
            recall_records.append(recall)
            relative_error_records.append(ARE)

            logging.info("{}, train: {}, val: {}, precision: {}, recall: {}, relative error: {}".format(
                epoch,
                train_loss,
                val_loss,
                precision,
                recall,
                ARE))
            
            loss_text = open('{}loss.txt'.format(weight_base_dir), "a")
            loss_text.write('{},{},{},{},{}\n'.format(train_loss,
                val_loss,
                precision,
                recall,
                ARE))
            loss_text.close()

            if train_loss < loss_best:
                loss_best = train_loss
                loss_best_epochs = epoch
                model.save_weights(
                    '{}yolov3_train_best.tf'.format(weight_base_dir))

            if val_loss < loss_val_best:
                loss_val_best = val_loss
                loss_val_best_epochs = epoch
                model.save_weights(
                    '{}yolov3_train_val_best.tf'.format(weight_base_dir))

            avg_loss.reset_states()
            avg_val_loss.reset_states()
            if epoch % 10 == 0:
                model.save_weights(
                    '{}yolov3_train_{}.tf'.format(weight_base_dir, epoch))
            save_loss_plot(train_loss = loss_records,val_loss = loss_val_records,precision=precision_records, recall = recall_records, are = relative_error_records, save_path='{}loss.png'.format(weight_base_dir))
            model.save_weights(
                    '{}yolov3_train_last.tf'.format(weight_base_dir))
            if epoch == FLAGS.epochs:
                logging.info("{}, loss_best: {} {}, loss_val_best: {} {}".format(
                    epoch,
                    loss_best_epochs,
                    loss_best,
                    loss_val_best_epochs,
                    loss_val_best))
    else:
        model.compile(optimizer=optimizer, loss=loss,
                      run_eagerly=(FLAGS.mode == 'eager_fit'))

        callbacks = [
            ReduceLROnPlateau(verbose=1),
            EarlyStopping(patience=3, verbose=1),
            ModelCheckpoint(weight_base_dir + 'yolov3_train_{epoch}.tf',
                            verbose=1, save_weights_only=True),
            TensorBoard(log_dir='logs')
        ]

        history = model.fit(train_dataset,
                            epochs=FLAGS.epochs,
                            callbacks=callbacks,
                            validation_data=val_dataset)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass