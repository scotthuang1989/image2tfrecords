"""
A example of converting cifar10 example to tfrecords.

Usage:

* Download train.7z and trainLabels.csv from kaggle: https://www.kaggle.com/c/cifar-10/data
* put them into /tmp/cifar10_data
* extract data with command: `7z x train.7z`
* Directory:/tmp/cifar10_data should look like this:
  ```
  train
  train.7z
  trainLabels.csv
  ```
  train contains all cifar10 images Convert label file to required format
"""
import math
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import slim

from image2tfrecords.image2tfrecords import Image2TFRecords
from image2tfrecords.imagedataset import ImageDataSet

CIFAR10_DATA_DIR = "/tmp/cifar10_data/train"
CIFAR10_LABELS = "/tmp/cifar10_data/trainLabels.csv"


def create_records():
    """Create tfrecords."""
    # Convert label file to required format
    # There is 1 steps
    # 1. add .png extension to filename.
    label_csv = pd.read_csv(CIFAR10_LABELS)
    # convert id column to str
    label_csv = label_csv.astype({"id": str}, copy=False)
    label_csv["id"] = label_csv["id"]+".png"

    modified_label_file = "/tmp/cifar10_data/train_labels_ext.csv"
    label_csv.to_csv(modified_label_file, index=False)

    img2tf = Image2TFRecords(
            CIFAR10_DATA_DIR,
            modified_label_file,
            val_size=0.3,
            test_size=0.1,
            dataset_name="cifar10"
            )
    img2tf.create_tfrecords(output_dir="/tmp/cifar10_data/tfrecords")


def small_net(inputs, num_class, istraining):
  """Build a small net works for test."""
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv3')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv4')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv5')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.fully_connected(net, 4096, scope='fc6')
    net = slim.dropout(net, 0.5, scope='dropout6', is_training=istraining)

    net = slim.fully_connected(net, 4096, scope='fc7')
    # net = slim.dropout(net, 0.5, scope='dropout7')
    net = slim.fully_connected(net, num_class, activation_fn=None, scope='fc8')
    net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
  return net


def batch_and_process(data_set, num_class, num_epochs=1):
    """Preprocess and bach image."""
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
                    data_set,
                    common_queue_capacity=64,
                    common_queue_min=32,
                    num_epochs=1, shuffle=False)
    image_raw, label, _ = data_provider.get(['image', 'label', 'filename'])
    # slim api require imge to be float.
    image_raw = tf.image.convert_image_dtype(image_raw, dtype=tf.float32)
    image_raw.set_shape([32, 32, 3])
    image_raw = tf.image.per_image_standardization(image_raw)

    # batch image to speed up.
    images, labels = tf.train.shuffle_batch(
                            [image_raw, label],
                            batch_size=16,
                            capacity=128,
                            min_after_dequeue=64,
                            num_threads=2)

    label_onehot = slim.one_hot_encoding(labels, num_class)
    return images, label_onehot, labels


def main():
    """Main function for train and validate."""
    tf.logging.set_verbosity(tf.logging.INFO)
    # get training data from tfrecords file.
    # Create a ImageDataSet
    with tf.Graph().as_default():
        image_dataset = ImageDataSet("/tmp/cifar10_data/tfrecords", 'cifar10')
        num_class = len(image_dataset.labels_df)
        # Get training split
        train_split = image_dataset.get_split("train")

        images, labels, _ = batch_and_process(train_split, num_class, num_epochs=None)
        # define a very simple network with slim
        predictions = small_net(images, num_class, istraining=True)
        # get loss
        slim.losses.softmax_cross_entropy(predictions, labels)
        total_loss = slim.losses.get_total_loss()
        # optimizer
        optimizer = tf.train.AdamOptimizer()
        # create training ops and train
        train_op = slim.learning.create_train_op(total_loss, optimizer)
        logdir = "/tmp/cifar10_data/train_dir"   # Where checkpoints are stored.

        slim.learning.train(
            train_op,
            logdir,
            number_of_steps=200000,
            save_summaries_secs=300,
            save_interval_secs=10,
            log_every_n_steps=10)

    # Evaluate the model
    # Load the data
    logdir = "/tmp/cifar10_data/train_dir"
    with tf.Graph().as_default():
        image_dataset = ImageDataSet("/tmp/cifar10_data/tfrecords", 'cifar10')
        num_class = len(image_dataset.labels_df)
        val_split = image_dataset.get_split("validation")
        val_images, _, val_labels = batch_and_process(val_split, num_class)
        # Define the network
        val_predictions = small_net(val_images, num_class, istraining=False)
        val_predictions = tf.argmax(val_predictions, 1)

        # Choose the metrics to compute:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': slim.metrics.streaming_accuracy(val_predictions, val_labels)
        })
        saver = tf.train.Saver()
        # Evaluate the model using 1000 batches of data:
        latest_chk = tf.train.latest_checkpoint(logdir)
        with tf.Session() as sess:
            saver.restore(sess, latest_chk)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            tf.train.start_queue_runners()

            for batch_id in range(math.floor(image_dataset.dataset_summary["val_number"]/16)):
                sess.run(names_to_updates)

            metric_values = sess.run(names_to_values)
            for metric, value in zip(names_to_values.keys(), metric_values.values()):
                print('Metric %s has value: %f' % (metric, value))


if __name__ == "__main__":
    main()
