import os

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.contrib import slim

from image2tfrecords.imagedataset import ImageDataSet

test_case_data_dir = "/home/scott/backup_data/github/kaggle_dog_breed/data/train"
test_case_label_file = "/home/scott/backup_data/github/kaggle_dog_breed/data/labels_ext.csv"
label_df = pd.read_csv(test_case_label_file)


image_dataset = ImageDataSet("/tmp/exxxabc", 'dog')

train_split = image_dataset.get_split("train")

data_provider = slim.dataset_data_provider.DatasetDataProvider(
                train_split,
                common_queue_capacity=64,
                common_queue_min=32,
                num_epochs=1, shuffle=False)
image_raw, label, filename = data_provider.get(['image', 'label', 'filename'])

sess = tf.Session()

train_number = image_dataset.dataset_summary["train_number"]

global_init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()
with sess.as_default() as sess:
  sess.run([global_init, local_init])
  tf.train.start_queue_runners()
  for i in range(train_number):
    r_image_raw, r_label, r_filename = sess.run([image_raw, label, filename])

    r_filename = r_filename.decode()
    # read the image directly from image file to verify
    image_file = os.path.join(test_case_data_dir, r_filename)
    verify_image_raw = tf.gfile.FastGFile(image_file, 'rb').read()
    verify_image_raw_decoder = tf.image.decode_image(verify_image_raw)
    verify_image = sess.run(verify_image_raw_decoder)

    # read class directly from label file
    verify_label = label_df[label_df["id"] == r_filename]["breed"].tolist()[0]
    verify_label = verify_label.strip()
    if np.array_equal(r_image_raw, verify_image) and\
            verify_label == image_dataset.labels_to_class_names[r_label]:
      print("\r verifiying %d / %d" % (i+1, train_number))
    else:
      pass
