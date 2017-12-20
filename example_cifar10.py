"""
A example of converting cifar10 example to tfrecords.

Usage:

* Download train.7z and trainLabels.csv from kaggle: https://www.kaggle.com/c/cifar-10/data
* put them into /tmp/cifar10_data
* extract data with command: `7z x train.7z`
* your directory should look like this:
  ```
  ├── train
  ├── train.7z
  └── trainLabels.csv
  ```
  train dir contains all cifar10 images
"""
import pandas as pd

from image2tfrecords.image2tfrecords import Image2TFRecords

CIFAR10_DATA_DIR = "/tmp/cifar10_data/train"
CIFAR10_LABELS = "/tmp/cifar10_data/trainLabels.csv"

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


## TODO: use tfrecord to train a model? train/val/test
