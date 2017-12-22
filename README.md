# image2tfrecords
--------

Convert images into tfrecord to comply with tensorflow best practice: [tensorflow doc link](https://tensorflow.google.cn/performance/performance_guide#input_pipeline_optimization).

# Supported platform

* OS
  * ubuntu


* Python
  * python3


* tensorflow
  * 1.4

# Installation

`pip install image2tfrecords`

# Features

* Stratified split between train/validation/test: so each split have same percentage of each class.

* Tensorflow Dataset API support: Provide a Class that read tfrecords files and return a Dataset, so developers can easily build tensorflow program with images.


# Tutorial

This simple tutorial will work you through creating cifar10 tfrecords for kaggle competition. yo can check `example_cifar10.py` for full code.

### Download cifar10 data.

* Download train.7z and trainLabels.csv from kaggle: https://www.kaggle.com/c/cifar-10/data
* put them into /tmp/cifar10_data
* extract data with command: `7z x train.7z`
* Directory:/tmp/cifar10_data should look like this:
  ```
  ├── train
  ├── train.7z
  └── trainLabels.csv
  ```
  "train" contains all cifar10 images

### Convert label file(a file list all images and its corresponding labels)

Because this module requires label file with following format:

```
image2class_file: csv file which contains classname for every image file
    Format:
        filename, class
        1.jpg   , cat
        2.jpg   , dog
        ..     , ..
    you must provide a valid header for csv file. Headers will be used in
    tfrecord to represent dataset-specific information.
```
First, we need convert cifar10 label file to this format:

```
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
```
After this step. the new label file is `/tmp/cifar10_data/train_labels_ext.csv`

### Create tfrecords

since we have what we need, now pass the image directory and label flle path to  **Image2TFRecords** and create the tfrecords
```
img2tf = Image2TFRecords(
        CIFAR10_DATA_DIR,
        modified_label_file,
        val_size=0.3,
        test_size=0.1,
        dataset_name="cifar10"
        )
img2tf.create_tfrecords(output_dir="/tmp/cifar10_data/tfrecords")
```
After run this, the tfrecords file will be at: `/tmp/cifar10_data/tfrecords`, it will looks like this:

```
-rw-rw-r-- 1 scott scott  169 Dec 21 10:57 cifar10_summary.json
-rw-rw-r-- 1 scott scott 2.3M Dec 21 10:57 cifar10_test_00001-of-00005.tfrecord
-rw-rw-r-- 1 scott scott 2.3M Dec 21 10:57 cifar10_test_00002-of-00005.tfrecord
-rw-rw-r-- 1 scott scott 2.4M Dec 21 10:57 cifar10_test_00003-of-00005.tfrecord
-rw-rw-r-- 1 scott scott 2.5M Dec 21 10:57 cifar10_test_00004-of-00005.tfrecord
-rw-rw-r-- 1 scott scott 2.3M Dec 21 10:57 cifar10_test_00005-of-00005.tfrecord
-rw-rw-r-- 1 scott scott  14M Dec 21 10:56 cifar10_train_00001-of-00005.tfrecord
-rw-rw-r-- 1 scott scott  14M Dec 21 10:56 cifar10_train_00002-of-00005.tfrecord
-rw-rw-r-- 1 scott scott  15M Dec 21 10:56 cifar10_train_00003-of-00005.tfrecord
-rw-rw-r-- 1 scott scott  15M Dec 21 10:56 cifar10_train_00004-of-00005.tfrecord
-rw-rw-r-- 1 scott scott  14M Dec 21 10:57 cifar10_train_00005-of-00005.tfrecord
-rw-rw-r-- 1 scott scott 6.8M Dec 21 10:57 cifar10_validation_00001-of-00005.tfrecord
-rw-rw-r-- 1 scott scott 6.9M Dec 21 10:57 cifar10_validation_00002-of-00005.tfrecord
-rw-rw-r-- 1 scott scott 7.1M Dec 21 10:57 cifar10_validation_00003-of-00005.tfrecord
-rw-rw-r-- 1 scott scott 7.4M Dec 21 10:57 cifar10_validation_00004-of-00005.tfrecord
-rw-rw-r-- 1 scott scott 6.9M Dec 21 10:57 cifar10_validation_00005-of-00005.tfrecord
-rw-rw-r-- 1 scott scott   95 Dec 21 10:57 labels.csv

```

### Train a model with these tfrecords file.

check the example_cifar10.py for full code.

But the key step is creating a `ImageDataSet`:

```
image_dataset = ImageDataSet("/tmp/cifar10_data/tfrecords", 'cifar10')
num_class = len(image_dataset.labels_df)
val_split = image_dataset.get_split("validation")
val_images, _, val_labels = batch_and_process(val_split, num_class)
```
After get a tensorflow DataSet. The most common practice is pass it to `DatasetDataProvider`, then get the data from this `DatasetDataProvider`.

```
data_provider = slim.dataset_data_provider.DatasetDataProvider(
                data_set,
                common_queue_capacity=64,
                common_queue_min=32,
                num_epochs=1, shuffle=False)

image_raw, label, _ = data_provider.get(['image', 'label', 'filename'])
```



# API Intruduction

This should be generated from comments automatically.
