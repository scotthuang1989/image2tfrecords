"""
Example usage of this module.

1. goto directory contains this lib.
2. run: python image2tfrecords/example.py

"""
from image2tfrecords.image2tfrecords import Image2TFRecords


img2tf = Image2TFRecords(
        "/home/scott/backup_data/github/kaggle_dog_breed/data/train",
        "/home/scott/backup_data/github/kaggle_dog_breed/data/labels_ext.csv",
        val_size=0.3,
        test_size=0,
        dataset_name="dog"
        )
img2tf.create_tfrecords(output_dir="/tmp/exxxabc")
