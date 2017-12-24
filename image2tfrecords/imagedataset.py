"""
Support tensorflow Dataset API.

Usage:
"""
import json
import os

import pandas as pd
import tensorflow as tf
from tensorflow.contrib import slim
# TODO: try tf.data API. because slim is not an official API.

from .settings import (DEFAULT_READ_FILE_PATTERN, LABELS_FILENAME,
                       SUMMARY_FILE_PATTERN, VALID_SPLIT_NAME)

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer represent class of sample',
}


class ImageDataSet(object):
  """
  Read Data From tfrecords.

  Parameters:
  ------------------------------------------------------------------------------
  tfrecords_dir: wheere to find tfrecords files.
  dataset_name: name of dataset

  Attributes:
  ------------------------------------------------------------------------------
  Dataset Information:
    tfrecords_dir
    labels_df: A Dataframe map class names to class id.
    labels_to_class_names: A dict. key is class id. value is class name.
    dataset_name: dataset name
    dataset_summary: A dict contails following information.
                    class_header
                    filename_header
                    test_number
                    train_number
                    class_id_header
                    val_number
                    total_number
  """

  def __init__(self, tfrecords_dir, dataset_name=''):
    """Create a ImageDataSet."""
    self.tfrecords_dir = tfrecords_dir
    self.dataset_name = dataset_name

    # read summary information
    try:
      summary_file = os.path.join(
                    tfrecords_dir,
                    SUMMARY_FILE_PATTERN % (self.dataset_name))
      with open(summary_file) as fd:
        self.dataset_summary = json.load(fd)
    except Exception:
      raise RuntimeError("summary file don't exsits: %s" % (summary_file,))

    # read label file
    try:
      label_file = os.path.join(self.tfrecords_dir, LABELS_FILENAME)
      self.labels_df = pd.read_csv(label_file)
      self.labels_to_class_names = {}
      for i in self.labels_df.index:
        self.labels_to_class_names[
          self.labels_df.loc[i, self.dataset_summary["class_id_header"]]] =\
          self.labels_df.loc[i, self.dataset_summary["class_header"]]
    except Exception:
      raise RuntimeError("label file don't exsits: %s" % (label_file,))

  # def _has_label_file(self):
  #   return os.path.isfile(os.path.join(self.tfrecords_dir, LABELS_FILENAME))
  #
  # def _read_label_file(self):
  #   labels_df = pd.read_csv(os.path.join(self.tfrecords_dir, LABELS_FILENAME))
  #   labels_to_class_names = {}
  #   for i in labels_df.index:
  #     labels_to_class_names[labels_df.loc[i, self.dataset_summary["class_id_header"]]] =\
  #       labels_df.loc[i, self.dataset_summary["class_header"]]
  #   return labels_to_class_names

  def get_split(self, split_name, file_pattern=None):
    """
    Get a DataSet from tfrecords file.

    Parameters:
      split_name: name of split: train, validation, test
      file_pattern: pattern to find tfrecord files from directory

    Returns:
      A DataSet namedtuple
    """
    if split_name not in VALID_SPLIT_NAME:
      raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
      file_pattern = DEFAULT_READ_FILE_PATTERN
    file_pattern = os.path.join(
          self.tfrecords_dir,
          file_pattern % (self.dataset_name, split_name))

    reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
        'filename': slim.tfexample_decoder.Tensor('image/filename'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    # labels_to_names = None
    # if self._has_label_file():
    #   labels_to_names = self._read_label_file()

    sample_name_dict = {"train": "train_number",
                        "validation": "val_number",
                        "test": "test_number"}
    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=self.dataset_summary[sample_name_dict[split_name]],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=len(self.labels_to_class_names.keys()),
        labels_to_names=self.labels_to_class_names)
