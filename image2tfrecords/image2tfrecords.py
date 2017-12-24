"""
Create tfrecords file from images.

Usage:

from image2tfrecords import Image2TFRecords

img2tf = Image2TFRecords(
        image_dir,
        label_dir,
        val_size=0.2,
        test_size=0.1
        )
img2tf.create_tfrecords(output_dir="/tmp/exxx")
"""
import os
import sys
import math
import json

import pandas as pd
import tensorflow as tf

from .settings import VALID_SPLIT_NAME, DEFAUT_SAVE_FILE_PATTERN
from .settings import LABELS_FILENAME, SUMMARY_FILE_PATTERN


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  # TODO: need save the label file. and information that tell size of each split
  #
  def __init__(self):
    """Create a Reader for reading image information."""
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_image(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    """Read dimension of image."""
    image = self.decode_image(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_image(self, sess, image_data):
    """Decode jpeg image."""
    # TODO: only support jpg format. add other formate support.
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


class Image2TFRecords(object):
  """
  Convert images and into tfrecords.

  Parameters:
  ------------------------------------------------------------------------------
  image_path: path where you put your images
  image2class_file: csv file which contains classname for every image file
      Format:
          filename, class
          1.jpg   , cat
          2.jpg   , dog
          ..     , ..
      you must provide a valid header for csv file. Headers will be used in
      tfrecord to represent dataset-specific information.
  dataset_name: the resuliting tfrecord file will have following name:
        dataset_name_splitname_xxxxx_of_xxxxx.tfrecords
  class2id_file: csv file which contains classid for every class in image2class_file
      Format:
          classname, class_id
          1.jpg   , cat
          2.jpg   , dog
          ..     , ..
      you must provide a valid header for csv file. Headers will be used in
      tfrecord to represent dataset-specific information.
  val_size: percentage for validation dataset. if you don't want split your data
            into train/validation/test. set it to 0
  test_size: same as val_size
  num_shards: The number of shards per dataset split.

  Attributes:
  ------------------------------------------------------------------------------
    Dataset Information:
        class_header
        class_id_header
        filename_header:
                        header names for class,
                        class_id, image file name column.
        image_path: directory path for finding images.
        image2class_file: map file from image names to class.
        image2class: A DataFrame of mapping image names to class names.
                     content of this Df is from image2class_file.
        class2id_file: map file from class name to class id.
        class2id: A DataFrame of mapping class names to class id.
                  content of this Df is either form class2id_file or build
                  from existing information on this dataset.
        dataset_name: name of dataset.
    Train/Validation/Test split:
        val_number
        test_number
        train_number
        total_number
        val_size
        test_size

    Others:
        num_shards: number of tfrecords file for each split.
  """

  def __init__(self,
               image_path,
               image2class_file,
               class2id_file=None,
               dataset_name='',
               val_size=0,
               test_size=0,
               num_shards=5):
    """
    Construtor of Image2TFRecords.

    only image_path and image2class_file is mandantory
    """
    # TODO: add path validation. check valid image exists
    # current support image formate: bmp, gif, jpeg,_png
    # Parameter validation
    if (val_size < 0 or val_size > 1) or\
       ((test_size < 0 or test_size > 1)) or\
       (val_size+test_size > 1):
      raise RuntimeError("val_size and test_size must between 0 and 1 and Their \
                         sum can't exceed 1")

    self.image_path = image_path
    # TODO: check map file format
    self.image2class_file = image2class_file
    self.class2id_file = class2id_file
    self.dataset_name = dataset_name
    self.val_size = val_size
    self.test_size = test_size
    self.num_shards = num_shards

    self.dataset_summary = {}

    # create class image_path
    self._create_class_map()
    # after create class map. we can get total number of samples
    self.total_number = len(self.image2class)
    self.dataset_summary["total_number"] = self.total_number

  def _save_summary_file(self, output_dir):
    summary_file = os.path.join(
                  output_dir,
                  SUMMARY_FILE_PATTERN % (self.dataset_name,))
    with open(summary_file, 'w') as fd:
      json.dump(self.dataset_summary, fd)
    print("write summary file done")

  def _write_class2id_file(self, output_dir):
    self.class2id.to_csv(
                  os.path.join(output_dir, LABELS_FILENAME),
                  index=False)
    print("write label file done")

  def _create_class_map(self):
    # 1. first read image2class_file
    self.image2class = pd.read_csv(self.image2class_file)
    # require filename at 1st column and class at 2nd will simplify the parameters
    # but may require use do more pre-process. which is better?
    self.filename_header = self.image2class.columns[0]
    self.class_header = self.image2class.columns[1]

    # 1.1 process image2class. strip padding space
    def strip_space(data):
      return pd.Series([d.strip() for d in data])

    self.image2class = self.image2class.apply(strip_space, axis=0)

    # 2. then check if there is: class 2 class_id file.
    # yes: read it
    # no: create one
    if self.class2id_file:
      self.class2id = pd.read_csv(self.class2id_file)
      self.class_id_header = self.class2id.columns[1]
    else:
      self.class_id_header = self.class_header+"_id"
      self.class2id = pd.DataFrame(columns=[self.class_header,
                                            self.class_id_header])
      id_count = 0
      for col in self.image2class[self.class_header]:
        if not (col in self.class2id[self.class_header].tolist()):
          self.class2id = pd.concat(
                [self.class2id,
                 pd.DataFrame({self.class_header: [col],
                              self.class_id_header: [id_count]})
                 ])
          id_count += 1
      self.class2id = self.class2id.reset_index(drop=True)
    # save header information to disk
    self.dataset_summary["filename_header"] = self.filename_header
    self.dataset_summary["class_header"] = self.class_header
    self.dataset_summary["class_id_header"] = self.class_id_header

    return self.image2class, self.class2id

  def _get_dataset_filename(self, split_name, shard_id, output_dir):
    output_filename = DEFAUT_SAVE_FILE_PATTERN % (
                      self.dataset_name,
                      split_name,
                      shard_id+1,
                      self.num_shards)
    if output_dir:
      return os.path.join(output_dir, output_filename)
    else:
      return output_filename

  def _int64_feature(self, values):
    """Return a TF-Feature of int64s.

    Args:
      values: A scalar or list of values.

    Returns:
      A TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
      values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

  def _bytes_feature(self, values):
    """Return a TF-Feature of bytes.

    Args:
      values: A string.

    Returns:
      A TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

  def _float_feature(self, values):
    """Return a TF-Feature of floats.

    Args:
      values: A scalar of list of values.

    Returns:
      A TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
      values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

  def _image_to_tfexample(self,
                          image_data,
                          image_format,
                          height,
                          width,
                          class_id,
                          filename):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': self._bytes_feature(image_data),
        'image/format': self._bytes_feature(image_format),
        'image/class/label': self._int64_feature(class_id),
        'image/height': self._int64_feature(height),
        'image/width': self._int64_feature(width),
        'image/filename': self._bytes_feature(filename)
    }))

  def _convert_dataset(self, split_name, image_index, output_dir):
    """Convert the images of give index into .

    Args:
      split_name: The name of the dataset, either 'train', 'validation' or "test"
      image_index: The index used to select image from image2class dataframe.
    """
    assert split_name in VALID_SPLIT_NAME
    assert len(image_index) > 0

    num_per_shard = int(math.ceil(len(image_index) / float(self.num_shards)))

    with tf.Graph().as_default():
      image_reader = ImageReader()

      with tf.Session('') as sess:
        # TODO: shards have problem, if total number of dataset is too small.
        for shard_id in range(self.num_shards):
          output_filename = self._get_dataset_filename(split_name, shard_id, output_dir)

          with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id+1) * num_per_shard, len(image_index))
            for i in range(start_ndx, end_ndx):
              sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                  i+1, len(image_index), shard_id))
              sys.stdout.flush()
              # Read the filename:
              image_filename = self.image2class.loc[image_index[i], self.filename_header]
              image_data = tf.gfile.FastGFile(
                os.path.join(self.image_path, image_filename),
                'rb').read()
              height, width = image_reader.read_image_dims(sess, image_data)

              class_name = self.image2class.loc[image_index[i], self.class_header]

              class_id = self.class2id[
                        self.class2id[self.class_header] == class_name][self.class_id_header]
              # at this step, class_id is a Series with only 1 element. convert it to int
              class_id = int(class_id)
              image_format = os.path.splitext(image_filename)[1][1:]
              example = self._image_to_tfexample(
                  image_data,
                  image_format.encode('utf-8'),
                  height,
                  width,
                  class_id,
                  image_filename.encode('utf-8'))
              tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()

  def create_tfrecords(self, output_dir):
    """
    Create tfrecord.

    Parameters:
      output_dir: Where to put the tfrecords file.
    """
    if not os.path.isdir(output_dir):
      os.makedirs(output_dir)

    train_index = []
    val_index = []
    test_index = []

    def draw_sample(df_class):
      # split data into 3 split
      # 1. calculate number of each split
      total_number = len(df_class.index)

      test_number = math.floor(total_number*self.test_size)
      val_number = math.floor(total_number*self.val_size)
      train_number = total_number - test_number - val_number
      # because I use floor when I calculate test and val size.
      # There is a chance that train_number is 1 but
      # self.test_size + self.val_number == 1
      # for example:
      # total: 99, eval=0.1, test=0.9
      #            9.9->9     89.1->89
      # in this case. add this train_number to test set
      if train_number == 1:
        train_number = 0
        test_number += 1

      if val_number > 0:
        t_val_index = df_class.sample(val_number).index
        df_class = df_class.drop(t_val_index)
        val_index.extend(t_val_index)

      if test_number > 0:
        t_test_index = df_class.sample(test_number).index
        df_class = df_class.drop(t_test_index)
        test_index.extend(t_test_index)

      if train_number:
        t_train_index = df_class.index
        train_index.extend(t_train_index)

    # self.image2class.groupby(self.class_header).apply(draw_sample)
    for name, group in self.image2class.groupby(self.class_header):
      draw_sample(group)

    self.train_number = len(train_index)
    self.val_number = len(val_index)
    self.test_number = len(test_index)

    assert((self.train_number + self.val_number + self.test_number) == self.total_number)

    # def _convert_dataset(self, split_name, image_index, output_dir)
    if self.train_number:
      self._convert_dataset("train", train_index, output_dir)

    if self.val_number:
      self._convert_dataset("validation", val_index, output_dir)

    if self.test_number:
      self._convert_dataset("test", test_index, output_dir)

    self.dataset_summary["train_number"] = self.train_number
    self.dataset_summary["val_number"] = self.val_number
    self.dataset_summary["test_number"] = self.test_number

    # write summary file
    self._save_summary_file(output_dir)
    # write lable file
    self._write_class2id_file(output_dir)
