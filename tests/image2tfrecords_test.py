"""
Test case for main class.

How to run:
1. Go to the directory contains this module
2. run: python -m unittest image2tfrecords/testcase/image2tfrecords_test.py
"""
import unittest
from unittest import TestCase
from image2tfrecords.image2tfrecords import Image2TFRecords

test_image_class_file = "image2tfrecords/testcase/image_class.csv"


class Test_Image2TFRecords(TestCase):
  def test_map_file(self):
    """Test if create map file correctly."""
    test_converter = Image2TFRecords("", test_image_class_file)
    image2class, class2id = test_converter._create_class_map()

    self.assertEqual(test_converter.filename_header, "test_file_header")
    self.assertEqual(test_converter.class_header, "test_class")
    self.assertEqual(test_converter.class_id_header, "test_class_id")
    self.assertEqual(len(class2id.index), 11)

  def test_create_and_read_tfrecord(self):
      pass
      #TODO: add testcase


if __name__ == '__main__':
  unittest.main()
