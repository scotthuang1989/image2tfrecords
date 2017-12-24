"""
A lib create tfrecords file from images and create dataest from tfrecords.

Introduction
------------------------------
Features:

* Create tfrecords from images.
    * split the images with stratified strategy.
* Provide a interface for tensorflow DataSet API.
"""
from os import path

from . import imagedataset
from . import image2tfrecords


here = path.abspath(path.dirname(__file__))

parent_path = path.dirname(here)

with open(path.join(parent_path, 'VERSION')) as version_file:
    version = version_file.read().strip()

__version__ = version
