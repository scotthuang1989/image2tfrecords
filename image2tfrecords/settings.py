"""Common setting share between module."""


VALID_SPLIT_NAME = ['train', 'validation', "test"]
# These is 2 file pattern for saveing and reading
# saving

DEFAUT_SAVE_FILE_PATTERN = '%s_%s_%05d-of-%05d.tfrecord'
DEFAULT_READ_FILE_PATTERN = '%s_%s_*.tfrecord'

LABELS_FILENAME = "labels.csv"

SUMMARY_FILE_PATTERN = "%s_summary.json"
