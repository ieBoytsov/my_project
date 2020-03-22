import errno
import os


def mkdir_if_missing(dir_path: str):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
