from PIL import Image, ImageOps
from scipy.ndimage.interpolation import zoom
import numpy as np
import os
import sys
from collections import OrderedDict


def get_ext(filename):
    name, ext = os.path.splitext(filename)
    return ext


def load_image(filename):
    img = Image.open(filename)
    return np.array(img)


def load_and_fit_image(filename, shape, method=Image.NEAREST):
    img = Image.open(filename)
    fitted = ImageOps.fit(img, shape[::-1], method=method)
    return np.array(fitted)


def load_file(filename):
    file = open(filename, "rb").read()
    return file


def load_file_chunked(filename, chunk_size=1024):
    file = open(filename, "rb")
    """Lazy function (generator) to read a file piece by piece.
    Default chunk size: 1k."""
    while True:
        data = file.read(chunk_size)
        if not data:
            break
        yield data


def get_directory_filenames(data_dir, extensions=False):
    return [f for f in os.listdir(data_dir) if not extensions or (extensions and get_ext(f) in extensions)]


def get_class_file_map(data_dir, extensions=False):
    sys.stdout.flush()
    classes = [cls for cls in os.listdir(
        data_dir) if os.path.isdir(data_dir + "/" + cls)]
    classes.sort()
    n_classes = len(classes)

    assert n_classes != 0, "get_class_file_map: No classes found in directory: " + data_dir

    file_map = {}
    for cls in classes:
        file_map[cls] = [cls + "/" + f for f in os.listdir(
            data_dir + "/" + cls) if not extensions or (extensions and get_ext(f) in extensions)]

    return file_map, n_classes, classes


class LimitedSizeDict(OrderedDict):

    def __init__(self, *args, **kwds):
        self.size_limit = kwds.pop("size_limit", None)
        OrderedDict.__init__(self, *args, **kwds)
        self._check_size_limit()

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)


class FIFOCache(object):
    """
    Pretty stupid cache that pops the first item once full. No strategy here as
    we usually don't want biased statistics anyway.
    """
    def __init__(self, size_limit=10000):
        self.size_limit = size_limit
        self.store = LimitedSizeDict(size_limit=size_limit)

    def get(self, key):
        if key not in self.store:
            return None
        else:
            return self.store[key]

    def set(self, key, value):
        self.store[key] = value
