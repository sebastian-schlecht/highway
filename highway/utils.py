from PIL import Image, ImageOps
from scipy.ndimage.interpolation import zoom
import numpy as np
import os, sys


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
        file_map[cls] = [cls + "/" + f for f in os.listdir(data_dir + "/" + cls) if not extensions or (extensions and get_ext(f) in extensions)]

    return file_map, n_classes, classes