from PIL import Image, ImageOps
from scipy.ndimage.interpolation import zoom
import numpy as np
import os


def get_ext(filename):
    name, ext = os.path.splitext(filename)
    return ext


def load_image(filename, shape, method=Image.NEAREST):
    img = Image.open(filename)
    fitted = ImageOps.fit(img, shape[::-1], method=method)
    return np.array(fitted)
