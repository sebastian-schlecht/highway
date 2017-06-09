import numpy as np
from scipy import ndimage

def flip_x(image):
    return image[:,::-1]

def flip_y(image):
    return image[::-1]

def pad(image, size, mode='constant'):
    if len(image.shape) == 3:
        return np.pad(image, (size[0], size[0]), (size[1], size[1]), (0,0), mode=mode)
    else:
        return np.pad(image, (size[0], size[0]), (size[1], size[1]), mode=mode)

def crop(image, center, size):
    cy, cx = center
    y, x = size
    return image[cy:cy+h, cx:cx+w]

def add_noise(tensor, strength=0.2, mu=0, sigma=50):
    noise = np.random.normal(self.mu, self.sigma, size=images.shape)
    noisy = image + self.strength * noise

def shift(image, translation, mode='constant'):
    y, x = translation
    return ndimage.interpolation.shift(image, (y, x), image.dtype, mode=mode)

def rotate(image, angle, order=0, reshape=False):
    return ndimage.interpolation.rotate(image, angle, order=order, reshape=reshape)

def clipped_zoom(img, zoom_factor, **kwargs):
    img = img.astype(np.float32)
    h, w = img.shape[:2]

    # width and height of the zoomed image
    zh = int(np.round(zoom_factor * h))
    zw = int(np.round(zoom_factor * w))

    # for multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)
    # zooming out
    if zoom_factor < 1:
        # bounding box of the clip region within the output array
        if h > zh and w > zw:
            top = np.random.randint(h - zh)
            left = np.random.randint(w - zw)
        else:
            top = 0
            left = 0
        # zero-padding
        out = np.zeros_like(img, dtype=np.float32)
        out[top:top + zh, left:left + zw] = zoom(img, zoom_tuple, np.float32, **kwargs)

    # zooming in
    elif zoom_factor > 1:
        # bounding box of the clip region within the input array
        if zh > h and zw > w:
            top = np.random.randint(zh - h)
            left = np.random.randint(zw - w)
        else:
            top = 0
            left = 0

        out = zoom(img, zoom_tuple, np.float32, **kwargs)
        out = out[top:top + h, left:left + w]

    # if zoom_factor == 1, just return the input array
    else:
        out = img
    return out

def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)
