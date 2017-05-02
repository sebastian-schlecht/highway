import numpy as np
from scipy import ndimage

from utils import clipped_zoom
from utils import image_histogram_equalization


class Transform(object):
    """
    Apply a certain transform onto a set of data tensors.
    """

    def apply(self, values, deterministic=False):
        raise NotImplementedError("Implement apply() of transform.")


class FlipX(Transform):
    """
    Flig image along x axis
    """

    def apply(self, values, deterministic=False):

        if deterministic:
            return values
        else:
            images = values[0]
            for idx in range(images.shape[0]):
                image = images[idx]
                p = np.random.randint(2)
                if p == 0:
                    if len(image.shape) == 3:
                        image = image[:, :, ::-1]
                    else:
                        image = image[:, ::-1]
                    images[idx] = image
            return values


class PadCrop(Transform):
    """
    Pad image with zeros and crop randomly
    """

    def __init__(self, padsize=4, mode='constant'):
        self.padsize = padsize
        self.mode = mode

    def apply(self, values, deterministic=False):

        if deterministic:
            return values
        else:
            images = values[0]
            for idx in range(images.shape[0]):
                image = images[idx]
                cx = np.random.randint(2 * self.padsize)
                cy = np.random.randint(2 * self.padsize)

                if len(image.shape) == 3:
                    padded = np.pad(image, ((0, 0), (self.padsize, self.padsize), (self.padsize, self.padsize)),
                                    mode=self.mode)
                    x = image.shape[2]
                    y = image.shape[1]
                    images[idx] = padded[:, cy:cy + y, cx:cx + x]
                else:
                    padded = np.pad(image, ((self.padsize, self.padsize), (self.padsize, self.padsize)), self.mode)
                    x = image.shape[1]
                    y = image.shape[0]
                    images[idx] = padded[cy:cy + y, cx:cx + x]

            return values


class AdditiveNoise(Transform):
    """
    Additive noise for images
    """

    def __init__(self, strength=0.2, mu=0, sigma=50):
        self.strength = strength
        self.mu = mu
        self.sigma = sigma

    def apply(self, values, deterministic=False):
        if deterministic:
            return values
        else:
            images = values[0]
            noise = np.random.normal(self.mu, self.sigma, size=images.shape)
            noisy = images + self.strength * noise
            values[0] = noisy
            return values


class Shift(Transform):
    """
    Shift/Translate image randomly. Shift indicates the percentage of the images width to be shifted
    """

    def __init__(self, shift, mode='constant'):
        self.shift = shift
        self.mode = mode

    def apply(self, values, deterministic=False):
        if deterministic:
            return values
        images = values[0]
        for idx in range(images.shape[0]):
            image = images[idx]
            x_range = self.shift * image.shape[1]
            y_range = self.shift * image.shape[0]
            x = np.random.randint(-x_range, x_range)
            y = np.random.randint(-y_range, y_range)

            images[idx] = ndimage.interpolation.shift(image, (y, x), np.float32, mode=self.mode)
        return values


class Rotate(Transform):
    """
    Rotate image along a random angle within (-angle, +angle)
    """

    def __init__(self, angle, order=0, reshape=False):
        self.angle = angle
        self.order = order
        self.reshape = reshape

    def apply(self, values, deterministic=False):
        if deterministic:
            return values
        images = values[0]
        for idx in range(images.shape[0]):
            image = images[idx]
            rot_angle = np.random.randint(-self.angle, self.angle)
            new_image = ndimage.interpolation.rotate(image, rot_angle, order=self.order, reshape=self.reshape)
            images[idx] = new_image
        return values


class Zoom(Transform):
    """
    Zoom image with a factor f in (1-fac, 1+fac)
    """

    def __init__(self, fac, order=0):
        self.fac = fac
        self.order = order

    def apply(self, values, deterministic=False):
        if deterministic:
            return values

        images = values[0]
        for idx in range(images.shape[0]):
            image = images[idx]
            fac = np.random.uniform(1 - self.fac, 1 + self.fac)
            new_image = clipped_zoom(image, zoom_factor=fac, order=self.order)
            images[idx] = new_image
        return values


class HistEq(Transform):
    def apply(self, values, deterministic=False):
        images = values[0]
        for idx in range(images.shape[0]):
            images[idx] = image_histogram_equalization(images[idx])
        return values


class VerticalSlicer(Transform):
    def __init__(self, height=0.33, offset=0):
        if height > 1.:
            raise ValueError("Slice must be float between 0 and 1.")
        if offset > 1.:
            raise ValueError("Offset must be float betwee 0 and 1.")
        self.height = height
        self.offset = offset

    def apply(self, values, deterministic=False):
        images = values[0]
        # NHWC
        image_height = images.shape[1]
        window_height = self.height * image_height
        offset_height = self.offset * image_height

        if image_height - 2 * offset_height < window_height:
            raise ValueError("Cannot fit slicing window with current offset specified. Lower offset value.")

        slices = []
        for idx in range(images.shape[0]):
            if deterministic:
                start = int((image_height - 2 * offset_height) // 2 + offset_height)
            else:
                start = int(np.random.randint(image_height - 2 * offset_height - window_height) + offset_height)
            slice = images[idx, start:start + int(window_height)]
            slices.append(slice[np.newaxis])
        slices = np.concatenate(slices)
        values[0] = slices
        return values
