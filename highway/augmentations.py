import numpy as np
from scipy import ndimage
from scipy import misc

from transforms import clipped_zoom
from transforms import image_histogram_equalization
from transforms import flip_x
from transforms import pad
from transforms import crop
from transforms import add_noise
from transforms import shift
from transforms import rotate


class Augmentation(object):
    """
    Apply a certain augmentation onto a set of data tensors.
    """

    def apply(self, values, deterministic=False):
        raise NotImplementedError("Implement apply() of Augmentation.")


class Resize(Augmentation):

    def __init__(self, shape, method='nearest', images_index=0):
        self.shape = shape
        self.method = method
        self.images_index = images_index

    def apply(self, values, deterministic=False):
        images = values[self.images_index]
        reshaped = []
        for image in images:
            reshaped.append(misc.imresize(image, self.shape,
                                          interp=self.method)[np.newaxis])

        values[self.images_index] = np.concatenate(reshaped)
        return values


class FlipX(Augmentation):
    """
    Flig image along x axis
    """

    def __init__(self, images_index=0):
        self.images_index = images_index

    def apply(self, values, deterministic=False):
        if deterministic:
            return values
        else:
            images = values[self.images_index]
            for idx in range(images.shape[0]):
                image = images[idx]
                p = np.random.randint(2)
                if p == 0:
                    image = flip_x(image)
                    images[idx] = image
            return values


class PadCrop(Augmentation):
    """
    Pad image with zeros and crop randomly
    """

    def __init__(self, padsize=4, mode='constant', images_index=0):
        self.padsize = padsize
        self.mode = mode
        self.images_index = images_index

    def apply(self, values, deterministic=False):

        if deterministic:
            return values
        else:
            images = values[self.images_index]
            for idx in range(images.shape[0]):
                image = images[idx]
                cx = np.random.randint(2 * self.padsize)
                cy = np.random.randint(2 * self.padsize)

                padded = pad(image, (self.padsize, self.padsize), self.mode)
                x = image.shape[2]
                y = image.shape[1]
                images[idx] = crop(padded, (cy, cx), (y, x))
            return values


class AdditiveNoise(Augmentation):
    """
    Additive noise for images
    """

    def __init__(self, strength=0.2, mu=0, sigma=50, images_index=0):
        self.strength = strength
        self.mu = mu
        self.sigma = sigma
        self.images_index = images_index

    def apply(self, values, deterministic=False):
        if deterministic:
            return values
        else:
            images = values[self.images_index]
            noisy = add_noise(images, self.strengt, self.mu, self.sigma)
            values[self.images_index] = noisy
            return values


class Shift(Augmentation):
    """
    Shift/Translate image randomly. Shift indicates the percentage of the images width to be shifted
    """

    def __init__(self, shift, mode='constant', images_index=0):
        self.shift = shift
        self.mode = mode
        self.images_index = images_index

    def apply(self, values, deterministic=False):
        if deterministic:
            return values
        images = values[self.images_index]
        for idx in range(images.shape[0]):
            image = images[idx]
            x_range = self.shift * image.shape[1]
            y_range = self.shift * image.shape[0]
            x = np.random.randint(-x_range, x_range)
            y = np.random.randint(-y_range, y_range)

            images[idx] = shift(image, (y, x),  mode=self.mode)
        return values


class Rotate(Augmentation):
    """
    Rotate image along a random angle within (-angle, +angle)
    """

    def __init__(self, angle, order=0, reshape=False, images_index=0):
        self.angle = angle
        self.order = order
        self.reshape = reshape
        self.images_index = images_index

    def apply(self, values, deterministic=False):
        if deterministic:
            return values
        images = values[self.images_index]
        for idx in range(images.shape[0]):
            image = images[idx]
            rot_angle = np.random.randint(-self.angle, self.angle)
            new_image = rotate(
                image, rot_angle, order=self.order, reshape=self.reshape)
            images[idx] = new_image
        return values


class Zoom(Augmentation):
    """
    Zoom image with a factor f in (1-fac, 1+fac)
    """

    def __init__(self, fac, order=0, images_index=0):
        self.fac = fac
        self.order = order
        self.images_index = images_index

    def apply(self, values, deterministic=False):
        if deterministic:
            return values

        images = values[self.images_index]
        for idx in range(images.shape[0]):
            image = images[idx]
            fac = np.random.uniform(1 - self.fac, 1 + self.fac)
            new_image = clipped_zoom(image, zoom_factor=fac, order=self.order)
            images[idx] = new_image
        return values


class HistEq(Augmentation):

    def __init__(self, images_index=0):
        self.images_index = images_index

    def apply(self, values, deterministic=False):
        images = values[self.images_index]
        for idx in range(images.shape[0]):
            images[idx] = image_histogram_equalization(images[idx])
        return values


class Slicer(Augmentation):

    def __init__(self, height=0.33, width=1.0, yoffset=0., xoffset=0., images_index=0):
        if height > 1. or width > 1.:
            raise ValueError("Edge length must be float between 0 and 1.")
        if xoffset > 1. or yoffset > 1.:
            raise ValueError("Offset must be float betwee 0 and 1.")

        self.height = height
        self.width = width
        self.xoffset = xoffset
        self.yoffset = yoffset
        self.images_index = images_index

    def apply(self, values, deterministic=False):
        images = values[self.images_index]
        # NHWC
        image_height = images.shape[1]
        image_width = images.shape[2]
        window_height = int(self.height * image_height)
        window_width = int(self.width * image_width)
        yoffset_height = int(self.yoffset * image_height)
        xoffset_height = int(self.xoffset * image_width)

        if image_height - 2 * yoffset_height < window_height:
            raise ValueError(
                "Cannot fit slicing window with current yoffset specified. Lower offset value.")

        if image_width - 2 * xoffset_height < window_width:
            raise ValueError(
                "Cannot fit slicing window with current xoffset specified. Lower offset value.")

        slices = []
        for idx in range(images.shape[0]):
            if deterministic:
                ystart = int((image_height - 2 * yoffset_height) //
                             2 + yoffset_height)
                xstart = int((image_width - 2 * xoffset_height) //
                             2 + xoffset_height)
            else:
                if image_height == window_height:
                    h = 0
                else:
                    h = np.random.randint(
                        image_height - 2 * yoffset_height - window_height)
                ystart = int(h + yoffset_height)

                if image_width == window_width:
                    w = 0
                else:
                    w = np.random.randint(
                        image_width - 2 * xoffset_height - window_width)
                xstart = int(w + xoffset_height)
            slice = images[idx, ystart:ystart +
                           int(window_height), xstart:xstart + int(window_width)]
            slices.append(slice[np.newaxis])
        slices = np.concatenate(slices)
        values[0] = slices
        return values


class RescaleImages(Augmentation):

    def __init__(self, scale=1. / 128., offset=128., images_index=0):
        self.scale = scale
        self.offset = offset
        self.images_index = images_index

    def apply(self, values, deterministic=False):
        images = values[self.images_index]
        images = (images - self.offset) * self.scale
        values[0] = images
        return values
