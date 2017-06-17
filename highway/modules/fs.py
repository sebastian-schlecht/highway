import numpy as np

from scipy.misc import imsave

from .base import StreamWriter
from ..engine import Node
from ..utils import get_directory_filenames, load_image, get_class_file_map, FIFOCache
from ..constants import IMAGE_FILETYPES


class ClfImgReader(Node):
    """
    Imagefile reader to stream classification data from a set of folders whose names are used as classes.
    Operations are random.
    TODO: Deterministic read in, add keys before the imgs are put on the queue for later (debug) identification
    """

    def __init__(self, data_dir, batch_size, shape, file_map=None, cache_size=10000):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shape = shape
        self.file_map = file_map
        self.cache = None

        if not self.file_map:
            self.file_map, self.n_classes, self.classes = get_class_file_map(
                self.data_dir)

        if cache_size:
            self.cache = FIFOCache(cache_size)

        super(ClfImgReader, self).__init__()

    def loop(self):
        labels = []
        images = []
        keys = []
        for idx in range(self.batch_size):
            cls_index = np.random.randint(self.n_classes)
            one_hot = np.zeros((1, self.n_classes), dtype=np.float32)
            one_hot[0, cls_index] = 1.

            # Load a random sample from that class
            files = self.file_map[self.classes[cls_index]]
            filename = self.data_dir + "/" + \
                files[np.random.randint(len(files))]

            if self.cache
                image = self.cache.get(filename)
            else:
                image = None

            if not image:
                image = load_and_fit_image(
                    filename, self.shape)[np.newaxis]
                if self.cache:
                    self.cache.set(filename, image)

            labels.append(one_hot)
            images.append(image)
            keys.append(filename)

        images = np.concatenate(images)
        labels = np.concatenate(labels)
        keys = np.array(keys)
        self.enqueue({'images': images,
                        'labels': labels,
                        'keys': keys})


class ImgReader(Node):
    """
    Reads (random) images in a directory and puts them onto the queue unaltered as numpy arrays.
    The batch size defines how many images are put into the queue in one slot.
    """

    def __init__(self, data_dir, batch_size=32, random=True, once=False, cache_size=100):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.random = random
        self.once = once
        self.cache = None
        if cache_size:
            self.cache = FIFOCache(cache_size)
        self.filenames = get_directory_filenames(self.data_dir, IMAGE_FILETYPES)
        self.n_files = len(filenames)
        self.gc = 0
        super(ImgReader, self).__init__()

    def loop(self):
        payload = []
        indexes = []

        for bct in range(self.batch_size):
            if self.random:
                idx = np.random.randint(self.n_files)
            else:
                if self.gc > n_files - 1:
                    if self.once:
                        return
                    self.gc = 0

                idx = self.gc
                self.gc += 1

            if self.cache:
                img = self.cache.get(idx)
            else:
                img = None

            if img is None:
                img = load_image(self.data_dir + "/" + self.filenames[idx])
                if self.cache:
                    self.cache.set(idx, img)

            payload.append(img)
            indexes.append(idx)

        self.enqueue({'images': payload, 'keys': indexes})


class ImgSaver(StreamWriter):
    """
    Saves all images in a batch to a directory named by the given keys.
    """

    def __init__(self, out_dir, file_type=".jpg"):
        self.out_dir = out_dir
        self.file_type = file_type
        super(ImgSaver, self).__init__(False)

    def dump_func(self, stream):
        batch = stream['images']
        keys = stream['keys']
        for item, key in zip(batch, keys):
            imsave(self.out_dir + "/" + str(key) + self.file_type, item)
