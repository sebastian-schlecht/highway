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
        self.cache = FIFOCache(cache_size)

        super(ClfImgReader, self).__init__()

    def run(self):
        if not self.file_map:
            self.file_map, self.n_classes, self.classes = get_class_file_map(
                self.data_dir)

        while True:
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

                image = self.cache.get(filename)
                if not image:
                    image = load_and_fit_image(
                        filename, self.shape)[np.newaxis]
                    self.cache.set(filename, image)

                labels.append(one_hot)
                images.append(image)
                keys.append(filename)

            images = np.concatenate(images)
            labels = np.concatenate(labels)
            keys = np.array(keys)
            self.enqueue({'images': images,
                            'labels': labels,
                            'keys': keys}, block=True)


class ImgReader(Node):
    """
    Reads (random) images in a directory and puts them onto the queue unaltered as numpy arrays.
    The batch size defines how many images are put into the queue in one slot.
    """

    def __init__(self, data_dir, batch_size=32, random=True, once=False, cache_size=10000):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.random = random
        self.once = once
        self.cache = FIFOCache(cache_size)
        super(ImgReader, self).__init__()

    def run(self):
        filenames = get_directory_filenames(self.data_dir, IMAGE_FILETYPES)
        n_files = len(filenames)
        gc = 0
        while True:
            payload = []
            indexes = []

            for bct in range(self.batch_size):
                if self.random:
                    idx = np.random.randint(n_files)
                else:
                    if gc > n_files - 1:
                        if self.once:
                            return
                        gc = 0

                    idx = gc
                    gc += 1
                img = self.cache.get(idx)
                if not img:
                    img = load_image(self.data_dir + "/" + filenames[idx])
                    self.cache.set(idx, img)

                payload.append(img)
                indexes.append(idx)

            self.enqueue({'images': payload, 'keys': indexes}, block=True)


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
