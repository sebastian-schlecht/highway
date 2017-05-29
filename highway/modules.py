from engine import Node
from utils import load_image, get_ext

import Queue
import os
import numpy as np
import msgpack
import msgpack_numpy as npack
import zmq

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Noise(Node):
    """
    Generate uniform noise for testing purposes.
    """

    def __init__(self, data_shape=(10, 10, 10), n_tensors=2, n_worker=1, queue_size=10):
        self.data_shape = data_shape
        self.n_tensors = n_tensors
        super(Noise, self).__init__(n_worker=n_worker, queue_size=queue_size)

    def run(self):
        while True:
            tensors = []
            for _ in range(self.n_tensors):
                tensors.append(np.random.uniform(size=self.data_shape))
            self.queue.put(tensors, block=True)


class Augmentation(Node):
    """
    Node that applies a certain set of transforms in sequence.
    """

    def __init__(self, transforms=(), deterministic=False, n_worker=4, queue_size=10):
        self.transforms = transforms
        self.deterministic = deterministic
        super(Augmentation, self).__init__(
            n_worker=n_worker, queue_size=queue_size)

    def run(self):
        while True and self.input:
            try:
                values = self.input.queue.get(
                    block=True, timeout=Node.DEFAULT_TIMEOUT)
            except Queue.Empty:
                continue

            if len(self.transforms) > 0:
                for transform in self.transforms:
                    values = transform.apply(values, self.deterministic)
            self.queue.put(values, block=True)


class ZMQSink(Node):

    def __init__(self, target, bind=True, encoding=npack.encode, flags=0):
        self.target = target
        self.bind = bind
        self.encoding = encoding
        self.flags = flags
        super(ZMQSink, self).__init__(n_worker=1)

    def run(self):
        ctx = zmq.Context()
        socket = ctx.socket(zmq.PUSH)

        if self.bind:
            socket.bind(self.target)
        else:
            socket.conncet(self.target)

        while True and self.input:
            try:
                values = self.input.queue.get(
                    block=True, timeout=Node.DEFAULT_TIMEOUT)
            except Queue.Empty:
                continue

            if values is not None:
                # Send values ZMQ
                serialized = msgpack.packb(values, default=self.encoding)
                result = socket.send(serialized, flags=self.flags)


class ZMQSource(Node):

    def __init__(self, source, bind=False, decoding=npack.decode, flags=0, copy=True, track=False):
        self.source = source
        self.bind = bind
        self.decoding = decoding
        self.flags = flags
        self.copy = copy
        self.track = track
        super(ZMQSource, self).__init__(n_worker=1)

    def run(self):
        ctx = zmq.Context()
        socket = ctx.socket(zmq.PULL)
        if self.bind:
            socket.bind(self.source)
        else:
            socket.connect(self.source)

        while True:
            try:
                serialized = socket.recv(
                    flags=self.flags, copy=self.copy, track=self.track)
                values = msgpack.unpackb(serialized, object_hook=self.decoding)
            except Exception as e:
                # raise everything for now
                raise e
            if values is not None:
                self.queue.put(values, block=True)


class ImageFileReader(Node):
    FILETYPES = [".jpg", ".jpeg", ".png", ".bmp"]
    """
    Imagefile reader to stream classification data from a set of folders whose names are used as classes
    """

    def __init__(self, data_dir, batch_size, shape, file_map=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shape = shape

        self.classes = None
        self.n_classes = None
        self.file_map = file_map or None

        super(ImageFileReader, self).__init__()

    def run(self):
        classes = [cls for cls in os.listdir(
            self.data_dir) if os.path.isdir(self.data_dir + "/" + cls)]
        classes.sort()
        self.classes = classes
        self.n_classes = len(self.classes)
        self.file_map = {}
        for cls in self.classes:
            self.file_map[cls] = [cls + "/" + f for f in os.listdir(self.data_dir + "/" + cls) if
                                  get_ext(f) in ImageFileReader.FILETYPES]

        while True:
            labels = []
            images = []
            for idx in xrange(self.batch_size):
                cls_index = np.random.randint(self.n_classes)
                one_hot = np.zeros((1, self.n_classes), dtype=np.float32)
                one_hot[0, cls_index] = 1.

                # Load a random sample from that class
                files = self.file_map[self.classes[cls_index]]
                filename = self.data_dir + "/" + \
                    files[np.random.randint(len(files))]

                image = load_image(filename, self.shape)[np.newaxis]

                labels.append(one_hot)
                images.append(image)

            images = np.concatenate(images)
            labels = np.concatenate(labels)
            self.queue.put([images.astype(np.float32),
                            labels.astype(np.float32)], block=True)
