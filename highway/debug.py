import os
import sys
import time
from .engine import Node
from .modules.base import StreamWriter

import pickle
from scipy.misc import imsave

try:
    import Queue
except:
    import queue as Queue


class ImgDumper(StreamWriter):
    """
    Saves all images in a batch to a directory named by the index in the batch.
    All images are overriden once a new batch arrives.
    This is meant as a debug tool in order to supervise augmentations and check the image stream.
    """

    def __init__(self, out_dir, enqueue=False, batch_naming=True, file_type=".jpg"):
        self.out_dir = out_dir
        self.batch_naming = batch_naming
        self.file_type = file_type
        super(ImgDumper, self).__init__(enqueue)

    def dump_func(self, stream):
        ct = 0
        batch = stream['images']
        for item in batch:
            imsave(self.out_dir + "/" + str(ct) + self.file_type, item)
            ct += 1


class Benchmark(Node):
    """
    Benchmarks the payload rate which is received by this node's input queue.
    Be aware that this node deques everything and therefore must be the last node in the pipeline.
    The benchmark's output is not entirely accurate since measurement of the payload requires serializing of the received objects
    which takes additional time.
    """

    def __init__(self, N=50, substract_pickling=False):
        self.N = N
        self.substract_pickling = substract_pickling
        super(Benchmark, self).__init__()

    def run(self):
        while True:
            n_bytes = 0
            n_iters = 0

            s = time.time()
            pickle_time = 0
            for _ in range(self.N):
                try:
                    stream = self.input.dequeue()

                    p_s = time.time()
                    n_bytes += len(pickle.dumps(stream))
                    p_e = time.time()
                    pickle_time += p_e - p_s

                    n_iters += 1
                except Queue.Empty:
                    continue

            e = time.time()
            d = e - s

            if self.substract_pickling:
                d -= pickle_time

            bs = (n_bytes / d) / 1000000.
            print ("MBytes/s: ", bs, " Iters: ", n_iters / d, end="\r")
            sys.stdout.flush()
