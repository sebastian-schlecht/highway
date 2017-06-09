import os, sys, time
from .engine import Node
from .modules.base import StreamWriter

import pickle
from scipy.misc import imsave

class JpgDumper(StreamWriter):
    """
    Saves all images in a batch to a directory named by the index in the batch.
    All images are overriden once a new batch arrives.
    This is meant as a debug tool in order to supervise augmentations and check the image stream.
    """
    def __init__(self, out_dir, enqueue=False, batch_naming=True):
        self.out_dir = out_dir
        self.batch_naming = batch_naming
        super(JpgDumper, self).__init__(enqueue)

    def dump_func(self, stream):
        ct = 0
        batch = stream['images']
        for item in batch:
            imsave(self.out_dir + "/" + str(ct) + ".jpg", item)
            ct += 1

class Benchmark(Node):
    """
    Benchmarks the payload rate which is received by this node's input queue.
    Be aware that this node deques everything and therefore must be the last node in the pipeline.
    The benchmark's output is not entirely accurate since measurement of the payload requires serializing of the received objects
    which takes additional time.
    """
    N = 100

    def __init__(self):
        super(QueueInputBenchmark, self).__init__()

    def run(self):
        while True:
            n_bytes = 0
            n_iters = 0
            s = time.time()
            for _ in range(QueueInputBenchmark.N):
                try:
                    stream = self.dequeue()
                    n_bytes += len(pickle.dumps(stream))
                    n_iters += 1
                except Queue.Empty:
                    continue

            e = time.time()
            d = e - s
            bs = (n_bytes / d) / 1000000.
            print ("MBytes/s: ", bs, " Iters: ", n_iters / d, end="\r")
            sys.stdout.flush()
