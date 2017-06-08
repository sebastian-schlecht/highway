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

"""
ct = 0
                    for item in stream[0]:
                        with open(self.out_dir + "/" + str(ct) + self.extension, 'wb') as target:
                            target.write(item)
                        sys.stdout.flush()
                        ct += 1
"""
def RandomDirectoryImageReaderDumper(payload, out_dir):
    """
    Since the format of a nodes' data is arbitrary, special implementations of debug functions are required.
    """
    ct = 0
    batch = payload[0]
    for item in batch:
        imsave(out_dir + "/" + str(ct) + ".jpg", item)
        ct += 1

def DirectoryImageReaderDumper(payload, out_dir):
    """
    Since the format of a nodes' data is arbitrary, special implementations of debug functions are required.
    """
    batch = payload[0]
    indexes = payload[1]
    ct = 0
    for item in batch:
        imsave(out_dir + "/" + str(indexes[ct]) + ".jpg", item)
        ct += 1

class QueueInputBenchmark(Node):
    """
    Benchmarks the payload rate which is received by this node's input queue.
    Be aware that this node deques everything and therefore must be the last node in the pipeline.
    The benchmark's output is not entirely accurate since measurement of the payload requires serializing of the received objects
    which takes additional time.
    """
    N = 5

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