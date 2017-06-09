"""
All adapters to tensorpack/tensorflow and all other libraries can be found here.
"""
from multiprocessing import Queue
import sys


class DequeueGenerator(object):
    """
    Tensorpack RNGDataFlow Adapter
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def get_data(self):
        while True:
            values = None
            try:
                values = self.pipeline.dequeue()
            except Queue.Empty:
                continue

            for data in values[0]:
                # print ( "asdasdasd##############", len(values), len(values[0]), len(values[1]) )
                # sys.stdout.flush()
                # im = data[:, :, ::-1]
                yield [data]

    def size(self):
        return 128

    def get_size(self):
        return 128

    def reset_state(self):
        pass
