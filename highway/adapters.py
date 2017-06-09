import sys

try:
    import Queue
except:
    import queue as Queue


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
                yield [data]

    def size(self):
        return 128

    def get_size(self):
        return 128

    def reset_state(self):
        pass
