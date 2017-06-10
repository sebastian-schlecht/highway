import sys

try:
    import Queue
except:
    import queue as Queue


class DequeueGenerator(object):
    """
    Tensorpack RNGDataFlow Adapter
    """

    def __init__(self, pipeline, queue_size=128):
        self.pipeline = pipeline
        self.queue_size = queue_size

    def get_data(self):
        while True:
            values = None
            try:
                values = self.pipeline.dequeue()
            except Queue.Empty:
                continue

            for data in values['images']:
                yield [data]

    def size(self):
        return self.queue_size

    def get_size(self):
        return self.queue_size

    def reset_state(self):
        pass
