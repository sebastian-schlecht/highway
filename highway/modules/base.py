import abc

from multiprocessing import Queue

from ..engine import Node

class StreamWriter(Node):
    """
    Dumps the items in the input queue and enques them again for further use.
    This node is intended to be a debugging feature which can be plugged into the pipeline at any point.
    The default dumper function writes all items in the stream out to a file in binary mode named: {batch_index}.{extension}
    """
    def __init__(self, enqueue=False):
        self.enqueue_items = enqueue
        super(StreamWriter, self).__init__()

    @abc.abstractmethod
    def dump_func():
        return

    def run(self):
        ct = 0
        while True:
            # try:
                stream = self.input.dequeue()
                self.dump_func(stream)
                if self.enqueue_items:
                    self.enqueue(stream)
            # except Queue.Empty:
                # continue