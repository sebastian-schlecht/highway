import msgpack
import msgpack_numpy as npack
import zmq
from ..engine import Node

try:
    import Queue
except:
    import queue as Queue


class ZMQSink(Node):

    def __init__(self, target, bind=True, encoding=npack.encode, flags=0):
        self.target = target
        self.bind = bind
        self.encoding = encoding
        self.flags = flags
        super(ZMQSink, self).__init__(n_worker=1)

    def setup(self):
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.PUSH)

        if self.bind:
            self.socket.bind(self.target)
        else:
            self.socket.connect(self.target)

    def loop(self):
        values = self.input.dequeue()
        if values is not None:
            # Send values ZMQ
            serialized = msgpack.packb(
                values, default=self.encoding, use_bin_type=True)
            result = self.socket.send(serialized, flags=self.flags)

    def close(self):
        self.socket.close()


class ZMQSource(Node):

    def __init__(self, source, bind=False, decoding=npack.decode, flags=0, copy=True, track=False):
        self.source = source
        self.bind = bind
        self.decoding = decoding
        self.flags = flags
        self.copy = copy
        self.track = track
        super(ZMQSource, self).__init__(n_worker=1)

    def setup(self):
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.PULL)
        if self.bind:
            self.socket.bind(self.source)
        else:
            self.socket.connect(self.source)

    def loop(self):
        serialized = self.socket.recv(
            flags=self.flags, copy=self.copy, track=self.track)
        values = msgpack.unpackb(
            serialized, object_hook=self.decoding, encoding='utf-8')
        if values is not None:
            self.enqueue(values)

    def close(self):
        self.socket.close()
