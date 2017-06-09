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

    def run(self):
        ctx = zmq.Context()
        socket = ctx.socket(zmq.PUSH)

        if self.bind:
            socket.bind(self.target)
        else:
            socket.connect(self.target)

        while True and self.input:
            try:
                values = self.input.queue.get(
                    block=True, timeout=Node.DEFAULT_TIMEOUT)
            except Queue.Empty:
                continue

            if values is not None:
                # Send values ZMQ
                serialized = msgpack.packb(
                    values, default=self.encoding, use_bin_type=True)
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
                values = msgpack.unpackb(
                    serialized, object_hook=self.decoding, encoding='utf-8')
            except Exception as e:
                # raise everything for now
                raise e
            if values is not None:
                self.queue.put(values, block=True)
