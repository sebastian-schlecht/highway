import multiprocessing
import numpy as np

import Queue

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Stop(Exception):
    pass

class Node(object):
    DEFAULT_TIMEOUT = 1

    def thread_proc(self, pid):
        np.random.seed(pid)
        self.run()

    def __init__(self, n_worker=1, queue_size=128):
        self.n_worker = n_worker
        self.queue = multiprocessing.Queue(maxsize=queue_size)
        self.lock = multiprocessing.Lock()
        self.stop = multiprocessing.Event()
        self.processes = []
        self.input = None

    def attach(self, node):
        self.input = node

    def close(self):
        self.stop.set()

    def dequeue(self, block=True, timeout=DEFAULT_TIMEOUT):
        while True:
            try:
                val = self.queue.get(block=block, timeout=timeout)
                break
            except Queue.Empty:
                if self.stop.is_set():
                    raise Stop()
        return val

    def enqueue(self, data, block=True, timeout=DEFAULT_TIMEOUT):
        while True:
            try:
                self.queue.put(data, block, timeout=timeout)
                break
            except Queue.Full:
                if self.stop.is_set():
                    raise Stop()

    def start_daemons(self):
        for pid in range(self.n_worker):
            p = multiprocessing.Process(target=self.thread_proc, args=(pid,))
            p.daemon = True
            self.processes.append(p)
            p.start()

    def run(self):
        self.setup()
        while True:
            try:
                self.loop()
                if self.stop.is_set():
                    break
            except Stop:
                break
        self.close()

    def setup(self):
        pass

    def close(self):
        pass

    def loop(self):
        raise NotImplementedError("Implement loop() in your node")


class Pipeline(object):

    def __init__(self, nodes):
        self.nodes = nodes
        for idx in range(1, len(nodes)):
            self.nodes[idx].attach(self.nodes[idx - 1])
        # Run nodes
        for node in self.nodes:
            node.start_daemons()

    def dequeue(self, block=True):
        value = self.nodes[-1].dequeue(block)
        if value is None:
            raise TypeError(
                "None type returned by pipeline. Are your nodes running?")
        return value

    def close(self):
        for node in self.nodes:
            node.close()
