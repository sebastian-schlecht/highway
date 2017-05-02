import multiprocessing
import numpy as np

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Node(object):
    DEFAULT_TIMEOUT = 5

    def thread_proc(self, pid):
        np.random.seed(pid)
        self.run()

    def __init__(self, n_worker=1, queue_size=10):
        self.n_worker = n_worker
        self.queue = multiprocessing.Queue(maxsize=queue_size)
        self.lock = multiprocessing.Lock()
        self.processes = []
        self.input = None

    def attach(self, node):
        self.input = node

    def dequeue(self):
        val = self.queue.get(block=True, timeout=Node.DEFAULT_TIMEOUT)
        return val

    def start_daemons(self):
        for pid in range(self.n_worker):
            p = multiprocessing.Process(target=self.thread_proc, args=(pid,))
            p.daemon = True
            self.processes.append(p)
            p.start()

    def run(self):
        raise NotImplementedError("Implement run() in your node")


class Pipeline(object):
    def __init__(self, nodes):
        self.nodes = nodes
        for idx in range(1, len(nodes)):
            self.nodes[idx].attach(self.nodes[idx - 1])
        # Run nodes
        for node in self.nodes:
            node.start_daemons()

    def dequeue(self):
        value = self.nodes[-1].dequeue()
        if value is None:
            raise TypeError("None type returned by pipeline. Are your nodes running?")
        return value
