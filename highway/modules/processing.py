import numpy as np
from ..engine import Node

try:
    import Queue
except:
    import queue as Queue


class Noise(Node):
    """
    Generate uniform noise for testing purposes.
    """

    def __init__(self, data_shape=(10, 10, 10), n_tensors=2, n_worker=1, force_constant=False, queue_size=10):
        self.data_shape = data_shape
        self.n_tensors = n_tensors
        self.force_constant = force_constant
        super(Noise, self).__init__(n_worker=n_worker, queue_size=queue_size)

    def loop(self):
        tensors = []
        for _ in range(self.n_tensors):
            if self.force_constant:
                data = np.zeros(self.data_shape, dtype=np.float32)
            else:
                data = np.random.uniform(size=self.data_shape)
            tensors.append(data[np.newaxis])
        tensors = np.concatenate(tensors)
        self.enqueue({"images": tensors})


class Augmentations(Node):
    """
    Node that applies a certain set of transforms in sequence.
    """

    def __init__(self, transforms=(), deterministic=False, n_worker=4, queue_size=10):
        self.transforms = transforms
        self.deterministic = deterministic

        super(Augmentations, self).__init__(
            n_worker=n_worker, queue_size=queue_size)

    def run(self):
        values = self.input.dequeue()
        if len(self.transforms) > 0:
            for transform in self.transforms:
                values = transform.apply(values, self.deterministic)
        self.enqueue(values)
