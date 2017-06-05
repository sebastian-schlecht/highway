import time
import sys

sys.path.append("..")

from highway.engine import Pipeline

from highway.modules import Augmentations
from highway.modules import Noise, ZMQSink, ZMQSource

from highway.augmentations import Resize

N = 50
data_shape = (32, 320, 240, 3)

source = Pipeline([Noise(data_shape=data_shape, n_tensors=3, n_worker=2, force_constant=True), Augmentations([Resize((255, 255))])])

print "Running benchmark..."

n_bytes = 0
n_iters = 0
s = time.time()
for _ in range(N):
    data = source.dequeue()
    for tensor in data:
        n_bytes += tensor.nbytes

    images = data[0]
    assert images.shape == (32, 255, 255, 3)

    n_iters += 1

e = time.time()
d = e - s
bs = (n_bytes / d) / 1000000.
print "MBytes/s: ", bs
print "Iters/s", n_iters/d
