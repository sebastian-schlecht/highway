import time
import sys

sys.path.append("..")

from highway.engine import Pipeline

from highway.modules.processing import Noise, Augmentations
from highway.modules.network import ZMQSink, ZMQSource

from highway.augmentations.img import Resize

N = 20
data_shape = (320, 240, 3)

source = Pipeline([Noise(data_shape=data_shape, n_tensors=32, n_worker=2, force_constant=True), Augmentations([Resize((255, 255), mode="resize", interp="nearest")])])

print "Running benchmark..."

n_bytes = 0
n_iters = 0
s = time.time()
for _ in range(N):
    data = source.dequeue()
    for key in data:
        n_bytes += data[key].nbytes

    images = data["images"]
    assert images.shape == (32, 255, 255, 3)
    n_iters += 1

e = time.time()
d = e - s
bs = (n_bytes / d) / 1000000.
print "MBytes/s: ", bs
print "Iters/s", n_iters/d
