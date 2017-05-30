import time
import sys

sys.path.append("..")

from highway.engine import Pipeline
from highway.modules import Noise, ZMQSink, ZMQSource

N = 1000
data_shape = (320, 240, 3)

source = Pipeline([Noise(data_shape=data_shape, n_tensors=3, n_worker=2, force_constant=True)])

print "Running benchmark..."

n_bytes = 0
n_iters = 0
s = time.time()
for _ in range(N):
    data = source.dequeue()
    for tensor in data:
        n_bytes += tensor.nbytes
    n_iters += 1

e = time.time()
d = e - s
bs = (n_bytes / d) / 1000000.
print "MBytes/s: ", bs
print "Iters/s", n_iters/d
