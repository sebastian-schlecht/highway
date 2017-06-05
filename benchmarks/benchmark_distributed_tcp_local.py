import time
import sys

sys.path.append("..")

from highway.engine import Pipeline
from highway.modules import Noise, ZMQSink, ZMQSource

N = 100
data_shape = (320, 240, 3)

sink = Pipeline([ZMQSource("tcp://127.0.0.1:5556")])
worker = Pipeline([ZMQSource("tcp://127.0.0.1:5555"), ZMQSink("tcp://127.0.0.1:5556")])
source = Pipeline([Noise(data_shape=data_shape, n_tensors=3, n_worker=4, force_constant=True), ZMQSink("tcp://127.0.0.1:5555")])

print "Running benchmark..."

n_bytes = 0
n_iters = 0

s = time.time()
for _ in range(N):
    data = sink.dequeue()
    for tensor in data:
        n_bytes += tensor.nbytes
    n_iters += 1

e = time.time()
d = e - s
bs = (n_bytes / d) / 1000000.
print "MBytes/s: ", bs
print "Iters/s", n_iters/d
