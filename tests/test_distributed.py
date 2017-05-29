import time

from highway.engine import Pipeline
from highway.modules import Noise, ZMQSink, ZMQSource


class TestPipeline:
    def test_zmq_loopback(self):
        sink = Pipeline([ZMQSource("tcp://127.0.0.1:5556")])
        worker = Pipeline([ZMQSource("tcp://127.0.0.1:5555"), ZMQSink("tcp://127.0.0.1:5556")])
        source = Pipeline([Noise(data_shape=(1, 2), n_tensors=1), ZMQSink("tcp://127.0.0.1:5555")])


        # Check 10 batches
        for _ in range(10):
            time.sleep(0.05)
            data = sink.dequeue()
            assert len(data) == 1
            assert data[0].shape == (1, 2)
