import time

from highway.engine import Pipeline
from highway.modules.processing import Noise, Augmentations


class TestEngine:
    def test_single_node(self):
        p = Pipeline([Noise(data_shape=(1, 2), n_tensors=1)])
        time.sleep(0.1)
        data = p.dequeue()
        assert len(data) == 1
        assert data["images"].shape == (1, 1, 2)
        p.stop()

    def test_two_nodes(self):
        p = Pipeline([Noise(data_shape=(3, 5), n_tensors=2), Augmentations()])
        time.sleep(0.1)
        data = p.dequeue()
        assert len(data["images"]) == 2
        assert data["images"].shape == (2, 3, 5)
        p.stop()
