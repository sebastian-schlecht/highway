import numpy as np

from highway.engine import Pipeline

from highway.modules.processing import Noise, Augmentations
from highway.modules.network import ZMQSink, ZMQSource

from highway.augmentations.img import Resize

class TestProcessingPipeline:
    def test_resizing(self):
        N = 5
        data_shape = (320, 240, 3)
        source = Pipeline([Noise(data_shape=data_shape, n_tensors=32, n_worker=2, force_constant=True), Augmentations([Resize((255, 255), mode="resize", interp="nearest")])])
        for _ in range(N):
            data = source.dequeue()
            images = data["images"]
            assert images.shape == (32, 255, 255, 3)

        source.stop()
