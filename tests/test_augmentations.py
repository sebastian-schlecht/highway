import numpy as np

from highway.augmentations import FlipX


class TestTransforms:
    def test_flip_x(self):
        arr = np.eye(10, 10, 3)[np.newaxis]
        r = FlipX().apply([arr])
        assert r[0].all() == arr[:, ::-1].all()
