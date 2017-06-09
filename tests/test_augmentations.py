import numpy as np

from highway.augmentations.img import FlipX


class TestTransforms:
    def test_flip_x(self):
        arr = np.eye(10, 10, 3)[np.newaxis]
        r = FlipX().apply({"images": arr})
        assert r["images"].all() == arr[:, ::-1].all()
