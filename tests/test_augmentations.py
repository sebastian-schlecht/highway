import numpy as np

from highway.augmentations.img import FlipX, Resize, TopCenterCrop


class TestAugmentations:
    def test_flip_x(self):
        arr = np.eye(10, 10, 3)[np.newaxis]
        r = FlipX().apply({"images": arr})
        assert r["images"].all() == arr[:, ::-1].all()

    def test_resize(self):
        arr = np.eye(30, 30, 3)

        # default mode which resizes with a given tuple
        r = Resize((20, 20)).apply({"images": [arr]})
        for image in r['images']:
            assert image.shape == (20, 20)

        arr = np.eye(100, 100, 3)
        # default mode with a integer (percentage scaling)
        r = Resize(10).apply({"images": [arr]})
        for image in r['images']:
            assert image.shape == (10, 10)

        arr = np.eye(10, 100, 3)
        # width mode which keeps the aspect ratio and resizes the image to a given width
        r = Resize(50, mode='width').apply({"images": [arr]})
        for image in r['images']:
            assert image.shape == (5, 50)

    def test_topcenter_crop(self):
        arr = np.eye(30, 30, 3)

        # output shape must be exactly the given input tuple
        r = TopCenterCrop((20, 20)).apply({"images": [arr]})
        for image in r['images']:
            assert image.shape == (20, 20)
