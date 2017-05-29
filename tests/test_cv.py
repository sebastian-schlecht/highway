import pytest
import os
import numpy as np
from PIL import Image
import shutil

from highway.train import CV

def cond_create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

class TestCV:
    @pytest.fixture()
    def tmp_dir(self):
        DIR = "tmp/"
        cond_create_dir(DIR)
        for idx in xrange(10):
            class_dir = DIR + "/" + str(idx)
            cond_create_dir(class_dir)
            for jdx in xrange(20):
                arr = np.random.uniform(0, 255, size=(24, 24, 3)).astype(np.uint8)
                img = Image.fromarray(arr)
                img.save(class_dir + "/" + str(jdx) + ".jpg")
        yield DIR
        shutil.rmtree(DIR)

    def test_cv(self, tmp_dir):
        """
        Make sure that train/val splits do not contain similar images
        """
        cv = CV.from_image_folder(tmp_dir, folds=4)
        assert len(cv) == 4
        for train, val in cv:
            assert len(train) == len(val)
            for key in train:
                for filename in train[key]:
                    for vkey in val:
                        assert filename not in val[vkey]
                        for valfile in val[vkey]:
                            img_a = np.array(Image.open(tmp_dir + "/" + filename))
                            img_b = np.array(Image.open(tmp_dir + "/" + valfile))
                            assert not np.allclose(img_a, img_b)
