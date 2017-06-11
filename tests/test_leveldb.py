import pytest
import os
import time
import numpy as np
from PIL import Image
import shutil

from highway.modules.db import LevelDBSource, LevelDBSink
from highway.modules.processing import Noise
from highway.engine import Pipeline

def cond_create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


class TestLevelDB:
    @pytest.fixture()
    def tmp_dir(self):
        DIR = "tmp/"
        cond_create_dir(DIR)
        # Do nothing else than create a temporary folder
        yield DIR
        # Cleanup
        shutil.rmtree(DIR)

    def test_db_pipe(self, tmp_dir):
        db_name = tmp_dir + "/test.db"
        image_shape = (320, 240, 3)
        p_a = Pipeline([Noise(data_shape=image_shape, n_tensors=10, force_constant=True), LevelDBSink(filename=db_name)])
        # Wait until all threads spin up
        time.sleep(0.5)
        assert os.path.exists(db_name)

        # todo This does not work as pipeline shutdown is currently not implemented
        """
        p_b = Pipeline([LevelDBSource(filename=db_name)])
        for batch in p_b.dequeue():
            images = batch["images"]
            assert images.shape == (10, 320, 240, 3)
        """
