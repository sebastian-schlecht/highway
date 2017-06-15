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
        shutil.rmtree(DIR, ignore_errors=True)

    def test_db_pipe(self, tmp_dir):
        db_name = tmp_dir + "testdb"
        image_shape = (320, 240, 3)
        p_a = Pipeline([Noise(data_shape=image_shape, n_tensors=10, force_constant=True), LevelDBSink(filename=db_name)])
        # Wait until all threads spin up
        time.sleep(.5)
        assert os.path.exists(db_name)
        # Stop pipeline a
        p_a.stop()
        # Wait a sec until all workers shut down
        time.sleep(.5)

        p_b = Pipeline([LevelDBSource(filename=db_name, batch_size=2)])
        batch = p_b.dequeue()
        images = batch[b"images"]
        assert images.shape == (2, 320, 240, 3)

        p_b.stop()
