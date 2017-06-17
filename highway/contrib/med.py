import logging
import os
from tempfile import mkdtemp
import numpy as np

from medpy.io import load
from ..engine import Node
from ..utils import FIFOCache, local_tmp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class NiftiSliceReader(Node):
    """
    Read nifti data from a directory given the naming convention
    volume-xx.nifti and segmentation-xx.nifti
    """
    def __init__(self, data_dir, batch_size=32, neighbor_slices=0, dtype=np.uint8):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.neighbor_slices = neighbor_slices
        self.dtype=dtype

        # Store the file prefixes
        self.file_map = []

        super(NiftiSliceReader, self).__init__(n_worker=1)

    def _map_nifti(self, filename):
        # Load from nifti
        image_data, _ = load(filename)
        # Convert to mmap file
        tmp_path = os.path.join(local_tmp(), os.path.basename(filename))
        nmap_filename = tmp_path.replace("nii", "dat")
        if not os.path.exists(nmap_filename):
            logger.info("Memory-mapping %s" % nmap_filename)
            fp = np.memmap(nmap_filename, dtype=self.dtype, mode="w+", shape=image_data.shape)
            fp[:] = image_data[:]

            # Flush contents
            del fp

        # Return
        logger.info("Preparing %s for memory access" % nmap_filename)
        fp = np.memmap(nmap_filename, dtype=self.dtype, mode="r", shape=image_data.shape)
        return fp

    def setup(self):
        # Convert the files into mmaped arrays
        files = os.listdir(self.data_dir)
        for f in files:
            fname = self.data_dir + "/" + f
            if "volume" in f:
                vol_p = self._map_nifti(fname)
                seg_p = self._map_nifti(fname.replace("volume", "segmentation"))
                self.file_map.append((vol_p, seg_p))


    def loop(self):
        volumes = []
        segmentations = []
        for _ in range(self.batch_size):
            # Pull a random slice
            volume_index = np.random.randint(len(self.file_map))
            vol_p, seg_p = self.file_map[volume_index]

            slice_index = np.random.randint(self.neighbor_slices, vol_p.shape[2] - self.neighbor_slices)

            volume_data = vol_p[:,:,slice_index - self.neighbor_slices: slice_index + self.neighbor_slices + 1]
            seg_data = seg_p[:,:,slice_index - self.neighbor_slices: slice_index + self.neighbor_slices + 1]

            volumes.append(volume_data[np.newaxis])
            segmentations.append(seg_data[np.newaxis])

        volumes = np.concatenate(volumes)
        segmentations = np.concatenate(segmentations)

        self.enqueue({"images": volumes, "labels": segmentations})
