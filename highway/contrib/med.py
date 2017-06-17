import os
from tempfile import mkdtemp
import numpy as np

from medpy.io import load
from ..engine import Node
from ..utils import FIFOCache


class NiftiSliceReader(Node):
    """
    Read nifti data from a directory given the naming convention
    volume-xx.nifti and segmentation-xx.nifti
    """
    def __init__(self, data_dir, batch_size=32, neighbor_slices=0):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.neighbor_slices = neighbor_slices

        # Store the file prefixes
        self.file_map = None

        super(NiftiSliceReader, self).__init__(n_worker=1)

    def _map_nifti(filename):
        # Load from nifti
        image_data, _ = load(filename)
        # Convert to mmap file
        tmp_path = path.join(mkdtemp(), os.path.basename(filename))
        nmap_filename = tmp_path.replace("nifti", "dat")
        fp = np.memmap(nmap_filename, dtype=np.float32, mode="w+", shape=image_data.shape)
        fp[:] = image_data[:]

        # Flush contents
        del fp

        # Return
        fp = np.memmap(nmap_filename, dtype=np.float32, mode="r", shape=image_data.shape)
        return fp

    def setup():
        # Convert the files into mmaped arrays
        files = os.listdir(self.data_dir)
        for f in files:
            if "volume" in f:
                vol_p = self._map_nifti(f)
                seg_p = self._map_nifti(f.replace("volume", "segmentation"))
                self.file_map.append((vol_p, seg_p))


    def loop():
        volumes = []
        segmentations = []
        for _ in range(self.batch_size):
            # Pull a random slice
            volume_index = np.random.randint(len(self.file_map))
            vol_p, seg_p = self.file_map[volume_index]

            slice_index = np.random.randint(neighbor_slices, vol_p.shape[0] - self.neighbor_slices)

            volume_data = vol_p[slice_index - neighbor_slices: neighbor_slices + neighbor_slices + 1]
            seg_data = seg_p[slice_index - neighbor_slices: neighbor_slices + neighbor_slices + 1]

            volumes.append(volume_data[np.newaxis])
            segmentations.append(seg_data[np.newaxis])

        volumes = np.concatenate(volumes)
        segmentations = np.concatenate(segmentations)

        self.enqueue({"images": volumes, "labels": segmentations})
