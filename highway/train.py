import os
import numpy as np
from utils import get_ext

class CV(object):
    @staticmethod
    def from_image_folder(data_dir, folds=5):
        FILETYPES = [".jpg", ".jpeg", ".png", ".bmp"]
        classes = [cls for cls in os.listdir(data_dir) if os.path.isdir(data_dir + "/" + cls)]

        results = []
        for cls in classes:
            files = [(cls, cls + "/" + f) for f in os.listdir(data_dir + "/" + cls) if
                                  get_ext(f) in FILETYPES]

            np.random.shuffle(files)
            size = len(files) / folds
            chunks = [files[i:i + size] for i in xrange(0, len(files), size)]

            results.append(chunks)

        segments = []
        # Resort and flatten
        for f in xrange(folds):
            files = [result[f] for result in results]
            final_list = []
            for f in files:
                final_list += f
            segments.append(final_list)

        # For each fold, create a train/val split
        train_val = []
        for f in xrange(folds):

            v = {}
            for s in segments[f]:
                if s[0] not in v:
                    v[s[0]] = []
                v[s[0]].append(s[1])

            t = {}
            for idx in xrange(folds):
                if idx != f:
                    for s in segments[idx]:
                        if s[0] not in t:
                            t[s[0]] = []
                        t[s[0]].append(s[1])

            train_val.append((t, v))

        return train_val
