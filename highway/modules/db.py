import plyvel
from ..engine import Node
import msgpack
import numpy as np
import msgpack_numpy as npack

try:
    import Queue
except:
    import queue as Queue

class LevelDBSource(Node):
    def __init__(self, filename, batch_size=32, decoding=npack.decode):
        self.filename = filename
        self.batch_size = batch_size
        self.decoding = decoding

        super(LevelDBSource, self).__init__(n_worker=1)

    def setup(self):
        self.db = plyvel.DB(self.filename, create_if_missing=False)

    def loop(self):
        samples = 0
        result_dict = {}
        for _, sample in self.db:
            sample = msgpack.unpackb(sample, object_hook=self.decoding)

            for key in sample:
                if key not in result_dict:
                    result_dict[key] = []
                    result_dict[key].append(sample[key][np.newaxis])
                else:
                    result_dict[key].append(sample[key][np.newaxis])
            samples += 1

            if samples == self.batch_size:
                # Concat
                for key in result_dict:
                    result_dict[key] = np.concatenate(result_dict[key])
                # Send
                self.enqueue(result_dict)
                samples = 0
                result_dict = {}
    def close(self):
        self.db.close()


class LevelDBSink(Node):
    def __init__(self, filename, encoding=npack.encode):
        self.filename = filename
        self.encoding = encoding
        self.global_idx = 0
        super(LevelDBSink, self).__init__(n_worker=1)

    def setup(self):
        self.db = plyvel.DB(self.filename, create_if_missing=True)

    def loop(self):
        data = self.input.dequeue()
        # get data list length
        n_samples = len(list(data.items())[0][1])
        for idx in range(n_samples):
            sample = {}
            for key in data:
                sample[key] = data[key][idx]
            serialized = msgpack.packb(sample, default=self.encoding)
            self.db.put(bytes(self.global_idx), serialized)
            self.global_idx += 1

    def close(self):
        self.db.close()
