import plyvel
from ..engine import Node
import msgpack
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
        self.db = plyvel.DB(self.filename, create_if_missing=False)
        super(LevelDBSource, self).__init__()

    def run():
        while True and self.db:
            samples = 0
            result_dict = {}
            for _, sample in self.db:
                sample = msgpack.unpackb(sample, object_hook=self.decoding, encoding='utf-8')
                for key in sample:
                    if key not in result_dict:
                        result_dict[key] = []
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


class LevelDBSink(Node):
    def __init__(self, filename, encoding=npack.encode):
        self.filename = filename
        self.encoding = encoding
        self.db = plyvel.DB(self.filename, create_if_missing=True)
        self.global_idx = 0
        super(LevelDBSink, self).__init__()

    def run():
        while True and self.db:
            try:
                data = self.input.dequeue()
            except Queue.Empty:
                continue

            # get data list length
            n_samples = len(data.items()[0][0])
            for idx in n_samples:
                sample = {}
                for key in data:
                    sample[key] = data[key][idx]
                serialized = msgpack.packb(sample, default=self.encoding, use_bin_type=True)
                self.db.put(bytes(self.global_idx), )
