# Highway
Multi-threaded data access pipeline for tensor data.

![CircleCI](https://circleci.com/gh/sebastian-schlecht/highway.svg?style=shield&circle-token=8ca49a7720b3ba3404a56f277d6a533c420b24cb)


## Install
run ```python setup.py install``` to install.


## API
Highway is built around a sequential API to construct a pipeline that moves around data.

Simple example:

```python
from highway.engine import Pipeline
from highway.modules import Augmentations, ImageFileReader
from highway.transforms import FlipX

data_dir = "../some-dir"

# Create an image reader that loads images from a directory containing sub-directories for each label
img_reader = ImageFileReader(data_dir, 16, (240, 320))

# Build the pipeline. Randomly flip images along the x axis with a probability p=0.5
p = Pipeline([img_reader, Augmentations([FlipX()])])

# Pop a batch to feed into NNs
images, labels = p.dequeue()
```


If you are handling massive data augmentations, you can distribute processing across different machines and scale augmentations according to the machines' CPU capabilities using the ZMQ transport layer. Note: Usually, ```bind``` is set to True on worker machines for the sink and False on the training machine for the source. The reason is to minimize port usage and thus the training machine collects data from all concurrent worker machines.

```python

from highway.engine import Pipeline
from highway.modules import Augmentations, ImageFileReader, ZMQSink, ZMQSource
from highway.transforms import FlipX

data_dir = "../some-dir"

if is_data_source_machine:
  # Create an image reader that loads images from a directory containing sub-directories for each label
  img_reader = ImageFileReader(data_dir, 16, (240, 320))
  p = Pipeline([img_reader, ZMQSink("tcp://some-ip:some-port")])

elif is_worker_machine:
  # In case we're on a worker machine, pull some remote batches, augment them and push them into the training machine
  p = Pipeline([ZMQSource("tcp://some-ip:some-port"),  Augmentations([FlipX(), ...], ZMQSink("tcp://some-other-ip:some-port", bind=False)])

else:
  # Training machine
  p = Pipeline([ZMQSource("tcp://some-other-ip:some-port", bind=True)])
  images, labels = p.dequeue()
```

## Testing
To run the test suite, install development dependencies ```pip install -e .test```.
From the project root, run ```pytest```.

## Benchmarking and performance
Benchmark scripts are located in the folder ```benchmarks```. Right now, we measured that inter-process communication roughly maxes out at 500 Mbytes/s (depending on the machine you're using). For TCP communication we measured roughly 120Mbytes/s to be the upper limit (again, depends on the machine you're using but this may provide an idea where we're heading). We're not planning to add custom queues and message services ourselves so performance may be inherently limited by the tools that are available. Right now, we use [ZMQ](http://zeromq.org/) for messaging and [msgpack](https://pypi.python.org/pypi/msgpack-python) for serialization which offer good performance but we yet have to validate whether that's enough in the future.

## What we're planning to add
Some features are not there yet but may come in the near future. This includes, but is not limited to:

- More augmentations and potentially a cleaner programming API for them
- More examples
- Support for object-detection in PASCAL VOC format.
- Performance tweaks
