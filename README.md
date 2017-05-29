# Highway
Multi-threaded data access pipeline for tensor data.

## Install
run ```python setup.py install``` to install.


## Usage
Highway is built around a sequential API to construct a pipeline that moves around data.

Example:

```python
from highway.engine import Pipeline
from highway.modules import Augmentation, ImageFileReader
from highway.transforms import FlipX

data_dir = "../some-dir"

# Create an image reader that loads images from a directory containing sub-directories for each label
img_reader = ImageFileReader(data_dir, 16, (240, 320))

# Build the pipeline. Randomly flip images along the x axis with a probability p=0.5
p = Pipeline([img_reader, Augmentation([FlipX()])])

# Pop a batch to feed into NNs
images, labels = p.dequeue()
```

## Testing
To run the test suite, install development dependencies ```pip install -e .test```.
From the project root, run ```pytest```.
