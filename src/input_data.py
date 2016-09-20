import numpy
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets import base
import model

class DataSet(object):

    def __init__(self,
          docs,
          labels,
          fake_data=False,
          one_hot=False,
          dtype=dtypes.float32,
          reshape=True):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert docs.shape[0] == labels.shape[0], (
                    'docs.shape: %s labels.shape: %s' % (docs.shape, labels.shape))
            self._num_examples = docs.shape[0]

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
        self._docs = docs
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def docs(self):
        return self._docs

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in range(batch_size)], [
                    fake_label for _ in range(batch_size)
            ]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._docs = self._docs[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._docs[start:end], self._labels[start:end]


def fake():
    return DataSet([], [], fake_data=True, one_hot=True, dtype=dtypes.float32)


def read_data(file_name):
    file = open(file_name, 'r')
    arr = numpy.loadtxt(file, delimiter=',')
    file.close()
    #random shuflle those data
    # perm = numpy.arange(numpy.shape(arr)[0])
    # numpy.random.shuffle(perm)
    # arr = arr[perm]
    train = DataSet(arr[:, 0:model.DOC_FEATURE_SIZE], arr[:, model.DOC_FEATURE_SIZE:(model.DOC_FEATURE_SIZE + 1)], dtype=dtypes.float32, reshape=False)
    validation = DataSet([], [], fake_data=True)
    test = DataSet([], [], fake_data=True)
    return base.Datasets(train=train, validation=validation, test=test)
