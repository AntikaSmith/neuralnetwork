import numpy
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets import base
import model


TRAIN_PARTITION_NO = 1


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
            # perm = numpy.arange(self._num_examples)
            # numpy.random.shuffle(perm)
            # self._docs = self._docs[perm]
            # self._labels = self._labels[perm]
            # Start next epoch
            #start = 0
            difference = self._index_in_epoch - self._num_examples
            ret_docs = numpy.concatenate((self._docs[start: self._num_examples], self._docs[0:difference]), axis=0)
            ret_labels = numpy.concatenate((self._labels[start: self._num_examples], self._labels[0:difference]), axis=0)
            self._index_in_epoch = difference
            assert batch_size <= self._num_examples
            return ret_docs, ret_labels
        end = self._index_in_epoch
        return self._docs[start:end], self._labels[start:end]


def fake():
    return DataSet([], [], fake_data=True, one_hot=True, dtype=dtypes.float32)


def random_split(arr, ratio):
    shape = arr.shape[0]
    size = int(shape * ratio)
    numpy.random.shuffle(arr)
    return arr[:size, :], arr[size:, :]


def construct_set(arr):
    return DataSet(arr[:, 0:model.DOC_FEATURE_SIZE], arr[:, model.DOC_FEATURE_SIZE].astype(int), dtype=dtypes.float32,
            reshape=False)


def balance_positve_trains(train):
    ones = train[:, model.DOC_FEATURE_SIZE]
    zeros = 1 - ones
    one_array = numpy.compress(ones, train, axis=0)
    one_size = len(one_array)
    zero_array = numpy.compress(zeros, train, axis=0)
    ret = [0] * TRAIN_PARTITION_NO
    for i in range(0, TRAIN_PARTITION_NO):
        start = i*one_size
        stop = (i+1)*one_size
        ret[i] = numpy.concatenate((one_array, zero_array[start:stop]), axis=0)
        numpy.random.shuffle(ret[i])
    return ret


def read_data(train_file, validate_file_name):
    train_file = open(train_file, 'r')
    training = numpy.loadtxt(train_file, delimiter=',')
    train_file.close()
    train_arrays = balance_positve_trains(training)
    #read validate file
    validate_file = open(validate_file_name, 'r')
    validating = numpy.loadtxt(validate_file, delimiter=',')
    validate_file.close()
    validation = construct_set(validating)
    #construct data sets
    ret = [0] * TRAIN_PARTITION_NO
    for i in range(0, TRAIN_PARTITION_NO):
        train = construct_set(train_arrays[i])
        test = DataSet([], [], fake_data=True)
        ret[i] = base.Datasets(train=train, validation=validation, test=test)
    return ret
