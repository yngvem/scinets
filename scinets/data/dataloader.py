"""
"""


__author__ = "Yngve Mardal Moe"
__email__ = "yngve.m.moe@gmail.com"


import h5py
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from .datareader import get_datareader
from .._backend_utils import SubclassRegister


dataset_register = SubclassRegister("data reader")

def get_dataset(dataset):
    return dataset_register.get_item(dataset)


@dataset_register.link_base
class BaseDataset:
    """Base class for datasets.

    To implement subclasses of this dataset, the init function should generate 
    three instances of the BaseReader class with the names:
      * train_data_reader
      * val_data_reader
      * test_data_reader
    """
    def _create_conditionals(self):
        """Set up conditional operators specifying which datasets to use.
        """
        with tf.variable_scope("test_train_or_val"):
            val_test_data = tf.cond(
                self.is_testing,
                true_fn=lambda: self._test_data,
                false_fn=lambda: self._val_data,
                name="use_test_data",
            )
            self._conditional_data = tf.cond(
                self.is_training,
                true_fn=lambda: self._train_data,
                false_fn=lambda: val_test_data,
                name="use_train_data",
            )

            val_test_target = tf.cond(
                self.is_testing,
                true_fn=lambda: self._test_target,
                false_fn=lambda: self._val_target,
                name="use_test_target",
            )
            self._conditional_target = tf.cond(
                self.is_training,
                true_fn=lambda: self._train_target,
                false_fn=lambda: val_test_target,
                name="use_train_target",
            )

            val_test_idxes = tf.cond(
                self.is_testing,
                true_fn=lambda: self._test_idxes,
                false_fn=lambda: self._val_idxes,
                name="use_test_idxes",
            )
            self._conditional_idxes = tf.cond(
                self.is_training,
                true_fn=lambda: self._train_idxes,
                false_fn=lambda: val_test_idxes,
                name="use_train_idxes",
            )

    @property
    def initializers(self):
        return [
            self.train_data_reader.initializer,
            self.test_data_reader.initializer,
            self.val_data_reader.initializer,
        ]

    @property
    def data(self):
        return self._conditional_data

    @property
    def target(self):
        return self._conditional_target

    @property
    def idxes(self):
        return self._conditional_idxes

    @property
    def _train_data(self):
        return self.train_data_reader.images

    @property
    def _train_target(self):
        return self.train_data_reader.targets

    @property
    def _train_idxes(self):
        return self.train_data_reader.idxes

    @property
    def _val_data(self):
        return self.val_data_reader.images

    @property
    def _val_target(self):
        return self.val_data_reader.targets

    @property
    def _val_idxes(self):
        return self.val_data_reader.idxes

    @property
    def _test_data(self):
        return self.test_data_reader.images

    @property
    def _test_target(self):
        return self.test_data_reader.targets

    @property
    def _test_idxes(self):
        return self.test_data_reader.idxes


class Dataset(BaseDataset):
    def __init__(self, reader_type, reader_kwargs, batch_size, is_training, is_testing):
        """Create a basic dataset.

        Parameters:
        -----------
        reader_type : str
            The name of a BaseDataReader subclass.
        reader_kwargs : dict
            Dictionary of dictionaries, should have the following structure
                {
                    "all": {
                        # kwargs for all data readers
                    },
                    "train": {
                        # kwargs for train data readers
                    },
                    "val": {
                        # kwargs for val data readers
                    },
                    "test": {
                        # kwargs for test data readers
                    },
                }
            If any of the keys above is missing, an empty dict will be used.
        batch_size : tuple[int]
            A list containing the batch size, first element is train batch size,
            second element is val batch size and final element is the test batch size.
        is_training : tf.placeholder
            Used to specify which datareader to feed into the TensorFlow graph.
        is_testing : tf.placeholder
            Used to specify which datareader to feed into the TensorFlow graph.
            This is not used if is_training is True.
        """
        if isinstance(batch_size, int):  # Check if batch_size is iterable
            batch_size = [batch_size] * 3
        self.batch_size = batch_size

        if is_training is None:
            is_training = tf.placeholder_with_default(
                True, shape=[], name="is_training"
            )
        if is_testing is None:
            is_testing = tf.placeholder_with_default(False, shape=[], name="is_testing")

        self.is_training = is_training
        self.is_testing = is_testing


        DataReader = get_datareader(reader_type)
        with tf.variable_scope("data_loader"):
            self.train_data_reader = DataReader(
                **reader_kwargs.get('all', {}),
                **reader_kwargs.get('train', {}),
                name="train_reader",
            )
            self.val_data_reader = DataReader(
                **reader_kwargs.get('all', {}),
                **reader_kwargs.get('val', {}),
                name="val_reader",
            )
            self.test_data_reader = DataReader(
                **reader_kwargs.get('all', {}),
                **reader_kwargs.get('test', {}),
                name="test_reader",
            )
            self._create_conditionals()


class HDFDataset(BaseDataset):
    """A wrapper for the dataset class that makes it easier to create datasets
    with HDFReaders.
    """

    def __init__(
        self,
        data_path,
        batch_size,
        train_group="train",
        val_group="validation",
        test_group="test",
        dataset="images",
        target="masks",
        prefetch=1,
        preprocessor=None,
        is_training=None,
        is_testing=None,
    ):
        """Setup the data reader.

        The HDF5 file should have one group for the training set, one for the
        test set and one for the validation set. The input data and
        and the labels should be in two different HDF5 dataset.

        Parameters:
        -----------
        data_path : str
            The path to the h5 file.
        batch_size : int or list
            The batch size(s) to use, if list, the first number will be used
            as training batch size, the second as the validation batch size
            and the last as the test batch size.
        train_group : str
            Internal path to the h5 group where the training data is.
        test_group : str
            Internal path to the h5 group where the test data is.
        val_group : str
            Internal path to the h5 group where the validation data is.
        dataset : str
            Name of the h5 dataset which contains the data points.
        target : str
            Name of the h5 dataset which contains the labels (or whichever
            output that is wanted from the network).
        prefetch : int
            Number of batches to load at the same time as a training step is
            performed. Used to reduce waiting time between training steps.
        preprocessor : str or None
            The preproccesing method to use. Must be an element of the
            `scinets.data.preprocessor` module.
        is_training : tensorflow.Placeholder(bool, [])
            Placeholder used to specify whether the training data should be
            the output or not.
        is_testing : tensorflow.Placeholder(bool, [])
            Placeholder used to specify whether the test or validation data
            should be used. If `is_training` is True, this is ignored.
        """
        self.data_path = str(data_path)

        reader_kwargs = {
            "all": {
                "data_path": data_path,
                "dataset": dataset,
                "prefetch": prefetch,
                "preprocessor": preprocessor,
            },
            "train": {"group": train_group},
            "val": {"group": val_group},
            "test": {"group": test_group},
        }

        super(batch_size=batch_size, is_training=is_training, is_testing=is_testing,
              reader_type='HDFReader', reader_kwargs=reader_kwargs)


class MNISTDataset(BaseDataset):
    """This is a hack so we can use the MNIST dataset.

    The test dataset is also used as the validation dataset, so this is ONLY for
    debugging.
    """

    def __init__(self, batch_size=128, is_training=None, is_testing=None, name="scope"):
        if is_training is None:
            is_training = tf.placeholder_with_default(
                True, shape=[], name="is_training"
            )
        self.is_training = is_training
        self.batch_size = batch_size
        self.epoch = 40000

        self._curr_it_num = 0
        self._data_it_num = 0
        self._labels_it_num = 0
        with tf.variable_scope(name):
            self.initializers = []
            self._train_next_el_op = self._get_next_el_op(self._iterate_train_dataset)
            self._val_next_el_op = self._get_next_el_op(self._iterate_val_dataset)

            self._create_conditionals()

    def _get_next_el_op(self, generator):
        dataset = tf.data.Dataset.from_generator(
            generator=generator,
            output_types=(tf.int16, tf.float32, tf.float32),
            output_shapes=(
                [],
                [self.batch_size, 28, 28, 1],
                [self.batch_size, 1, 1, 10],
            ),
        )
        tf_iterator = dataset.make_one_shot_iterator()
        return tf_iterator.get_next()

    def _iterate_train_dataset(self):
        dataset = input_data.read_data_sets("MNIST_data", one_hot=True)
        while True:
            x, y = dataset.train.next_batch(self.batch_size)
            x = x.reshape((self.batch_size, 28, 28, 1))
            y = y.reshape((self.batch_size, 1, 1, 10))
            yield 0, x, y

    def _iterate_val_dataset(self):
        dataset = input_data.read_data_sets("MNIST_data", one_hot=True)
        while True:
            x, y = dataset.test.next_batch(self.batch_size)
            x = x.reshape((self.batch_size, 28, 28, 1))
            y = y.reshape((self.batch_size, 1, 1, 10))
            yield 0, x, y

    def _create_conditionals(self):
        """Set up conditional operators specifying which datasets to use.
        """
        with tf.variable_scope("test_train_or_val"):
            self._conditional_data = tf.cond(
                self.is_training,
                true_fn=lambda: self._train_data,
                false_fn=lambda: self._val_data,
                name="use_train_data",
            )

            self._conditional_target = tf.cond(
                self.is_training,
                true_fn=lambda: self._train_target,
                false_fn=lambda: self._val_target,
                name="use_train_target",
            )

            self._conditional_idxes = tf.cond(
                self.is_training,
                true_fn=lambda: self._train_idxes,
                false_fn=lambda: self._val_idxes,
                name="use_train_idxes",
            )

    @property
    def data(self):
        return self._conditional_data

    @property
    def target(self):
        return self._conditional_target

    @property
    def idxes(self):
        return self._conditional_idxes

    @property
    def _train_data(self):
        return self._train_next_el_op[1]

    @property
    def _train_target(self):
        return self._train_next_el_op[2]

    @property
    def _train_idxes(self):
        return self._train_next_el_op[0]

    @property
    def _val_data(self):
        return self._val_next_el_op[1]

    @property
    def _val_target(self):
        return self._val_next_el_op[2]

    @property
    def _val_idxes(self):
        return self._val_next_el_op[0]


if __name__ == "__main__":
    pass
