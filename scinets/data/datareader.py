"""
"""


__author__ = "Yngve Mardal Moe"
__email__ = "yngve.m.moe@gmail.com"


import numpy as np
import h5py
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from .preprocessing import get_preprocessor
from .._backend_utils import SubclassRegister
from abc import ABC, abstractmethod


datareader_register = SubclassRegister("data reader")


def get_datareader(datareader):
    return datareader_register.get_item(datareader)


@datareader_register.link_base
class BaseReader(ABC):
    """Wrapper for a dataset that can be accessed by indexes.
    """

    def __init__(self, batch_size, prefetch, preprocessor, name):
        """
        Parameters:
        -----------
        batch_size : int
        group : str
            Internal path to the h5 group where the data is.
        dataset : str
            Name of the h5 dataset which contains the data points.
        target : str
            Name of the h5 dataset which contains the labels (or whichever
            output that is wanted from the network).
        prefetch : int
            Number of batches to load at the same time as a training step is
            performed. Used to reduce waiting time between training steps.
        """
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.preprocessor = self._get_preprocessor(preprocessor)

        # The order in which the samples was drawn
        self.trained_order = []

        # Get the shapes of the input and output data
        self._len, self._data_shape, self._target_shape = self.get_shapes()

        with tf.variable_scope(name):
            self.tf_iterator = self._get_tf_iterator()
            self.initializer = self.tf_iterator.initializer
            self._next_el_op = self.tf_iterator.get_next()

    @property
    def data_shape(self):
        """Return the output shape of this dataset after preprocessing.
        """
        channels = self._data_shape[-1]
        output_channels = self.preprocessor.output_channels(channels)
        return (*self._data_shape[:-1], output_channels)

    @property
    def target_shape(self):
        """Return the target shape of this dataset after preprocessing.
        """
        targets = self._target_shape[-1]
        output_targets = self.preprocessor.output_targets(targets)
        return (*self._target_shape[:-1], output_targets)

    def __len__(self):
        return self._len

    @staticmethod
    def _get_preprocessor(preprocessor):
        if preprocessor is None:
            return get_preprocessor("Preprocessor")()
        elif isinstance(preprocessor, dict):
            operator = preprocessor["operator"]
            kwargs = preprocessor.get("arguments", {})
            return get_preprocessor(operator)(**kwargs)
        else:
            raise ValueError("`preprocess` must be either `None` or a dict")

    @abstractmethod
    def _iterate_dataset_randomly(self):
        """A generator that iterates through the dataset in a random order.

        No preprocessing is performed here.

        Yields:
        -------
        int : index of the currently yielded datapoint
        input : The input to the neural network
        target : The target to the neural network.
        """
        pass

    def iterate_dataset_randomly(self):
        """A generator that iterates through the dataset in a random order.

        Yields a tuple containing a single index, data point and target.
        """
        for idx, input_, target in self._iterate_dataset_randomly():
            yield (idx, *self.preprocessor(input_, target))

    def _get_tf_iterator(self):
        """Returns an initializable TensorFlow iterator that iterates the dataset.
        """
        output_types = (tf.int16, tf.float32, tf.float32)
        output_shapes = ([], self.data_shape, self.target_shape)

        tf_dataset = (
            tf.data.Dataset.from_generator(
                generator=self.iterate_dataset_randomly,
                output_types=output_types,
                output_shapes=output_shapes,
            )
            .repeat()
            .batch(self.batch_size)
            .prefetch(self.prefetch)
        )

        return tf_dataset.make_initializable_iterator()

    @property
    def idxes(self):
        """The indices of the current image batch.
        """
        return self._next_el_op[0]

    @property
    def images(self):
        """The tensorflow operator to get the current image batch.
        """
        return self._next_el_op[1]

    @property
    def targets(self):
        """The Tensorflow operator to get the current batch of masks.
        """
        return self._next_el_op[2]


class HDFReader(BaseReader):
    """Wrapper for HDF5 files for tensorflow. Creates a Tensorflow dataset.
    """

    def __init__(
        self,
        data_path,
        batch_size,
        group="train",
        dataset="images",
        target="masks",
        prefetch=1,
        preprocessor=None,
        name="data_reader",
    ):
        """Setup the data reader.

        This will not prepare the dequeue instances. The `prepare_data`
        and functions must be ran for the dequeue object to be created.

        Parameters:
        -----------
        data_path : str
            The path to the h5 file.
        batch_size : int
        group : str
            Internal path to the h5 group where the data is.
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
        """
        group = group if group[0] == "/" else "/" + group

        self.data_path, self.group = data_path, group
        self.data_name, self.target_name = dataset, target

        # Get tensorflow dataset and related objects
        super().__init__(
            batch_size=batch_size,
            prefetch=prefetch,
            preprocessor=preprocessor,
            name=name,
        )

    @property
    def h5_shape(self):
        return self._data_shape

    def get_shapes(self):
        with h5py.File(self.data_path, "r") as h5:
            g = h5[self.group]
            _len = g[self.data_name].shape[0]
            h5_shape = self._get_hdf5_dataset_shape(g, self.data_name)
            target_shape = self._get_hdf5_dataset_shape(g, self.target_name)

        return _len, h5_shape, target_shape

    def _get_hdf5_dataset_shape(self, h5, name):
        """Return the shape of the h5 dataset.
        """
        g = h5[self.group]
        if "shape" in g[name].attrs:
            return tuple(g[name].attrs["shape"])
        else:
            return tuple(g[name].shape[1:])

    def _get_input_and_target(self, idx, h5):
        """Extract the input and output data of given index from a HDF5 file.

        Parameters:
        -----------
        idx : int
            Index of the data point to return
        h5 : h5py.File or dict
            An opened HDF5 file to extract the data from or a dict emulating
            a h5py.File object.
        """
        group = h5[self.group]

        image = group[self.data_name][idx]
        image = image.reshape(self.h5_shape)

        target = group[self.target_name][idx]
        target = target.reshape(self.target_shape)

        return image, target

    def _iterate_dataset_randomly(self):
        idxes = np.arange(len(self))
        np.random.shuffle(idxes)
        with h5py.File(self.data_path, "r") as h5:
            for idx in idxes:
                image, target = self._get_input_and_target(idx, h5)
                yield idx, image, target
