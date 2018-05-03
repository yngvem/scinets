"""
TODO: Remove unnecessary batch sizes!!!
"""


__author__ = 'Yngve Mardal Moe'
__email__ = 'yngve.m.moe@gmail.com'


import numpy as np
import tftables
import h5py
import tensorflow as tf
from contextlib import contextmanager


class HDFData:
    """Wrapper for HDF5 files for tensorflow. Creates a Tensorflow dataset.
    """
    def __init__(self, data_path, batch_size, group='train', dataset='images',
                 target='masks', name='data_reader'):
        """Setup the data reader.

        This will not prepare the dequeue instances. The `prepare_data`
        and functions must be ran for the dequeue object to be created.

        Parameters:
        -----------
        data_path : str
            The path to the h5 file.
        group : str
            Internal path to the h5 group where the data is.
        dataset : str
            Name of the h5 dataset which contains the data points.
        target : str
            Name of the h5 dataset which contains the labels (or whichever
            output that is wanted from the network).
        """
        group = group if group[0] == '/' else '/'+ group

        self.data_path = str(data_path)
        self.group = group
        self.data_name = dataset
        self.target_name = target

        # The order in which the samples was drawn
        self.trained_order = []

        # Get the shapes of the input and output data
        self.shapes = {}
        with h5py.File(data_path) as h5:
            g = h5[self.group]
            self._len = g[dataset].shape[0]

            if 'shape' in g[dataset].attrs:
                self.shapes[dataset] = tuple(g[dataset].attrs['shape'])
            else:
                self.shapes[dataset] = tuple(g[dataset].shape[1:])

            if 'shape' in g[target].attrs:
                self.shapes[target] = tuple(g[target].attrs['shape'])
            else:
                self.shapes[target] = tuple(g[target].shape[1:])

        # Get tensorflow dataset and related objects
        with tf.variable_scope(name):
            self._tf_dataset = tf.data.Dataset.from_generator(
                generator=self._iterate_dataset_randomly_forever,
                output_types=(tf.int16, tf.float32, tf.float32),
                output_shapes=([], self.shapes[dataset], self.shapes[target])
            )
            self._tf_dataset = self._tf_dataset.repeat().batch(5)
            self._tf_iterator = self._tf_dataset.make_one_shot_iterator()
            self._next_el_op = self._tf_iterator.get_next()

    def _get_image_and_target(self, idx, h5):
        """Extract the input and output data of given index from a HDF5 file.

        Parameters:
        -----------
        idx : int
            Index of the data point to return
        h5 : h5py.File
            An opened HDF5 file to extract the data from.
        """
        group = h5[self.group]

        image = group[self.data_name][idx]
        image = image.reshape(self.shapes[self.data_name])

        target = group[self.target_name][idx]
        target = target.reshape(self.shapes[self.target_name])

        return image, target

    def _iterate_dataset_randomly_forever(self):
        """Infinite generator, returns a random image from the dataset.

        It is not completely random, rather, the dataset is shuffled and
        every element is yielded before shuffling it again.
        """
        while True:
            for idx, image, target in self._iterate_dataset_randomly_once():
                yield idx, image, target

    def _iterate_dataset_randomly_once(self):
        """Iterates through the dataset in random order
        """
        with h5py.File(self.data_path, 'r') as h5:
            idxes = np.arange(len(self))
            np.random.shuffle(idxes)
            for idx in idxes:
                yield (idx, *self._get_image_and_target(idx, h5))

    def __len__(self):
        return self._len

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


class HDFReader:
    def __init__(self, data_path, batch_size, train_group='train', 
                 val_group='validation', test_group='test', dataset='images',
                 target='masks', is_training=None, is_testing=None):
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
        is_training : tensorflow.Placeholder(bool, [])
            Placeholder used to specify whether the training data should be
            the output or not.
        is_testing : tensorflow.Placeholder(bool, [])
            Placeholder used to specify whether the test or validation data
            should be used. If `is_training` is True, this is ignored.
        """
        if '__iter__' not in dir(batch_size):  # Check if batch_size is iterable
            batch_size = [batch_size]*3

        if is_training is None:
            is_training = tf.placeholder_with_default(True, shape=[], 
                                                      name='is_training')
        if is_testing is None:
            is_testing = tf.placeholder_with_default(False, shape=[],
                                                     name='is_testing')
        self.data_path = str(data_path)

        self.is_training = is_training
        self.is_testing = is_testing
        self.batch_size = batch_size
        with tf.variable_scope('data_loader'):
            self.train_data_reader = HDFData(
                data_path=data_path,
                batch_size=batch_size[0],
                group=train_group,
                dataset=dataset,
                target=target,
                name='train_reader'
            )
            self.val_data_reader = HDFData(
                data_path=data_path,
                batch_size=batch_size[1],
                group=val_group,
                dataset=dataset,
                target=target,
                name='val_reader'
            )
            self.test_data_reader = HDFData(
                data_path=data_path,
                batch_size=batch_size[2],
                group=test_group,
                dataset=dataset,
                target=target,
                name='test_reader'
            )
            self._create_conditionals()

    def _create_conditionals(self):
        """Set up conditional operators specifying which datasets to use.
        """
        with tf.variable_scope('test_train_or_val'):
            val_test_data = tf.cond(
                self.is_testing,
                true_fn=lambda: self.test_data,
                false_fn=lambda: self.val_data,
                name='use_test_data'
            )
            self.conditional_data = tf.cond(
                self.is_training,
                true_fn=lambda: self.train_data,
                false_fn=lambda: val_test_data,
                name='use_train_data'
            )

            val_test_target = tf.cond(
                self.is_testing,
                true_fn=lambda: self.test_target,
                false_fn=lambda: self.val_target,
                name='use_test_target'
            )
            self.conditional_target = tf.cond(
                self.is_training,
                true_fn=lambda: self.train_target,
                false_fn=lambda: val_test_target,
                name='use_train_target'
            )

            val_test_idxes = tf.cond(
                self.is_testing,
                true_fn=lambda: self.test_idxes,
                false_fn=lambda: self.val_idxes,
                name='use_test_idxes'
            )
            self.conditional_idxes = tf.cond(
                self.is_training,
                true_fn=lambda: self.train_idxes,
                false_fn=lambda: val_test_idxes,
                name='use_train_idxes'
            )

    @property
    def train_data(self):
        return self.train_data_reader.images
    
    @property
    def train_target(self):
        return self.train_data_reader.targets

    @property
    def train_idxes(self):
        return self.train_data_reader.idxes

    @property
    def val_data(self):
        return self.val_data_reader.images
    
    @property
    def val_target(self):
        return self.val_data_reader.targets

    @property
    def val_idxes(self):
        return self.val_data_reader.idxes

    @property
    def test_data(self):
        return self.test_data_reader.images
    
    @property
    def test_target(self):
        return self.test_data_reader.targets

    @property
    def test_idxes(self):
        return self.test_data_reader.idxes

    @property
    def data(self):
        return self.conditional_data

    @property
    def target(self):
        return self.conditional_target

    @property
    def idxes(self):
        return self.conditional_idxes


if __name__ == '__main__':
    pass

