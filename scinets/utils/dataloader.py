"""
TODO: Remove unnecessary batch sizes!!!
"""


__author__ = 'Yngve Mardal Moe'
__email__ = 'yngve.m.moe@gmail.com'


import numpy as np
import h5py
import tensorflow as tf
from contextlib import contextmanager
from tensorflow.examples.tutorials.mnist import input_data


class HDFData:
    """Wrapper for HDF5 files for tensorflow. Creates a Tensorflow dataset.
    """
    def __init__(self, data_path, batch_size, group='train', dataset='images',
                 target='masks', prefetch=1, keep_in_ram=False,
                 name='data_reader'):
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
        prefetch : int
            Number of batches to load at the same time as a training step is
            performed. Used to reduce waiting time between training steps.
        keep_in_ram : bool
            If true, the whole dataset will be loaded to RAM.
        """
        group = group if group[0] == '/' else '/'+ group

        self.data_path = str(data_path)
        self.group = group
        self.data_name = dataset
        self.target_name = target
        self.keep_in_ram = keep_in_ram
        self.batch_size = batch_size
        self.prefetch = prefetch

        # The order in which the samples was drawn
        self.trained_order = []

        # Get the shapes of the input and output data
        with h5py.File(data_path) as h5:
            g = h5[self.group]
            self._len = g[dataset].shape[0]
            self.data_shape = self._get_dataset_shape(g, dataset)
            self.target_shape = self._get_dataset_shape(g, target)

        if keep_in_ram:
            self.dataset = self._extract_dataset_as_dict()

        # Get tensorflow dataset and related objects
        with tf.variable_scope(name):
            self._next_el_op = self._get_next_el_op()

    def _get_dataset_shape(self, h5, name):
        g = h5[self.group]
        if 'shape' in g[name].attrs:
            return tuple(g[name].attrs['shape'])
        else:
            return tuple(g[name].shape[1:])

    def _get_next_el_op(self):
        output_types = (tf.int16, tf.float32, tf.float32)
        output_shapes = ([], self.data_shape, self.target_shape)

        tf_dataset = tf.data.Dataset.from_generator(
            generator=self.iterate_dataset_randomly,
            output_types=output_types,
            output_shapes=output_shapes
            ).repeat().batch(self.batch_size).prefetch(self.prefetch)

        tf_iterator = tf_dataset.make_one_shot_iterator()
        return tf_iterator.get_next()

    def _get_image_and_target(self, idx, h5):
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
        image = image.reshape(self.data_shape)

        target = group[self.target_name][idx]
        target = target.reshape(self.target_shape)

        return image, target

    def iterate_dataset_randomly(self):
        """Iterates through the dataset in random order
        """
        if self.keep_in_ram:
            yield from self._iterate_dataset_randomly(self.data_dict)
        else:
            with h5py.File(self.data_path, 'r') as h5:
                yield from self._iterate_dataset_randomly(h5)
            
    def _iterate_dataset_randomly(self, dataset):
        idxes = np.arange(len(self))
        np.random.shuffle(idxes)
        for idx in idxes:
            yield (idx, *self._get_image_and_target(idx, dataset))

    def _extract_dataset_as_dict(self):
        """Returns a dictionary of numpy arrays that contain the dataset.

        The returned dictionary has one key, which is equal to the group name.
        The value of this key is again another dictionary with two key-value 
        paris, one for the input data and one for the targets.

        Returns:
        --------
        dict :
            Dictionary containing the contents of the specified HDF5 group.
        """
        group_dict = {}
        with h5py.File(self.data_path, 'r') as h5:
            group = h5[self.group]

            images = group[self.data_name][:]
            group_dict[self.data_name] = image.reshape(-1, *self.data_shape)

            targets = group[self.target_name][:]
            group_dict[self.target_name] = target.reshape(-1, *self.target_shape)
        return {self.group: group_dict}

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


class HDFDataset:
    def __init__(self, data_path, batch_size, train_group='train', 
                 val_group='validation', test_group='test', dataset='images',
                 target='masks', prefetch=1, keep_in_ram=False,
                 is_training=None, is_testing=None):
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
        keep_in_ram : bool
            If true, the whole dataset will be loaded to RAM.
        is_training : tensorflow.Placeholder(bool, [])
            Placeholder used to specify whether the training data should be
            the output or not.
        is_testing : tensorflow.Placeholder(bool, [])
            Placeholder used to specify whether the test or validation data
            should be used. If `is_training` is True, this is ignored.
        """
        if isinstance(batch_size, int):  # Check if batch_size is iterable
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
                prefetch=prefetch,
                keep_in_ram=keep_in_ram,
                name='train_reader'
            )
            self.val_data_reader = HDFData(
                data_path=data_path,
                batch_size=batch_size[1],
                group=val_group,
                dataset=dataset,
                target=target,
                prefetch=prefetch,
                keep_in_ram=keep_in_ram,
                name='val_reader'
            )
            self.test_data_reader = HDFData(
                data_path=data_path,
                batch_size=batch_size[2],
                group=test_group,
                dataset=dataset,
                target=target,
                prefetch=prefetch,
                keep_in_ram=keep_in_ram,
                name='test_reader'
            )
            self._create_conditionals()

    def _create_conditionals(self):
        """Set up conditional operators specifying which datasets to use.
        """
        with tf.variable_scope('test_train_or_val'):
            val_test_data = tf.cond(
                self.is_testing,
                true_fn=lambda: self._test_data,
                false_fn=lambda: self._val_data,
                name='use_test_data'
            )
            self._conditional_data = tf.cond(
                self.is_training,
                true_fn=lambda: self._train_data,
                false_fn=lambda: val_test_data,
                name='use_train_data'
            )

            val_test_target = tf.cond(
                self.is_testing,
                true_fn=lambda: self._test_target,
                false_fn=lambda: self._val_target,
                name='use_test_target'
            )
            self._conditional_target = tf.cond(
                self.is_training,
                true_fn=lambda: self._train_target,
                false_fn=lambda: val_test_target,
                name='use_train_target'
            )

            val_test_idxes = tf.cond(
                self.is_testing,
                true_fn=lambda: self._test_idxes,
                false_fn=lambda: self._val_idxes,
                name='use_test_idxes'
            )
            self._conditional_idxes = tf.cond(
                self.is_training,
                true_fn=lambda: self._train_idxes,
                false_fn=lambda: val_test_idxes,
                name='use_train_idxes'
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


class MNISTDataset(HDFDataset):
    def __init__(self, batch_size=128, is_training=None, is_testing=None,
                 name='scope'):
        if is_training is None:
            is_training = tf.placeholder_with_default(True, shape=[], 
                                                      name='is_training')
        self.is_training = is_training
        self.batch_size = batch_size
        self.epoch = 40000

        self._curr_it_num = 0
        self._data_it_num = 0
        self._labels_it_num = 0
        with tf.variable_scope(name):
            self._train_next_el_op = self._get_next_el_op(self._iterate_train_dataset)
            self._val_next_el_op = self._get_next_el_op(self._iterate_val_dataset)

            self._create_conditionals()

    def _get_next_el_op(self, generator):
        dataset = tf.data.Dataset.from_generator(
            generator=generator,
            output_types=(tf.int16, tf.float32, tf.float32),
            output_shapes=([], [self.batch_size, 28, 28, 1],
                            [self.batch_size, 1, 1, 10])
        )
        tf_iterator = dataset.make_one_shot_iterator()
        return tf_iterator.get_next()

    def _iterate_train_dataset(self):
        dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
        while True:
            x, y = dataset.train.next_batch(self.batch_size)
            x = x.reshape((self.batch_size, 28, 28, 1))
            y = y.reshape((self.batch_size, 1, 1, 10))
            yield 0, x, y

    def _iterate_val_dataset(self):
        dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
        while True:
            x, y = dataset.test.next_batch(self.batch_size)
            x = x.reshape((self.batch_size, 28, 28, 1))
            y = y.reshape((self.batch_size, 1, 1, 10))
            yield 0, x, y

    def _create_conditionals(self):
        """Set up conditional operators specifying which datasets to use.
        """
        with tf.variable_scope('test_train_or_val'):
            self._conditional_data = tf.cond(
                self.is_training,
                true_fn=lambda: self._train_data,
                false_fn=lambda: self._val_data,
                name='use_train_data'
            )

            self._conditional_target = tf.cond(
                self.is_training,
                true_fn=lambda: self._train_target,
                false_fn=lambda: self._val_target,
                name='use_train_target'
            )

            self._conditional_idxes = tf.cond(
                self.is_training,
                true_fn=lambda: self._train_idxes,
                false_fn=lambda: self._val_idxes,
                name='use_train_idxes'
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





if __name__ == '__main__':
    pass

