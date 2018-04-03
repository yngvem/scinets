"""

"""


__author__ = 'Yngve Mardal Moe'
__email__ = 'yngve.m.moe@gmail.com'


import numpy as np
import tftables
import h5py
import tensorflow as tf
from contextlib import contextmanager


def join(a, *args):
    """Join internal HDF5 paths.
    """
    a = str(a)
    for subdir in args:
        a += '/' + str(subdir)
    return a


class DataReader:
    """Wrapper for HDF5 files for tensorflow.
    """

    def __init__(self, data_path, train_group='train', test_group='test',
                 val_group='validation', dataset='images', target='masks'):
        """Setup the data reader.

        This will not prepare the dequeue instances. The `prepare_train_data`
        and `prepare_test_data` functions must be ran for the train dequeue
        and test dequeue objects to be created (one can run the 
        `prepare_trian_data` if one only needs the train deuqueue instances
        and vice versa).

        The HDF5 file should have one group for the training set, one for the
        test set and one for the validation set. The input data and
        and the labels should be in two different HDF5 dataset.

        Parameters:
        -----------
        data_path : str
            The path to the h5 file.
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
        """
        self.data_path = str(data_path)
        self.group_names = {
            'train': train_group,
            'val': val_group,
            'test': test_group
        }
        self.data_names = {'dataset': dataset, 'target': target}
        
        with h5py.File(data_path) as h5:
            self.shapes = self._get_shapes(h5)

    def _get_shapes(self, dataset_paths):
        """Create a dictionary with the shapes of the entries in the data tables.
        """
        dataset_paths = [
            join(group_name, dataset_name)
                for group_name in self.group_names.items()
                    for dataset_name in self.data_names.items()
        ]
        shapes = {}
        for dataset_path in dataset_paths:
            if dataset_path not in h5group:
                continue
            if 'shape' in h5group[dataset_path].attrs:
                shapes[dataset_path] = h5group[dataset_path].attrs['shape']
        return shapes

    def prepare_group_loader(self, group, batch_size, queue_size, n_procs=3):
        """Prepare the FIFO loader and dequeue objects for a given group.
        """
        reader = tftables.open_file(self.data_path,
                                    batch_size=batch_size)
        
        data_placeholder = _train_reader.get_batch(
            path=join(self.group_names[group], self.data_names['dataset']),
            block_size=2*batch_size+1   # Found this in tftables unittests
        )
        target_placeholder = _train_reader.get_batch(
            path=join(self.group_names[group], self.data_names['target']),
            block_size=2*batch_size+1   # Found this in tftables unittests
        )
        placeholders = {
            'data': data_placeholder,
            'target': target_placeholder
        }

        loader = reader.get_fifoloader(
            queue_size=queue_size
            inputs = [data_placeholder, target_placeholder],
            threads=1
        )
            
        dequeue = self._train_loader.dequeue()

        return reader, placeholders, loader, dequeue

    def prepare_train_data(batch_size, queue_size, n_procs=1):
        """Create dequeue objects for the training data.

        Parameters:
        -----------
        batch_size : int
            Batch size.
        queue_size : int
            Maximum number of data points to keep in RAM at the same time.
        n_procs : int (optional)
            Number of processes to use for reading data.
        """
        reader, placeholders, loader, dequeue = self.prepare_group_loader(
            group='train',
            batch_size=batch_size,
            queue_size=queue_size,
            n_procs=n_procs
        )
        self.train_reader = reader
        self.train_placeholders = placeholders
        self.train_loader = loader
        self.train_dequeue = dequeue

    def prepare_test_data(batch_size, queue_size, n_procs=1):
        """Create dequeue objects for the test data.

        Parameters:
        -----------
        batch_size : int
            Batch size.
        queue_size : int
            Maximum number of data points to keep in RAM at the same time.
        n_procs : int (optional)
            Number of processes to use for reading data.
        """
        reader, placeholders, loader, dequeue = self.prepare_group_loader(
            group='test',
            batch_size=batch_size,
            queue_size=queue_size,
            n_procs=n_procs
        )
        self.test_reader = reader
        self.test_placeholders = placeholders
        self.test_loader = loader
        self.test_dequeue = dequeue
    
    def prepare_val_data(batch_size, queue_size, n_procs=1):
        """Create dequeue objects for the validation data.

        Parameters:
        -----------
        batch_size : int
            Batch size.
        queue_size : int
            Maximum number of data points to keep in RAM at the same time.
        n_procs : int (optional)
            Number of processes to use for reading data.
        """
        reader, placeholders, loader, dequeue = self.prepare_group_loader(
            group='val',
            batch_size=batch_size,
            queue_size=queue_size,
            n_procs=n_procs
        )
        self.val_reader = reader
        self.val_placeholders = placeholders
        self.val_loader = loader
        self.val_dequeue = dequeue

    def _reshape(self, dataset_path, dequeue, idx):
        if dataset_path in self.shapes:
            return tf.reshape(deuqueue[idx], self.shapes[dataset_path])
        return dequeue[idx]

    @property
    def train_data(self):
        path = join(self.group_names['train'], self.data_names['dataset'])
        return self._reshape(path, self.train_dequeue, 0)
    
    @property
    def train_masks(self):
        path = join(self.group_names['train'], self.data_names['target'])
        return self._reshape(path, self.train_dequeue, 1)

    @property
    def test_data(self):
        path = join(self.group_names['test'], self.data_names['dataset'])
        return self._reshape(path, self.test_dequeue, 0)
    
    @property
    def test_masks(self):
        path = join(self.group_names['test'], self.data_names['target'])
        return self._reshape(path, self.test_dequeue, 1)

    @property
    def val_data(self):
        path = join(self.group_names['val'], self.data_names['dataset'])
        return self._reshape(path, self.val_dequeue, 0)
    
    @property
    def val_masks(self):
        path = join(self.group_names['val'], self.data_names['target'])
        return self._reshape(path, self.val_dequeue, 1)

    @contextmanager
    def train_loader(self, sess):
        self.train_loader.start(sess)
        yield
        self.train_loader.stop(sess)
        self.train_loader.monitor_thread = None

    @contextmanager
    def test_loader(self, sess):
        self.test_loader.start(sess)
        yield
        self.test_loader.stop(sess)
        self.test_loader.monitor_thread = None

    @contextmanager
    def val_loader(self, sess):
        self.val_loader.start(sess)
        yield
        self.val_loader.stop(sess)
        self.val_loader.monitor_thread = None

    @contextmanager
    def train_val_loader(self, sess):
        self.train_loader.start(sess)
        self.val_loader.start(sess)
        yield
        self.train_loader.stop(sess)
        self.train_loader.monitor_thread = None
        self.val_loader.stop(sess)
        self.val_loader.monitor_thread = None

    @contextmanager
    def test_val_loader(self, sess):
        self.test_loader.start(sess)
        self.val_loader.start(sess)
        yield
        self.test_loader.stop(sess)
        self.test_loader.monitor_thread = None
        self.val_loader.stop(sess)
        self.val_loader.monitor_thread = None
        
    @contextmanager
    def all_loaders(self, sess):
        self.train_loader.start(sess)
        self.test_loader.start(sess)
        self.val_loader.start(sess)
        yield
        self.train_loader.stop(sess)
        self.train_loader.monitor_thread = None
        self.test_loader.stop(sess)
        self.test_loader.monitor_thread = None
        self.val_loader.stop(sess)
        self.val_loader.monitor_thread = None

    
if __name__ == '__main__':
    pass

