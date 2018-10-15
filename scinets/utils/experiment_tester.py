"""

"""


__author__ = 'Yngve Mardal Moe'
__email__ = 'yngve.m.moe@gmail.com'


import numpy as np
import tensorflow as tf


class ExperimentTester:
    def __init__(self, experiment, evaluator):
        self.experiment = experiment
        self.evaluator = evaluator

    def get_numits(self, dataset):
        dataset = f'{dataset}_data_reader'
        dataset = getattr(self.experiment.dataset, dataset)
        data_len = len(dataset)
        batch_size = dataset.batch_size

        return np.ceil(data_len/batch_size)
    
    def get_placeholders(self, dataset, train_name='is_training',
                         test_name='is_testing'):
        if dataset == 'train':
            return {train_name: True}
        elif dataset == 'val':
            return {train_name: False, test_name: False}
        elif dataset == test:
            return {train_name: False, test_name: True}
        else:
            raise ValueError('`dataset` must be either `train`, `val` or `test`')

    def test_experiment(self, dataset):
        placeholders = self.get_placeholders(datset)
        num_its = self.get_numits(self, dataset)

        # TODO: Finish this
        





if __name__ == '__main__':
    pass

