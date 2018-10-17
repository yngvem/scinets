"""
TODO: BATCH NORM WILL NOW THINK WE ARE TRAINING IF WE WANT TO COMPUTE PERFORMANCE
      METRICS ON TRAINING SET!!!
"""


__author__ = 'Yngve Mardal Moe'
__email__ = 'yngve.m.moe@gmail.com'


import numpy as np
import tensorflow as tf


class ExperimentTester:
    def __init__(self, metrics, dataset, evaluator):
        self.metrics = metrics
        self.performance_ops = [
            {metric: getattr(evaluator, metric)} for metric in metrics
        ]
        self.dataset = dataset
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
        elif dataset == 'test':
            return {train_name: False, test_name: True}
        else:
            raise ValueError('`dataset` must be either `train`, `val` or `test`')

    @staticmethod
    def _join_performance_metric(performances, metric):
        return np.concatenate([batch[metric] for batch in performance], axis=0)

    @staticmethod
    def _compute_performances(performances, metric):
        performances = self._join_performance_metric(performances, metric)
        return performance.mean(), performance.std(ddof=1)

    @staticmethod
    def _create_performance_dict(performances):
        return {
            metric: self._compute_performances(perfomances, metric)
                for metric in performances[0]
        }

    def test_model(self, data_type, sess):
        """Compute the performance metrics using the specified evaluator.

        Arguments:
        ----------
        data_type : str
            Specifies which dataset to use, should be equal to `train`, 
            `val`, or `test`
        sess : tensorflow.Session
            The specified tensorflow session to use. All variables must be
            initialised beforehand.
        Returns:
        --------
        dict : 
            Dictionary specifying the average and standard deviation of all
            specified performance metrics. The keys are the metric names
            and the values are tuples where the first element is the mean
            and the second is the standard deviation.
        """
        feed_dict = self.get_feed_dict(data_type)
        num_its = self.get_numits(self, data_type)

        performance = []
        for i in range(num_its):
            performance.append(
                sess.run(self.performance_metrics, feed_dict=feed_dict)
            )

        return self._create_performance_dict(performances)


if __name__ == '__main__':
    pass

