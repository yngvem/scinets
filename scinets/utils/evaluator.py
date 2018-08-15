__author__ = 'Yngve Mardal Moe'
__email__ = 'yngve.m.moe@gmail.com'


import tensorflow as tf
import numpy as np


class ClassificationEvaluator:
    def __init__(self, network, scope='evaluator'):
        self.network = network
        self.input = network.input
        self.loss = network.loss
        self.out = network.out
        self.true_out = network.true_out

        self._out_channels = self.out.get_shape().as_list()[-1]

        with tf.variable_scope(scope):
            self.target = self._init_target()
            self.probabilities = self._init_probabilities()
            self.prediction = self._init_prediction()
            self.accuracy = self._init_accuracy()

    def _init_probabilities(self):
        final_activation = self.network.architecture[-1]['activation']
        if (final_activation == 'sigmoid'):
            return self.out

        with tf.variable_scope('probabilities'):
            return tf.nn.sigmoid(self.out)

    def _init_prediction(self):
        with tf.variable_scope('prediction'):
            return tf.cast(
                self.probabilities > 0.5,
                tf.float32,
                name='prediction'
            )

    def _init_target(self):
        with tf.variable_scope('target'):
            return tf.cast(self.network.true_out, tf.float32)

    def _init_accuracy(self):
        with tf.variable_scope('accuracy'):
            accuracy = tf.reduce_mean(
                tf.cast(
                    tf.equal(self.prediction, self.target),
                    tf.float32
                ),
                axis=tf.range(1, tf.rank(self.prediction))
            )
        return accuracy


class BinaryClassificationEvaluator(ClassificationEvaluator):
    def __init__(self, network, scope='evaluator'):
        super().__init__(network, scope)
        with tf.variable_scope(scope+'/'):
            self.num_elements = self._init_num_elements()
            self.true_positives = self._init_true_positives()
            self.true_negatives = self._init_true_negatives()
            self.false_positives = self._init_false_positives()
            self.false_negatives = self._init_false_negatives()

            self.precision = self._init_precision()
            self.recall = self._init_recall()
            self.dice = self._init_dice()

    def _init_num_elements(self):
        with tf.variable_scope('num_elements'):
            shape = self.out.get_shape().as_list()
            return np.prod(shape[1:])

    def _init_true_positives(self):
        with tf.variable_scope('true_positives'):
            true_positives = tf.count_nonzero(
                self.prediction * self.target,
                axis=tf.range(1, tf.rank(self.prediction)),
                dtype=tf.float32
            )
            return true_positives/self.num_elements

    def _init_true_negatives(self):
        with tf.variable_scope('true_negatives'):
            true_negatives =  tf.count_nonzero(
                (self.prediction - 1) * (self.target - 1),
                axis=tf.range(1, tf.rank(self.prediction)),
                dtype=tf.float32
            )
            return true_negatives/self.num_elements

    def _init_false_positives(self):
        with tf.variable_scope('fasle_positives'):
            false_positives = tf.count_nonzero(
                self.prediction * (self.target - 1),
                axis=tf.range(1, tf.rank(self.prediction)),
                dtype=tf.float32
            )
            return false_positives/self.num_elements

    def _init_false_negatives(self):
        with tf.variable_scope('false_negatives'):
            false_negatives = tf.count_nonzero(
                (self.prediction - 1) * self.target,
                axis=tf.range(1, tf.rank(self.prediction)),
                dtype=tf.float32
            )
            return false_negatives/self.num_elements

    def _init_precision(self):
        with tf.variable_scope('precision'):
            return self.true_positives / (self.true_positives + self.false_positives)

    def _init_recall(self):
        with tf.variable_scope('recall'):
            return self.true_positives / (self.true_positives + self.false_negatives)

    def _init_dice(self):
        with tf.variable_scope('dice'):
            dice = ((2*self.true_positives) 
                 / (2*self.true_positives + self.false_negatives
                    + self.false_positives))
        return dice


if __name__ == '__main__':
    pass
