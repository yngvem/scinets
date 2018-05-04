__author__ = 'Yngve Mardal Moe'
__email__ = 'yngve.m.moe@gmail.com'


import tensorflow as tf


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
                )
            )
        return accuracy

    
class BinaryClassificationEvaluator(ClassificationEvaluator):
    def __init__(self, network, scope='evaluator'):
        super().__init__(network, scope)
        with tf.variable_scope(scope+'/'):
            self.true_positives = self._init_true_positives()
            self.true_negatives = self._init_true_negatives()
            self.false_positives = self._init_false_positives()
            self.false_negatives = self._init_false_negatives()

            self.precision = self._init_precision()
            self.recall = self._init_recall()
            self.dice = self._init_dice()

    def _init_true_positives(self):
        with tf.variable_scope('true_positives'):
            return tf.count_nonzero(self.prediction * self.target,
                                    dtype=tf.float32)

    def _init_true_negatives(self):
        with tf.variable_scope('true_negatives'):
            return tf.count_nonzero((self.prediction - 1) * (self.target - 1),
                                    dtype=tf.float32)

    def _init_false_positives(self):
        with tf.variable_scope('fasle_positives'):
            return tf.count_nonzero(self.prediction * (self.target - 1),
                                    dtype=tf.float32)

    def _init_false_negatives(self):
        with tf.variable_scope('false_negatives'):
            return tf.count_nonzero((self.prediction - 1) * self.target,
                                    dtype=tf.float32)

    def _init_precision(self):
        with tf.variable_scope('precision'):
            return self.true_positives / (self.true_positives + self.false_positives)

    def _init_recall(self):
        with tf.variable_scope('recall'):
            return self.true_positives / (self.true_positives + self.false_negatives)

    def _init_dice(self):
        with tf.variable_scope('dice'):
            dice = (2*(self.precision*self.recall)) \
                    /(self.precision + self.recall)
        return dice

if __name__ == '__main__':
    pass

