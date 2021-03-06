"""
TODO: BATCH NORM WILL NOW THINK WE ARE TRAINING IF WE WANT TO COMPUTE PERFORMANCE
      METRICS ON TRAINING SET!!!
"""
__author__ = "Yngve Mardal Moe"
__email__ = "yngve.m.moe@gmail.com"


import itertools
import tensorflow as tf
import numpy as np
import h5py
from .._backend_utils import SubclassRegister


evaluator_register = SubclassRegister("model evaluator")


def get_evaluator(evaluator):
    return evaluator_register.get_item(evaluator)


@evaluator_register.link_base
class BaseEvaluator:
    def __init__(self, network, scope="evaluator"):
        self.network = network
        self.input = network.input
        self.loss = network.loss
        self.out = network.out
        self.true_out = network.true_out
        self._out_channels = self.out.get_shape().as_list()[-1]

        self.scope = scope


class ClassificationEvaluator(BaseEvaluator):
    def __init__(self, network, scope="evaluator"):
        super().__init__(network, scope)

        with tf.variable_scope(self.scope):
            self.target = self._init_target()
            self.probabilities = self._init_probabilities()
            self.prediction = self._init_prediction()
            self.accuracy = self._init_accuracy()

    def _init_probabilities(self):
        if "activation" in self.network.architecture[-1]:
            final_activation = self.network.architecture[-1]["activation"]
            if final_activation["operator"] == "Sigmoid":
                return self.out

        with tf.variable_scope("probabilities"):
            return tf.nn.sigmoid(self.out)

    def _init_prediction(self):
        with tf.variable_scope("prediction"):
            return tf.cast(self.probabilities > 0.5, tf.float32, name="prediction")

    def _init_target(self):
        with tf.variable_scope("target"):
            return tf.cast(self.network.true_out, tf.float32)

    def _init_accuracy(self):
        with tf.variable_scope("accuracy"):
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.prediction, self.target), tf.float32),
                axis=tf.range(1, tf.rank(self.prediction)),
            )
        return accuracy


class BinaryClassificationEvaluator(ClassificationEvaluator):
    def __init__(self, network, scope="evaluator"):
        super().__init__(network, scope)
        with tf.variable_scope(scope + "/"):
            self.num_elements = self._init_num_elements()
            self.true_positives = self._init_true_positives()
            self.true_negatives = self._init_true_negatives()
            self.false_positives = self._init_false_positives()
            self.false_negatives = self._init_false_negatives()

            self.sensitivity = self._init_sensitivity()
            self.specificity = self._init_specificity()
            self.precision = self._init_precision()
            self.recall = self.sensitivity
            self.dice = self._init_dice()

    def _init_num_elements(self):
        with tf.variable_scope("num_elements"):
            shape = self.out.get_shape().as_list()
            return np.prod(shape[1:])

    def _init_true_positives(self):
        with tf.variable_scope("true_positives"):
            true_positives = tf.count_nonzero(
                self.prediction * self.target,
                axis=tf.range(1, tf.rank(self.prediction)),
                dtype=tf.float32,
            )
            return true_positives

    def _init_true_negatives(self):
        with tf.variable_scope("true_negatives"):
            true_negatives = tf.count_nonzero(
                (self.prediction - 1) * (self.target - 1),
                axis=tf.range(1, tf.rank(self.prediction)),
                dtype=tf.float32,
            )
            return true_negatives

    def _init_false_positives(self):
        with tf.variable_scope("fasle_positives"):
            false_positives = tf.count_nonzero(
                self.prediction * (self.target - 1),
                axis=tf.range(1, tf.rank(self.prediction)),
                dtype=tf.float32,
            )
            return false_positives

    def _init_false_negatives(self):
        with tf.variable_scope("false_negatives"):
            false_negatives = tf.count_nonzero(
                (self.prediction - 1) * self.target,
                axis=tf.range(1, tf.rank(self.prediction)),
                dtype=tf.float32,
            )
            return false_negatives

    def _init_precision(self):
        with tf.variable_scope("precision"):
            return self.true_positives / (
                self.true_positives + self.false_positives + 1e-8
            )

    def _init_sensitivity(self):
        with tf.variable_scope("sensitivity"):
            return self.true_positives / (
                self.true_positives + self.false_negatives + 1e-8
            )

    def _init_specificity(self):
        with tf.variable_scope("specificity"):
            return self.true_negatives / (
                self.true_negatives + self.false_positives + 1e-8
            )

    def _init_dice(self):
        with tf.variable_scope("dice"):
            dice = (2 * self.true_positives) / (
                2 * self.true_positives
                + self.false_negatives
                + self.false_positives
                + 1e-8
            )
        return dice


class NetworkTester:
    """Used to compute the final performance of the network.

    This class will iterate through the whole training, validation or test
    dataset and compute the average performance metrics and their standard
    deviation.
    """

    def __init__(self, metrics, dataset, evaluator, is_training, is_testing):
        self.metrics = metrics
        self.performance_ops = {
            metric: getattr(evaluator, metric) for metric in metrics
        }
        self.dataset, self.evaluator = dataset, evaluator
        self.is_training, self.is_testing = is_training, is_testing

    def get_dataset(self, dataset_type):
        dataset = f"{dataset_type}_data_reader"
        return getattr(self.dataset, dataset)

    def get_numits(self, dataset_type):
        dataset = self.get_dataset(dataset_type)
        data_len = len(dataset)
        batch_size = dataset.batch_size
        return int(np.ceil(data_len / batch_size))

    def get_feed_dict(self, dataset):
        """Return the `feed_dict` used to choose the correct dataset type.

        Parameters:
        -----------
        dataset_type : str
            Used to decide which dataset to use, must be `train`, `val`
            or `test`.

        Returns:
        --------
        dict :
            The feed dict to use when running the performance metrics.
        """
        if dataset == "train":
            return {self.is_training: True}
        elif dataset == "val":
            return {self.is_training: False, self.is_testing: False}
        elif dataset == "test":
            return {self.is_training: False, self.is_testing: True}
        else:
            raise ValueError("`dataset` must be either `train`, `val` or `test`")

    def _compute_performances(self, performances, metric):
        # performances = self._join_performance_metric(performInes, metric)
        return performances[metric].mean(), performances[metric].std(ddof=1)

    def _create_performance_dict(self, performances):
        return {
            metric: self._compute_performances(performances, metric)
            for metric in performances
        }

    def test_model(self, dataset_type, sess):
        """Compute the performance metrics using the specified evaluator.

        Arguments:
        ----------
        dataset_type : str
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
        feed_dict = self.get_feed_dict(dataset_type)
        num_its = self.get_numits(dataset_type)
        dataset = self.get_dataset(dataset_type)
        print(len(dataset))

        performances = []
        for i in range(num_its):
            performances.append(sess.run(self.performance_ops, feed_dict=feed_dict))

        performance_dict = {}
        for metric in self.metrics:
            performance_dict[metric] = np.array(list(
                itertools.chain(*[performance[metric] for performance in performances])
            ))
            performance_dict[metric] = performance_dict[metric][: len(dataset)]

        return self._create_performance_dict(performance_dict)

    def _init_output_file(self, dataset_type, filename):
        """Initiate a dataset output file.
        """
        dataset = self.get_dataset(dataset_type)
        data_len = len(dataset)

        with h5py.File(filename, "w") as h5:
            data_group = h5.create_group(dataset_type)
            data_group.create_dataset("idxes", dtype=np.int32, shape=[data_len])
            data_group.create_dataset(
                "images", dtype=np.float32, shape=(data_len, *dataset.data_shape)
            )
            data_group.create_dataset(
                "prediction", dtype=np.float32, shape=(data_len, *dataset.target_shape)
            )
            data_group.create_dataset(
                "masks", dtype=np.float32, shape=(data_len, *dataset.target_shape)
            )
            for performance_metric in self.performance_ops:
                data_group.create_dataset(
                    performance_metric, dtype=np.float32, shape=(data_len,)
                )

    def _update_outputs(self, outputs, it_num, h5, dataset_type):
        """Save the latest batch output to the h5file at the correct location.
        """
        dataset = self.get_dataset(dataset_type)
        data_len = len(dataset)
        batch_size = dataset.batch_size
        prev_length = it_num * batch_size
        new_length = (it_num + 1) * batch_size

        # Update new length if dataset is fully iterated through
        if new_length > data_len:
            extra_evals = new_length - data_len
            new_length = data_len

            for key, output in outputs.items():
                outputs[key] = outputs[key][:-extra_evals]

        # Insert new evaluations
        group = h5[f"{dataset_type}"]
        for output_type, output in outputs.items():
            group[output_type][prev_length:new_length] = outputs[output_type]

    def save_outputs(self, dataset_type, filename, sess, save_probabilities=False):
        num_its = self.get_numits(dataset_type)
        feed_dict = self.get_feed_dict(dataset_type)

        prediction_op = self.evaluator.prediction
        if save_probabilities:
            prediction_op = self.evaluator.probabilities

        run_ops = {
            "prediction": prediction_op,
            "idxes": self.dataset.idxes,
            "images": self.dataset.data,
            "masks": self.dataset.target,
            **self.performance_ops,
        }

        self._init_output_file(dataset_type=dataset_type, filename=filename)
        with h5py.File(filename, "a") as h5:
            for it in range(num_its):
                outputs = sess.run(run_ops, feed_dict=feed_dict)
                self._update_outputs(
                    outputs, it_num=it, h5=h5, dataset_type=dataset_type
                )


if __name__ == "__main__":
    pass
