import tensorflow as tf
from abc import ABC, abstractmethod
from .._backend_utils import SubclassRegister


loss_register = SubclassRegister("loss")


def get_loss(loss):
    return loss_register.get_item(loss)


@loss_register.link_base
class BaseLoss(ABC):
    def __init__(self, name="loss_function"):
        self.name = name

    def __call__(self, prediction, target, **kwargs):
        return self._build_loss(prediction, target, **kwargs)

    @abstractmethod
    def _build_loss(self, prediction, target):
        pass


class SoftmaxCrossEntropyWithLogits(BaseLoss):
    def _build_loss(self, prediction, target):
        return tf.nn.softmax_cross_entropy_with_logits(
            labels=target, logits=prediction, name=self.name
        )


class SigmoidCrossEntropyWithLogits(BaseLoss):
    def _build_loss(self, prediction, target):
        return tf.nn.sigmoid_cross_entropy_with_logits(
            labels=target, logits=prediction, name=self.name
        )


class BinaryFBeta(BaseLoss):
    def __init__(self, beta, name="loss_function"):
        self.name = name
        self.beta = beta

    def _build_loss(self, prediction, target):
        size = len(prediction.get_shape().as_list())
        reduce_ax = list(range(1, size))
        eps = 1e-8

        with tf.variable_scope(self.name):
            true_positive = tf.reduce_sum(prediction * target, axis=reduce_ax)
            target_positive = tf.reduce_sum(tf.square(target), axis=reduce_ax)
            predicted_positive = tf.reduce_sum(tf.square(prediction), axis=reduce_ax)

            fb_numerator = (1 + self.beta ** 2) * true_positive + eps
            fb_denominator = (
                (self.beta ** 2) * target_positive + predicted_positive + eps
            )

            return 1 - fb_numerator / fb_denominator


class BinaryDice(BinaryFBeta):
    def __init__(self, name="loss_function"):
        self.name = name
        self.beta = 1
