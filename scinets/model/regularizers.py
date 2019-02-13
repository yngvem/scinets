from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
from .._backend_utils import SubclassRegister


regularizer_register = SubclassRegister("layer")


def get_regularizer(regularizer):
    return regularizer_register.get_item(regularizer)


@regularizer_register.link_base
class BaseRegularizer:
    @abstractmethod
    def _build_regularizer(self, x):
        pass

    def __call__(self, x, **kwargs):
        return self._build_regularizer(x, **kwargs)


class WeightDecay(BaseRegularizer):
    """Weight decay regularisation.

    Arguments
    ---------
    x : tf.Variable
        The variable to regularise
    amount : float
        Regularisation parameter
    name : str
        The name of the regularisation parameter
    """

    def __init__(self, amount, name="weight_decay", **kwargs):
        self.amount = amount
        self.name = name

    def _build_regularizer(self, x):
        num_params = np.prod(x.get_shape().as_list()[1:])
        amount = np.float32(self.amount / num_params)
        return tf.multiply(amount, tf.reduce_sum(tf.pow(x, 2)), name=self.name)
