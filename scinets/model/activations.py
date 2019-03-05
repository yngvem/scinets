from abc import ABC, abstractmethod
from .._backend_utils import SubclassRegister
import tensorflow as tf


activation_register = SubclassRegister("activation function")


def get_activation(activation):
    return activation_register.get_item(activation)


@activation_register.link_base
class BaseActivation(ABC):
    def __call__(self, x):
        return self._build_activation(x)

    @abstractmethod
    def _build_activation(self, x):
        pass


class Linear(BaseActivation):
    def _build_activation(self, x):
        return x


class ReLU(BaseActivation):
    def _build_activation(self, x):
        return tf.nn.relu(x)


class LeakyReLU(BaseActivation):
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def _build_activation(self, x):
        return tf.nn.leaky_relu(x, alpha=self.alpha)


class Sigmoid(BaseActivation):
    def _build_activation(self, x):
        return tf.nn.sigmoid(x)


class Softmax(BaseActivation):
    def _build_activation(self, x):
        return tf.nn.softmax(x)
