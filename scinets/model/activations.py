from .._backend_utils import DictionaryRegister
import tensorflow as tf


relu = tf.nn.relu
sigmoid = tf.nn.sigmoid
softmax = tf.nn.softmax


def linear(x):
    return x


activation_register = {
    "relu": tf.nn.relu,
    "sigmoid": tf.nn.sigmoid,
    "softmax": tf.nn.softmax,
    "linear": linear
}

activation_register = DictionaryRegister(activation_register)


def get_activation(activation):
    return activation_register.get_item(activation)
