__author__ = "Yngve Mardal Moe"
__email__ = "yngve.m.moe@gmail.com"


import tensorflow as tf
from .._backend_utils import DictionaryRegister

_ver = [int(v) for v in tf.__version__.split(".")]
keras = tf.keras if _ver[0] >= 1 and _ver[1] >= 4 else tf.contrib.keras


initializer_register = {
    "he_normal": keras.initializers.he_normal,
    "he_uniform": keras.initializers.he_uniform,
    "glorot_normal": keras.initializers.glorot_normal,
    "glorot_uniform": keras.initializers.glorot_uniform,
    "normal": tf.initializers.random_normal,
}


initializer_register = DictionaryRegister(initializer_register)


def get_initializer(item):
    return initializer_register.get_item(item)
