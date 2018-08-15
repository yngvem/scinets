__author__ = 'Yngve Mardal Moe'
__email__ = 'yngve.m.moe@gmail.com'


import tensorflow as tf

_ver = [int(v) for v in tf.__version__.split('.')]
keras = tf.keras if _ver[0] >= 1 and _ver[1] >= 4 else tf.contrib.keras


def _check_dtype(initializer):
    def typesafe_initializer(seed, dtype=tf.float32):
        if dtype != tf.float32:
            raise ValueError(
                '{} is not supported for other types than `tf.float32`'.format(
                    initializer.__name__
                )

            )
        return initializer(seed)
    typesafe_initializer.__name__ = "typesafe_{}".format(initializer.__name__)
    
    return typesafe_initializer


he_normal = _check_dtype(keras.initializers.he_normal)
he_uniform = _check_dtype(keras.initializers.he_uniform)
glorot_normal = keras.initializers.glorot_normal
glorot_uniform = keras.initializers.glorot_uniform
normal = tf.initializers.random_normal

