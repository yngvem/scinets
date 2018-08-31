__author__ = 'Yngve Mardal Moe'
__email__ = 'yngve.m.moe@gmail.com'


import tensorflow as tf

_ver = [int(v) for v in tf.__version__.split('.')]
keras = tf.keras if _ver[0] >= 1 and _ver[1] >= 4 else tf.contrib.keras


he_normal = keras.initializers.he_normal
he_uniform = keras.initializers.he_uniform
glorot_normal = keras.initializers.glorot_normal
glorot_uniform = keras.initializers.glorot_uniform
normal = tf.initializers.random_normal

