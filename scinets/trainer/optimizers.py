import tensorflow as tf
from .._backend_utils import DictionaryRegister


optimizer_register = {
    "AdamOptimizer": tf.train.AdamOptimizer,
    "AdadeltaOptimizer": tf.train.AdadeltaOptimizer,
    "GradientDescentOptimizer": tf.train.GradientDescentOptimizer,
    "MomentumOptimizer": tf.train.MomentumOptimizer,
    "RMSPropOptimizer": tf.train.RMSPropOptimizer,
}


optimizer_register = DictionaryRegister(optimizer_register)


def get_optimizer(item):
    return optimizer_register.get_item(item)
