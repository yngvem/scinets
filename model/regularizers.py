import tensorflow as tf


def weight_decay(x, amount=1, name='weight_decay', **kwargs):
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
    return amount*tf.reduce_sum(tf.pow(x, 2, name=name))/tf.reduce_sum(x)
