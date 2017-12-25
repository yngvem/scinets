import tensorflow as tf
import numpy as np

def log_variable(var, log_mean=True, log_stddev=True, log_max=True, log_min=True, 
                 log_histogram=True, var_name=None):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    summaries = []
    scope = 'summaries' if var_name is None else 'summaries ' + var_name
    with tf.variable_scope('summaries'):
        if log_mean:
            mean = tf.reduce_mean(var)
            summaries.append(tf.summary.scalar('mean', mean))
        if log_stddev:
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            summaries.append(tf.summary.scalar('stddev', stddev))
        if log_max:
            summaries.append(tf.summary.scalar('max', tf.reduce_max(var)))
        if log_min:
            summaries.append(tf.summary.scalar('min', tf.reduce_min(var)))
        if log_hist:
            summaries.append(tf.summary.histogram('histogram', var))
    return summaries


def log_sparsity(var, axis=None, name=None):
    """Creates a summary with the precentage of zero-valued indices in a tensor.
    """
    shape = var.get_shape().as_list()
    num_total = np.prod(shape[:axis] + shape[axis+1:])
    for i, n in enumerate(shape):
        num_total += n if i != axis else 0
    
    nonzero = tf.count_nonzero(var, axis)
    sparsity = 1-nonzero/num_total
    with tf.variable_scope('summaries' + name if type(name) == str else ''):
        return tf.summary.histogram(sparsity)
    


def log_dict(var_dict, name=None):
    """Logs all the variable in a dictionary, using their keys as name.
    """
    summaries = []
    for var_name, var in var.items():
        summaries += log_variable(var, var_name=var_name)
    
    merged = tf.summaries.merge(summaries, name=name)
    return merged


#TODO: Implement this
def log_PCA_images(images, principal_component=1):
    """Create an image where pixel-values are the PCA of the channel activation.

    Parameters
    ----------
    image : tf.Tensor
        A tensorflow image stack.
            * First axis is image number.
            * Second axis is height.
            * Third axis is width.
            * Fourth axis is channels.
    principal_component : int
        What principal component to display in the image. 1 is most influential,
        2 is the second most influential and so on.

    Returns : tf.Tensor
        The summary protocol buffer (which is sent to a file-writer)
    """
    raise NotImplementedError
