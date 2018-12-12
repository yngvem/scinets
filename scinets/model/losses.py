import tensorflow as tf


def softmax_cross_entropy_with_logits(prediction, target, name=None):
    return tf.nn.softmax_cross_entropy_with_logits(
        labels=target, logits=prediction, name=name
    )


def sigmoid_cross_entropy_with_logits(prediction, target, name=None):
    return tf.nn.sigmoid_cross_entropy_with_logits(
        labels=target, logits=prediction, name=name
    )


def binary_dice(prediction, target, name="binary_dice_loss"):
    size = len(prediction.get_shape().as_list())
    reduce_ax = list(range(1, size))
    eps = 1e-8

    with tf.variable_scope(name):
        dice_numerator = 2 * tf.reduce_sum(prediction * target, axis=reduce_ax) + eps
        dice_denominator = (
            tf.reduce_sum(tf.square(target), axis=reduce_ax)
            + tf.reduce_sum(tf.square(prediction), axis=reduce_ax)
            + eps
        )
        return 1 - dice_numerator / dice_denominator


def binary_f_beta(prediction, target, beta=1, name="binary_dice_loss"):
    size = len(prediction.get_shape().as_list())
    reduce_ax = list(range(1, size))
    eps = 1e-8

    with tf.variable_scope(name):
        true_positive = tf.reduce_sum(prediction * target, axis=reduce_ax)
        target_positive = tf.reduce_sum(tf.square(target), axis=reduce_ax)
        predicted_positive = tf.reduce_sum(tf.square(prediction), axis=reduce_ax)
        dice_numerator = (1 + beta ** 2) * true_positive + eps
        dice_denominator = (beta ** 2) * target_positive + predicted_positive + eps

        return 1 - dice_numerator / dice_denominator
