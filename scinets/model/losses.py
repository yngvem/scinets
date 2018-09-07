import tensorflow as tf

softmax_cross_entropy_with_logits = tf.nn.softmax_cross_entropy_with_logits
sigmoid_cross_entropy_with_logits = tf.nn.sigmoid_cross_entropy_with_logits

def softmax_cross_entropy_with_logits(prediction, target, name):
    return softmax_cross_entropy_with_logits(labels=target, logits=prediction,
                                             name=name)

def sigmoid_cross_entropy_with_logits(prediction, target, name):
    return sigmoid_cross_entropy_with_logits(labels=target, logits=prediction,
                                             name=name)

def binary_dice(prediction, target, name='binary_dice_loss'):
    size = len(prediction.get_shape().as_list())
    reduce_ax = list(range(1, size))
    eps = 1e-4

    with tf.variable_scope(name):
        dice_numerator = 2*tf.reduce_sum(prediction*target, axis=reduce_ax) + eps
        dice_denominator = (tf.reduce_sum(tf.square(target), axis=reduce_ax) + 
                            tf.reduce_sum(tf.square(prediction), axis=reduce_ax)+
                            eps)
        return 1-dice_numerator/dice_denominator
    
