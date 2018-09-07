import tensorflow as tf

softmax_cross_entropy_with_logits = tf.nn.softmax_cross_entropy_with_logits
sigmoid_cross_entropy_with_logits = tf.nn.sigmoid_cross_entropy_with_logits


def binary_dice(prediction, target, scope='binary_dice_loss'):
    with tf.variable_scope(scope):
        dice_numerator = 2*tf.reduce_sum(prediction*target)
        dice_denominator = (tf.reduce_sum(tf.square(target)) + 
                            tf.reduce_sum(tf.square(prediction))
        return dice_numerator/dice_denominator
    
