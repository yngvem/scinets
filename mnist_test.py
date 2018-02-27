import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model import NeuralNet
from utils import TensorboardLogger
from trainer import NetworkTrainer
import sys

if __name__ == '__main__':
    class Dataset:
        def __init__(self):
            self.epoch_size = 10000
            self.dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
        
        @property
        def train(self):
            return self.dataset.train
        
        @property
        def test(self):
            return self.dataset.test
    
    mnist = Dataset()

    _x = tf.placeholder(tf.float32, shape=[None, 784])
    x = tf.reshape(_x, [-1, 28, 28, 1])

    true_y = tf.placeholder(tf.float32, shape=[None, 10])

    architecture = [
            {
                'layer': 'conv2d',
                'scope': 'conv1',
                'out_size': 8,
                'k_size': 5,
                'batch_norm': True,
                'activation': 'relu',
                'regularizer': {
                    'function': 'weight_decay',
                    'arguments': {
                        'amount': 0.5,
                        'name': 'weight_decay'
                    }
                }
            },
            {
                'layer': 'conv2d',
                'scope': 'conv2',
                'out_size': 16,
                'k_size': 5,
                'strides': 2,
                'batch_norm': True,
                'activation': 'relu',
            },
            {
                'layer': 'conv2d',
                'scope': 'conv3',
                'out_size': 16,
                'k_size': 5,
                'strides': 2,
                'batch_norm': True,
                'activation': 'relu',
            },
            {
                'layer': 'fc_layer',
                'scope': 'fc',
                'out_size': 10,
                'batch_norm': True,
                'activation': 'linear',
            }
    ]

    name = sys.argv[1] if len(sys.argv) > 1 else 'TestNet'
    network = NeuralNet(x, architecture, verbose=True, name=name)
    network.set_loss(
        true_labels=true_y,
        loss_function='softmax_cross_entropy_with_logits'
    )
    trainer = NetworkTrainer(network, mnist)
    log_dict = {
        'loss': [
            {
                'log_name': 'Loss',
                'log_type': 'scalar'
            },
            {
                'log_name': 'Log loss',
                'log_type': 'log_scalar'
            }
        ],
        'input': [{
            'log_name': 'Images',
            'log_type': 'image'
        }]
    }

    logger = TensorboardLogger(network, log_dict)
    
    correct_prediction = tf.equal(tf.argmax(network.out, 1), tf.argmax(network.true_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        logger.init_file_writers(sess)
        
        batch = mnist.test.next_batch(100)
        val_accuracy, = sess.run([network.accuracy], feed_dict={
            _x: batch[0], true_y: batch[1], network.is_training: False})
        print('step %d, val accuracy %g' % (0, val_accuracy))


        for i in range(100):
            summaries, steps = trainer.train(sess, 10, additional_ops=[logger.train_summary_op])
            summaries = [summary[0] for summary in summaries]
            logger.log_multiple(summaries, steps)

            batch = mnist.test.next_batch(100)
            val_accuracy, = sess.run([network.accuracy], feed_dict={
                _x: batch[0], true_y: batch[1], network.is_training: False})
            print('step %d, val accuracy %g' % (0, val_accuracy))
            

        print('test accuracy %g' % accuracy.eval(feed_dict={
          _x: mnist.test.images, true_y: mnist.test.labels, network.is_training: False}))
