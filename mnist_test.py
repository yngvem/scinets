from base_model.train_model import *

if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
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
                'k_size': (5, 1),
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
                'k_size': (5, 1),
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

    network = NeuralNet(x, architecture, verbose=True, name='TestNet')
    network.set_loss(
        true_labels=true_y,
        loss_function='softmax_cross_entropy_with_logits'
    )
    trainer = NetworkTrainer(network, mnist)
    
    correct_prediction = tf.equal(tf.argmax(network.out, 1), tf.argmax(network.true_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        trainer.load_state(sess, 500)
        for i in range(10):
            trainer.train(sess, 100, 100)
            if i % 100 == 0:
                batch = mnist.train.next_batch(100)
                train_accuracy = accuracy.eval(feed_dict={
                    _x: batch[0], true_y: batch[1], network.is_training: False})
                print('step %d, training accuracy %g' % (i, train_accuracy))

        print('test accuracy %g' % accuracy.eval(feed_dict={
          _x: mnist.test.images, true_y: mnist.test.labels, network.is_training: False}))
