import tensorflow as tf
import layers


class NeuralNet:
    def __init__(self, x, architecture, is_training=None, verbose=False):
        self.input = x
        self.architecture = architecture
        self.verbose = verbose
        self.is_training = tf.placeholder(tf.bool, []) if is_training is None else is_training

        self.layer_outs, self.params = self.assemble_network()
        self.out = self.layer_outs[-1]

    def assemble_network(self):
        out = self.input
        outs = [self.input]
        params = {}
        for layer in self.architecture:
            out, _params = getattr(layers, layer['layer'])(
                out,
                is_training=self.is_training,
                verbose=self.verbose,
                **layer
            )
            outs.append(out)

            for pname, param in _params.items():
                params[layer['scope'] + '/' + pname] = param

        return outs, params


if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    _x = tf.placeholder(tf.float32, shape=[None, 784])
    x = tf.reshape(_x, [-1, 28, 28, 1])

    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    architecture = [
            {
                'layer': 'conv2d',
                'scope': 'conv1',
                'out_size': 8,
                'k_size': 5,
                'batch_norm': True,
                'activation': 'relu',
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

    network = NeuralNet(x, architecture, verbose=True)
    y = network.out
    is_training = network.is_training

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    optimizer = tf.train.AdamOptimizer()
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_step = optimizer.minimize(loss)

    with tf.Session() as sess:
        batch = mnist.train.next_batch(100)
        sess.run(tf.global_variables_initializer())
        for i in range(2000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
              train_accuracy = accuracy.eval(feed_dict={
                  _x: batch[0], y_: batch[1], is_training: False})
              print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={_x: batch[0], y_: batch[1], is_training: True})

        print('test accuracy %g' % accuracy.eval(feed_dict={
          _x: mnist.test.images, y_: mnist.test.labels, is_training: False}))
