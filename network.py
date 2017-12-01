import tensorflow as tf
import layers
import losses
import optimizers

class EmptyNet:
    """Container for all properties of NeuralNet classes
    """
    @property
    def input(self):
        return self._input

    @property
    def architecture(self):
        return self._architecture

    @property
    def verbose(self):
        return self._verbose

    @property
    def is_training(self):
        return self._is_training

    @property
    def layer_outs(self):
        return self._layer_outs

    @property
    def params(self):
        return self._params

    @property
    def reg_lists(self):
        return self._reg_lists

    @property
    def reg_op(self):
        return self._reg_op

    @property
    def out(self):
        return self_out

    @property
    def true_out(self):
        return self._true_out

    @property
    def loss(self):
        return self._loss

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def train_step(self):
        return self._train_step


class NeuralNet(EmptyNet):
    def __init__(self, x, architecture, name=None, is_training=None, verbose=False):
        """
        Create a standard feed-forward net.

        Parameters
        ----------
        x : tf.Variable
            The input variable to the network
        architecture : array_like
            An array of dictionaries. The first dictionary specifies the parameters of the
            first layer, the second dictionary specifies the parameters of the second layer
            and so on.
        is_training : tf.placeholder(bool, [])
            Variable used to specify wether the net is training or not. For example used in
            batch normalisation and stochastic depth.
        verbose : bool
            Used to specify if information about the networkshould be printed to the
            terminal window.
        """
        self._input = x
        self._architecture = architecture
        self._verbose = verbose
        self._is_training = tf.placeholder(tf.bool, []) if is_training is None else is_training

        self._layer_outs, self._params, self._reg_lists, self._reg_op = self.assemble_network()
        self._out = self.layer_outs[-1]

    def set_loss(self, true_output, loss_function, true_name='labels', predicted_name='logits',
                 **kwargs):
        self._true_out = true_out
        loss_func = getattr(losses, loss_function)
        self._loss = tf.reduce_mean(loss_func(
            self.out,
            self.true_out
        ) + self.reg_op

    def set_train_op(self, train_op, **kwargs):
        if self.loss_function is None:
            raise RuntimeError(
            'The layer\'s `set_loss` function must be runned before setting training operator.')

        Optimizer = getattr(optimizers, train_op)
        self._optimizer = Optimizer(**kwargs)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self._train_step = optimizer.minimize(loss)

    def assemble_network(self):
        out = self.input
        outs = [self.input]
        params = {}
        reg_lists = {}
        with tf.variable_scope(self.name):
            for layer in self.architecture:
                if 'regularizer' in architecture:
                    raise RuntimeWarning('Regularization not implemented yet.')

                out, _params, _reg_lists = getattr(layers, layer['layer'])(
                    out,
                    is_training=self.is_training,
                    verbose=self.verbose,
                    **layer
                )
                outs.append(out)

                for pname, param in _params.items():
                    params[layer['scope'] + '/' + pname] = param
                reg_lists[layer['scope']] = _reg_lists

        reg_list = []
        for regs in reg_lists.values():
            reg_list += regs

        reg_op = tf.add_n(reg_list)
        return outs, params, reg_lists, reg_op


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
