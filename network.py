# TODO: Check regularization!

import tensorflow as tf
import layers
import losses
import optimizers

class EmptyNet:
    """Container for all properties of NeuralNet classes
    """
    def __init__(self):
        self._name = None
        self._input = None
        self._architecture = None
        self._verbose = None
        self._params = None
        self._layer_outs = None
        self._reg_lists = None
        self._reg_op = None
        self._out = None
        self._loss = None
        self._optimizer = None
        self._train_step = None

        # Placeholders
        self._is_training = None
        self._true_labels = None

    @property
    def name(self):
        """The name of this network.
        """
        return self._name

    @property
    def input(self):
        """The input tensor of the network.
        """
        return self._input

    @property
    def architecture(self):
        """List of dictionaries paramterising the network.
        """
        return self._architecture

    @property
    def verbose(self):
        """Bool showing if extra information should be printed during assembly.
        """
        return self._verbose

    @property
    def is_training(self):
        """Placeholder (bool) used for batch normalization.
        """
        return self._is_training

    @property
    def layer_outs(self):
        """List of tensors containing the output of each layer in the network.
        """
        return self._layer_outs

    @property
    def params(self):
        """Dictionary containing the weights used in the network
        """
        return self._params

    @property
    def reg_lists(self):
        """List with all regularisation penalties.
        """
        return self._reg_lists

    @property
    def reg_op(self):
        """Sum of all regularisation penalty tensors.
        """
        return self._reg_op

    @property
    def out(self):
        """Tensor containing the output of the network.
        """
        return self._out

    @property
    def true_labels(self):
        """Placeholder containing the true labels for the input placeholder.
        """
        return self._true_labels

    @property
    def loss(self):
        """The total loss of the network.
        """
        return self._loss

    @property
    def optimizer(self):
        """The optimizer operator used for the network.
        """
        return self._optimizer

    @property
    def train_step(self):
        """The training step operator called each iteration of training.
        """
        return self._train_step


class NeuralNet(EmptyNet):
    def __init__(self, x, architecture, name=None, is_training=None,
                 true_labels=None, loss_function=None, verbose=False):
        """
        Create a standard feed-forward net.

        Parameters
        ----------
        x : tf.Variable
            The input variable to the network
        architecture : array_like
            An array of dictionaries. The first dictionary specifies the 
            parameters of the first layer, the second dictionary specifies the 
            parameters of the second layer and so on.
        is_training : tf.placeholder(bool, [])
            Variable used to specify wether the net is training or not. For 
            example used in batch normalisation and stochastic depth.
        verbose : bool
            Used to specify if information about the networkshould be printed to
            the terminal window.
        """
        # Check if placeholders are supplied
        if is_training is not None:
            self._is_training = is_training
        else: 
            self._is_training = tf.placeholder(tf.bool, [])
        if true_labels is not None:
            self._true_labels = true_labels
        else:
            self._true_labels = None

        # Set pre-assembly properties
        self._input = x
        self._architecture = architecture
        self._verbose = verbose
        self._name = name

        # Assemble network
        l, p, rl, ro = self.assemble_network()
        self._layer_outs, self._params = l, p
        self._reg_lists, self._reg_op = rl, ro
        self._out = self.layer_outs[-1]

        # Set loss function
        if true_labels is not None and loss_function is not None:
            self.set_loss(true_labels, loss_function)

    def set_loss(self, true_labels, loss_function, true_name='labels', 
                 predicted_name='logits', **kwargs):
        """
        Set the loss function of the network.

        Parameters
        ----------
        true_output : tf.Tensor
            Tensor containing the true labels
        loss_function : str
            Name of loss function, must be name of a function in the `losses.py`
            file.
        true_name : str
            Keyword for the true labels for the loss function
        predicted_name : str
            Keyword for the predicted labels for the loss function
        """
        self._true_labels = true_labels
        loss_func = getattr(losses, loss_function)
        self._loss = tf.reduce_mean(loss_func(
            **{
                predicted_name: self.out,
                true_name: self.true_labels
            }
        )) + self.reg_op

    def set_train_op(self, train_op, **kwargs):
        """Set the operator used for weight updates.

        Parameters
        ----------
        train_op : str
            The optimizer to use for weight updates. Must be the name of an 
            element of the `optimizers.py` file.
        
        """
        if self.loss is None:
            raise RuntimeError(
                'The `set_loss` function must be ran before `set_train`.'
            )

        Optimizer = getattr(optimizers, train_op)
        self._optimizer = Optimizer(**kwargs)

        UPDATE_OPS = tf.GraphKeys.UPDATE_OPS
        with tf.control_dependencies(tf.get_collection(UPDATE_OPS)):
            self._train_step = optimizer.minimize(loss)

    def assemble_network(self):
        """Assemble the network.

        Returns
        -------
        outs : list
            List of tensors with the output of each layer in the network.
        params : dict
            Dictionary with all the trainables of the network.
        reg_list : list
            List with the all the regularisation weights.
        reg_op : tf.Tensor
            The sum of all the regularisation weights.
        """
        out = self.input
        outs = [self.input]
        params = {}
        reg_lists = {}
        with tf.variable_scope(self.name):
            for layer in self.architecture:
                layer_func = getattr(layers, layer['layer'])
                out, _params, _reg_lists = layer_func(
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

        reg_op = tf.add_n(reg_list) if len(reg_list) > 0 else 0
        return outs, params, reg_lists, reg_op



if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

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
    network.set_train_op('AdamOptimizer')
    loss = network.loss
    y = network.out
    is_training = network.is_training

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(true_y, 1))
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
                    _x: batch[0], true_y: batch[1], is_training: False})
                print('step %d, training accuracy %g' % (i, train_accuracy))

            train_step.run(feed_dict={_x: batch[0], true_y: batch[1], is_training: True})

        print('test accuracy %g' % accuracy.eval(feed_dict={
          _x: mnist.test.images, true_y: mnist.test.labels, is_training: False}))
