from copy import copy
import tensorflow as tf
from . import layers
from . import losses

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
        self._accuracy = None

        # Placeholders
        self._is_training = None
        self._true_out = None

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
    def true_out(self):
        """Placeholder containing the true labels for the input placeholder.
        """
        return self._true_out

    @property
    def loss(self):
        """The loss operator of the network.
        """
        return self._loss

    @property
    def accuracy(self):
        """The accuracy operator of the network.
        """
        return self._accuracy

class NeuralNet(EmptyNet):
    def __init__(self, x, architecture, name=None, is_training=None,
                 true_out=None, loss_function=None, verbose=False):
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
        is_training : tf.placeholder(bool, []) (optional)
            Variable used to specify wether the net is training or not. For 
            example used in batch normalisation and stochastic depth. 
            A placeholder will be generated for this if it is not provided.
        true_out : tf.Variable (optional)
            The wanted output of the network for the given input.
        loss_function : str (optional)
            The name of the loss function used during training. Must be the name
            of a function in the `segmentation_nets.model.losses` file.
        verbose : bool (optional)
            Used to specify if information about the networkshould be printed to
            the terminal window.
        """
        # Check if placeholders are supplied
        self._is_training = is_training if is_training is not None \
                else tf.placeholder(tf.bool, [])
        self._true_out = true_out

        
        # Set pre-assembly properties
        self._input = x
        self._architecture = architecture
        self._verbose = verbose
        self._name = name

        # Assemble network
        with tf.variable_scope(self.name) as self.vscope:
            self.build_model()

        # Set loss function
        if true_out is not None and loss_function is not None:
            self.set_loss(true_out, loss_function)

    def set_loss(self, true_out, loss_function, true_name='labels', 
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
        self._true_out = true_out
        self.init_accuracy()

        loss_func = getattr(losses, loss_function)
        with tf.variable_scope(self.name+'/loss'):
            uregularised_loss = tf.reduce_mean(
                    loss_func(
                        **{predicted_name: self.out,
                           true_name: self.true_out}
                    ),
                    name='loss_function'
            )
            self._loss = tf.add(uregularised_loss, self.reg_op,
                                name='regularised_Loss')

    def init_accuracy(self):
        """Initiate the accuracy operator of the network.
        """
        correct_predictions = tf.equal(
            tf.argmax(self.out, axis=-1), tf.argmax(self.true_out, axis=-1)
        )
        self._accuracy = tf.reduce_mean(
            tf.cast(correct_predictions, tf.float32)
        )

    def collect_regularizers(self):
        """Combine all the regularizer lists into one operator.
        """
        self.reg_list = []
        for regs in self.reg_lists.values():
            self.reg_list += regs

        self._reg_op = 0
        if len(self.reg_list) > 0:
            self._reg_op = tf.add_n(self.reg_list, name='regularizers')

    def assemble_layer(self, layer):
        """Assemble a single layer.
        """
        layer_func = getattr(layers, layer['layer'])
        self._out, params, reg_lists = layer_func(
            self._out,
            is_training=self.is_training,
            verbose=self.verbose,
            **layer
        )

        self.outs[layer['scope']] = self._out
        self.reg_lists[layer['scope']] = reg_lists
        for pname, param in params.items():
            self._params[layer['scope'] + '/' + pname] = param

    def build_model(self):
        """Assemble the network.
        """
        self._out = self.input
        self.outs = {'input': self.input}
        self._params = {}
        self._reg_lists = {}

        if self.verbose:
            print('\n'+25*'-'+'Assembling network'+25*'-')

        for layer in self.architecture:
            self.assemble_layer(layer)

        if self.verbose:
            print(25*'-'+'Finished assembling'+25*'-'+'\n')
            
        self.collect_regularizers()
