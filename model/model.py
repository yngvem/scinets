__author__ = 'Yngve Mardal Moe'
__email__ = 'yngve.m.moe@gmail.com'


from copy import copy
import tensorflow as tf
from . import layers
from . import losses


class NeuralNet:
    def __init__(self, input_var, architecture, name=None, is_training=None,
                 true_out=None, loss_function=None, loss_kwargs=None, 
                 verbose=False):
        """
        Create a standard feed-forward net.

        Parameters
        ----------
        input_var : tf.Variable
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
        loss_kwargs : dict (optional)
            Keyword arguments to supply to the loss function.
        verbose : bool (optional)
            Used to specify if information about the networkshould be printed to
            the terminal window.
        """
        # Check if placeholders are supplied
        self.is_training = is_training if is_training is not None \
                else tf.placeholder(tf.bool, [])
        self.true_out = true_out

        
        # Set pre-assembly properties
        self.input = input_var
        self.architecture = architecture
        self.verbose = verbose
        self.name = name

        # Assemble network
        with tf.variable_scope(self.name) as self.vscope:
            self.build_model()

        # Set loss function
        if true_out is not None and loss_function is not None:
            self.set_loss(true_out, loss_function, loss_kwargs)
        

    def set_loss(self, true_out, loss_function, true_name='labels', 
                 predicted_name='logits', loss_kwargs=None, **kwargs):
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
        loss_kwargs : dict (optional)
            Keyword arguments for the loss function.
        """
        self.true_out = true_out
        loss_kwargs = loss_kwargs if loss_kwargs is not None else {}

        loss_function = getattr(losses, loss_function)
        with tf.variable_scope(self.name+'/loss'):
            uregularised_loss = tf.reduce_mean(
                    loss_function(
                        **{predicted_name: self.out,
                           true_name: self.true_out},
                        **loss_kwargs
                    ),
                    name='loss_function'
            )

            self.loss = tf.add(uregularised_loss, self.reg_op,
                                name='regularised_Loss')

    def collect_regularizers(self):
        """Combine all the regularizer lists into one operator.
        """
        self.reg_list = []
        for regs in self.reg_lists.values():
            self.reg_list += regs

        self.reg_op = 0
        if len(self.reg_list) > 0:
            self.reg_op = tf.add_n(self.reg_list, name='regularizers')

    def assemble_layer(self, layer):
        """Assemble a single layer.
        """
        layer_func = getattr(layers, layer['layer'])
        self.out, params, reg_lists = layer_func(
            self.out,
            is_training=self.is_training,
            verbose=self.verbose,
            **layer
        )

        self.outs[layer['scope']] = self.out
        self.reg_lists[layer['scope']] = reg_lists
        for pname, param in params.items():
            self.params[layer['scope'] + '/' + pname] = param

    def build_model(self):
        """Assemble the network.
        """
        self.out = self.input
        self.outs = {'input': self.input}
        self.params = {}
        self.reg_lists = {}

        if self.verbose:
            print('\n'+25*'-'+'Assembling network'+25*'-')

        for layer in self.architecture:
            self.assemble_layer(layer)

        if self.verbose:
            print(25*'-'+'Finished assembling'+25*'-'+'\n')
            
        self.collect_regularizers()
