"""
TODO: FIX loss kwargs
"""


__author__ = "Yngve Mardal Moe"
__email__ = "yngve.m.moe@gmail.com"


from copy import copy
import tensorflow as tf
import math as m
from . import layers
from . import losses


class BaseModel:
    def __init__(
        self,
        input_var,
        architecture,
        name=None,
        is_training=None,
        true_out=None,
        loss_function=None,
        loss_kwargs=None,
        verbose=False,
    ):
        """
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
        device : str (optional)
            Used to force which device the network should be placed on.
        verbose : bool (optional)
            Used to specify if information about the networkshould be printed to
            the terminal window.
        """
        # Check if placeholders are supplied
        self.is_training = (
            is_training if is_training is not None else tf.placeholder(tf.bool, [])
        )
        self.true_out = true_out

        # Set pre-assembly properties
        self.input = input_var
        self.architecture = architecture
        self.verbose = verbose
        self.name = name

        # Assemble network
        self.out = self.input
        self.outs = {"input": self.input}
        self.layers = []
        self.params = {}
        self.reg_lists = {}

        with tf.variable_scope(self.name) as self.vscope:
            self.build_model()

        # Set loss function
        if true_out is not None and loss_function is not None:
            self.set_loss(true_out, loss_function, loss_kwargs=loss_kwargs)

    def set_loss(self, true_out, loss_function, loss_kwargs=None, **kwargs):
        """
        Set the loss function of the network.

        Parameters
        ----------
        true_output : tf.Tensor
            Tensor containing the true labels
        loss_function : str
            Name of loss function, must be name of a function in the `losses.py`
            file.
        loss_kwargs : dict (optional)
            Keyword arguments for the loss function.
        """
        self.true_out = true_out
        loss_kwargs = loss_kwargs if loss_kwargs is not None else {}

        loss_function = getattr(losses, loss_function)
        with tf.variable_scope(self.name + "/loss"):
            uregularised_loss = tf.reduce_mean(
                loss_function(prediction=self.out, target=self.true_out, **loss_kwargs),
                name="loss_function",
            )

            self.loss = tf.add(uregularised_loss, self.reg_op, name="regularised_Loss")

    def collect_regularizers(self):
        """Combine all the regularizer lists into one operator.
        """
        self.reg_list = []
        for regs in self.reg_lists.values():
            self.reg_list += regs

        self.reg_op = 0
        if len(self.reg_list) > 0:
            self.reg_op = tf.add_n(self.reg_list, name="regularizers")

    def buil_model(self):
        pass

    def assemble_layer(self, layer_dict):
        """Assemble a single layer.
        """
        layer_class = layers.get_layer(layer_dict["layer"])
        layer = layer_class(
            self.out, is_training=self.is_training, verbose=self.verbose, **layer_dict
        )

        self.layers.append(layer)
        self.out = layer.output

        self.outs[layer_dict["scope"]] = self.out
        self.reg_lists[layer_dict["scope"]] = layer.reg_list
        for pname, param in layer.params.items():
            self.params[layer_dict["scope"] + "/" + pname] = param


class NeuralNet(BaseModel):
    def build_model(self):
        """Assemble the network.
        """
        if self.verbose:
            print("\n" + 25 * "-" + "Assembling network" + 25 * "-")

        for layer in self.architecture:
            self.assemble_layer(layer)

        if self.verbose:
            print(25 * "-" + "Finished assembling" + 25 * "-" + "\n")

        self.collect_regularizers()


class UNet(BaseModel):
    def __init__(
        self,
        input_var,
        architecture,
        skip_connections,
        name=None,
        is_training=None,
        true_out=None,
        loss_function=None,
        loss_kwargs=None,
        verbose=False,
    ):
        self.skip_connections = skip_connections
        super().__init__(
            input_var=input_var,
            architecture=architecture,
            name=name,
            is_training=is_training,
            true_out=true_out,
            loss_function=loss_function,
            loss_kwargs=loss_kwargs,
            verbose=verbose,
        )

    def is_skip_connection_target(self, layer):
        """Returns whether the current layer is the target of a skip connection.
        """
        for skip_connection in self.skip_connections:
            if skip_connection[1] == layer["scope"]:
                return True

    def get_skip_connection(self, layer):
        """Returns the skip connection where the given layer is the target.
        """
        for i, skip_connection in enumerate(self.skip_connections):
            if skip_connection[1] == layer["scope"]:
                return i, skip_connection

    def create_skip_connection(self, curr_layer):
        """Return a the skip connection that ends in the current layer.
        """
        skip_id, skip_connection = self.get_skip_connection(curr_layer)
        with tf.variable_scope("skip_connection_{}".format(skip_id)) as vscope:
            self.out = self._safe_concat(
                self.outs[skip_connection[0]], self.outs[skip_connection[1]]
            )
            self.outs[vscope.name] = self.out
            self.reg_lists[skip_id] = []

    @staticmethod
    def _safe_concat(first_tensor, second_tensor):
        size = first_tensor.get_shape().as_list()[1:-1]
        second_tensor = tf.image.resize_images(
            second_tensor,
            size=size,
            method=tf.image.ResizeMethod.BILINEAR,
            align_corners=True,
        )
        return tf.concat((first_tensor, second_tensor), axis=-1)

    def build_model(self):
        """Assemble the network.
        """
        if self.verbose:
            print("\n" + 25 * "-" + "Assembling network" + 25 * "-")

        for i, layer in enumerate(self.architecture):
            self.assemble_layer(layer)
            if self.is_skip_connection_target(layer):
                self.create_skip_connection(layer)

        if self.verbose:
            print(25 * "-" + "Finished assembling" + 25 * "-" + "\n")

        self.collect_regularizers()
