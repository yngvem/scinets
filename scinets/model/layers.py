__author__ = "Yngve Mardal Moe"
__email__ = "yngve.m.moe@gmail.com"


import tensorflow as tf
import numpy as np
from . import activations
from . import regularizers
from . import normalizers
from . import initializers


_ver = [int(v) for v in tf.__version__.split(".")]
keras = tf.keras if _ver[0] >= 1 and _ver[1] >= 4 else tf.contrib.keras


class BaseLayer:
    """Base class for all layers.

    To create new layers, one only needs to overload the `_build_layer`
    function, which should return the output tensor of the layer. Additionally,
    for verbose network building to be supported, the `_print_info` function
    must be overloaded.
    """

    def __init__(
        self,
        x,
        initializer=None,
        regularizer=None,
        activation=None,
        normalizer=None,
        is_training=None,
        scope=None,
        layer_params=None,
        verbose=False,
        *args,
        **kwargs
    ):
        if normalizer is not None and is_training is None:
            raise ValueError(
                "You have to supply the `is_training` placeholder for batch norm."
            )
        layer_params = layer_params if layer_params is not None else {}

        self.input = x
        self.is_training = is_training
        self.scope = self._get_scope(scope)

        self.initializer, self._init_str = self._generate_initializer(initializer)
        self.activation, self._act_str = self._generate_activation(activation)
        self.regularizer, self._reg_str = self._generate_regularizer(regularizer)
        self.normalizer, self._normalizer_str = self._generate_normalizer(normalizer)

        # Build layer
        with tf.variable_scope(scope) as self.vscope:
            self.output = self._build_layer(**layer_params)
            self.params, self.reg_list = self._get_returns(self.vscope)

            if verbose:
                self._print_info(layer_params)

    def _get_scope(self, scope):
        if scope is None:
            scope = type(self).__name__
        return scope

    @staticmethod
    def _get_operator(module, operator_dict):
        """Use a dictionary specification and module to extract a tf operator.

        Parameters
        ----------
        module : module
            The module (name scope) to extract the operator from.
        operator_dict : dict
            Dictionary specifying the name of the operator and its arguments.

        Returns
        -------
        str :
            The name of the operator
        function :
            Function of one variable that applies the operator to its input.
        """
        operator_name = operator_dict["operator"]
        operator_args = operator_dict.get("arguments", {})
        operator = getattr(module, operator_name)

        def operator_func(*args, **kwargs):
            return operator(*args, **kwargs, **operator_args)

        return operator_func, operator_name

    def _generate_initializer(self, initializer):
        """Generates a initializer from dictionary.

        The format of the dictionary is as follows
        ```
            {
                'operator': <initializer_name>,
                'arguments': {
                                 'arg1': <value_1>,
                                 'arg2': <value_2>
                             }
            }
        ```
        thus, if you want to use gaussian initialization with a standard
        deviation of 0.1, the dicitonary should look like this
        ```
            {
                'operator': 'gaussian',
                'arguments': {
                                 'stddev': 0.1
                             }
            }
        ```
        for a full description of all possible initializers and their
        parameters, see the `initializers.py` file.

        Parameters
        ----------
        initializers : dict
            What normalizer to use, see examples above.

        Returns
        -------
        normalizer_func : function
        normalizer_str : str
            A string describing the initialiser returned 
        """
        """Generates an initializer instance from a dictionary

        Parameters
        ----------
        initializer : dict

        Returns
        -------
        init_instance : tf.keras.initializer.Initializer
        init_str : str
            A string describing the initialiser returned 
        """
        if str(initializer) == "None":
            initializer = {"operator": "he_normal"}

        init_class, init_str = self._get_operator(initializers, initializer)
        return init_class(None), init_str

    def _generate_activation(self, activation):
        """Generates an activation function from string.

        Parameters
        ----------
        activation : str
            What initializer to use. Acceptable values are the name of 
            callables in the `activations.py` file, or None.

        Returns
        -------
        act_func : function
        act_str : str
            A string describing the initialiser returned 
        """
        if str(activation) == "None":
            activation = {"operator": "linear"}
        return self._get_operator(activations, activation)

    def _generate_regularizer(self, regularizer):
        """Generates a regularizer from dictionary.

        The format of the dictionary is as follows
        ```
            {
                'operator': <operator_name>,
                'arguments': {
                                 'arg1': <value_1>,
                                 'arg2': <value_2>
                             }
            }
        ```
        thus, if you want to use weight decay regularization with a 
        regularization parameter of 0.2, the correct dictionary would look
        like this
        ```
            {
                'operator': 'weight_decay',
                'arguments': {
                                 'reg_param': 0.2,
                             }
            }
        ```
        for a full description of all possible regularizers and their
        parameters, see the `regularizers.py` file.

        Parameters
        ----------
        regularizer : dict
            What regularizer to use, see examples above.

        Returns
        -------
        reg_func : function
        reg_str : str
            A string describing the initialiser returned 
        """
        if str(regularizer) == "None" or regularizer == {}:
            return None, "No regularization."
        else:
            return self._get_operator(regularizers, regularizer)

    def _generate_normalizer(self, normalizer):
        """Generates a normalizer from dictionary.

        The format of the dictionary is as follows
        ```
            {
                'operator': <operator_name>,
                'arguments': {
                                 'arg1': <value_1>,
                                 'arg2': <value_2>
                             }
            }
        ```
        thus, if you want to use batch normalization, the dicitonary
        ```
            {
                'operator': 'batch_normalization'
            }
        ```
        for a full description of all possible normalizers and their
        parameters, see the `normalizers.py` file.

        Parameters
        ----------
        normalizer : dict
            What normalizer to use, see examples above.

        Returns
        -------
        normalizer_func : function
        normalizer_str : str
            A string describing the initialiser returned 
        """
        if str(normalizer) == "None" or normalizer == {}:

            def normalizer(x, *args, **kwargs):
                return x

            return normalizer, "No normalization"
        else:
            if str(self.is_training) == "None":
                raise RuntimeError(
                    "The `is_training` placeholder must be set if normalization "
                    "is used."
                )
            return self._get_operator(normalizers, normalizer)

    def _get_returns(self, scope):
        """Get the parameters to return from a layer
        """
        trainable = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name
        )
        # TODO: FIX THIS SHIT!
        params = {var.name.split(scope.name)[-1][1:]: var for var in trainable}
        reg_list = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope.name
        )

        return params, reg_list

    def _build_layer(activation, initalizer, regularizer):
        raise RuntimeError("This should be implemented")

    def print_layer_info(self, layer_params):
        self._print_info(layer_params)
        self._print_parameter_shapes()

    def _print_info(self, layer_params):
        raise RuntimeError("This should be implemented!")

    def _print_parameter_shapes(self):
        print(" Parameter shapes:")
        for pname, param in self.params.items():
            print("  {}: {}".format(pname, param.get_shape().as_list()))


class FCLayer(BaseLayer):
    def _flatten(self, x):
        """Flattens `x` if `x` is not a batch of 1 dimensional vectors.

        Parameters
        ----------
        x : tf.Variable

        Returns
        -------
        out : tf.Variable
            A batch of 1 dimensional vectors.
        shape : tuple
            The new shape of this tensor.
        flattened : bool
            Whether `x` was flattened.
        """
        shape = x.get_shape().as_list()
        if len(shape) > 2:
            print(shape)
            shape[0] = -1 if shape[0] is None else shape[0]
            shape = (shape[0], np.prod(shape[1:]))
            out = tf.reshape(x, shape, name="Flatten")
            flattened = True
        else:
            out = x
            flattened = False

        return out, shape, flattened

    def _build_layer(self, out_size, use_bias=True):
        out, shape, self.flattened = _flatten(self.input)
        out = tf.layers.dense(
            out,
            out_size,
            use_bias=use_bias,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
        )
        out = self.activation(out)
        out = self.normalizer(out, training=self.is_training, name="BN")

        # Get variables
        return out

    def _print_info(self, layer_params):
        print(
            "\n________________Fully connected layer________________\n"
            "Variable_scope: {}\n".format(self.vscope.name),
            "Flattened input: {}\n".format(self.flattened),
            "Kernel initialisation: {}\n".format(self._init_str),
            "Activation function: {}\n".format(self._act_str),
            "Kernel regularisation: {}\n".format(self._reg_str),
            "Number of regularizer loss: {}".format(len(self.reg_list)),
            "Use bias: {}\n".format(layer_params["use_bias"]),
            "Normalization: {}\n".format(self._normalizer_str),
            "Input shape: {}\n".format(self.input.get_shape().as_list()),
            "Output shape: {}".format(self.output.get_shape().as_list()),
        )


class Conv2D(BaseLayer):
    """A standard convolutional layer.
    """

    def _build_layer(
        self,
        out_size,
        k_size=3,
        use_bias=True,
        dilation_rate=1,
        strides=1,
        padding="SAME",
    ):
        """
        Creates a convolution layer with `out_size` different filters.

        Parameters
        ----------
        x : tensorflow.Variable
            The input tensor to this layer
        out_size : int
            The shape of the vector out of this layer
        use_bias : bool
            Wether or not this layer should have a bias term
        dilation_rate : int or array_like(length=2)
            The dilation rate of this layer (for atrous convolutions). Asymmetric
            dilations are accepted as a length two array, where the first number
            is the vertical dilation rate and the second number is the horizontal
            strides. Setting a dilation rate and stride different from 1 at 
            the same time is not supported.
        strides : int or array_like(length=2)
            The strides used for this layer. Asymmetric strides are accepted as a 
            length two array, where the first number is the vertical strides and
            the second number is the horizontal strides.
        padding : str
            How to deal with boundary conditions in the convolutions. Accepted 
            values are 'SAME' and 'VALID'. 'SAME' uses the value of the nearest 
            pixel and 'VALID' crops the image so that boundary conditions aren't
            a problem.

        Returns:
        --------
        out : tensorflow.Variable
            Output tensor of this layer
        """
        out = tf.layers.conv2d(
            self.input,
            out_size,
            kernel_size=k_size,
            use_bias=use_bias,
            kernel_initializer=self.initializer,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
            kernel_regularizer=self.regularizer,
        )
        out = self.activation(out)
        out = self.normalizer(out, training=self.is_training, name="BN")

        return out

    def _print_info(self, layer_params):
        print(
            "________________Convolutional layer________________\n",
            "Variable_scope: {}\n".format(self.vscope.name),
            "Kernel size: {}\n".format(layer_params.get("k_size", 3)),
            "Output filters: {}\n".format(layer_params["out_size"]),
            "Strides: {}\n".format(layer_params.get("strides", 1)),
            "Dilation rate: {}\n".format(layer_params.get("dilation_rate", 1)),
            "Padding: {}\n".format(layer_params.get("padding", "SAME")),
            "Kernel initialisation: {}\n".format(self._init_str),
            "Activation function: {}\n".format(self._act_str),
            "Kernel regularisation: {}\n".format(self._reg_str),
            "Number of regularizer loss: {}".format(len(self.reg_list)),
            "Use bias: {}\n".format(layer_params.get("use_bias", True)),
            "Normalization: {}\n".format(self._normalizer_str),
            "Input shape: {}\n".format(self.input.get_shape().as_list()),
            "Output shape: {}".format(self.output.get_shape().as_list()),
        )


class Upconv2D(BaseLayer):
    """A standard upconv layer (sometimes called deconv or transposed conv).
    """

    def _build_layer(
        self, out_size, k_size=3, use_bias=True, strides=1, padding="SAME"
    ):
        """
        Creates a convolution layer with `out_size` different filters.

        Parameters
        ----------
        x : tensorflow.Variable
            The input tensor to this layer
        out_size : int
            The shape of the vector out of this layer
        use_bias : bool
            Wether or not this layer should have a bias term
        strides : int or array_like(length=2)
            The strides used for this layer. Asymmetric strides are accepted as a 
            length two array, where the first number is the vertical strides and
            the second number is the horizontal strides.
        padding : str
            How to deal with boundary conditions in the convolutions. Accepted 
            values are 'SAME' and 'VALID'. 'SAME' uses the value of the nearest 
            pixel and 'VALID' crops the image so that boundary conditions aren't
            a problem.

        Returns:
        --------
        out : tensorflow.Variable
            Output tensor of this layer
        """
        out = tf.layers.conv2d_transpose(
            self.input,
            out_size,
            kernel_size=k_size,
            use_bias=use_bias,
            kernel_initializer=self.initializer,
            strides=strides,
            padding=padding,
            kernel_regularizer=self.regularizer,
        )
        out = self.activation(out)
        out = self.normalizer(out, training=self.is_training, name="BN")

        return out

    def _print_info(self, layer_params):
        print(
            "________________Convolutional layer________________\n",
            "Variable_scope: {}\n".format(self.vscope.name),
            "Kernel size: {}\n".format(layer_params.get("k_size", 3)),
            "Output filters: {}\n".format(layer_params["out_size"]),
            "Strides: {}\n".format(layer_params.get("strides", 1)),
            "Padding: {}\n".format(layer_params.get("padding", "SAME")),
            "Kernel initialisation: {}\n".format(self._init_str),
            "Activation function: {}\n".format(self._act_str),
            "Kernel regularisation: {}\n".format(self._reg_str),
            "Number of regularizer loss: {}".format(len(self.reg_list)),
            "Use bias: {}\n".format(layer_params.get("use_bias", True)),
            "Normalization: {}\n".format(self._normalizer_str),
            "Input shape: {}\n".format(self.input.get_shape().as_list()),
            "Output shape: {}".format(self.output.get_shape().as_list()),
        )


class GlobalConvolutionalLayer(BaseLayer):
    """A Global Convolutional layer as described in [1].

    [1]: Large Kernel Matters - Improve Semantic Segmentation by Global
         Convolutional Network.
    """

    def _gcn_convolution(
        in_tensor,
        out_size,
        k_size=21,
        use_bias=True,
        dilation_rate=1,
        strides=1,
        padding="SAME",
    ):
        if isinstance(strides, int):
            strides = [strides, strides]
        if isinstance(dilation_rate, int):
            dilation_rate = [dilation_rate, dilation_rate]

        out1 = tf.layers.conv2d(
            in_tensor,
            out_size,
            kernel_size=[1, k_size],
            use_bias=use_bias,
            kernel_initializer=self.initializer,
            strides=[1, strides[1]],
            dilation_rate=[1, dilation_rate[1]],
            padding=padding,
            kernel_regularizer=self.regularizer,
        )
        out1 = tf.layers.conv2d(
            in_tensor,
            out_size,
            kernel_size=[k_size, 1],
            use_bias=use_bias,
            kernel_initializer=self.initializer,
            strides=[strides[0], 1],
            dilation_rate=[dilation_rate[0], 1],
            padding=padding,
            kernel_regularizer=self.regularizer,
        )

        out2 = tf.layers.conv2d(
            in_tensor,
            out_size,
            kernel_size=[k_size, 1],
            use_bias=use_bias,
            kernel_initializer=self.initializer,
            strides=[strides[0], 1],
            dilation_rate=[dilation_rate[0], 1],
            padding=padding,
            kernel_regularizer=self.regularizer,
        )
        out2 = tf.layers.conv2d(
            in_tensor,
            out_size,
            kernel_size=[1, k_size],
            use_bias=use_bias,
            kernel_initializer=self.initializer,
            strides=[1, strides[1]],
            dilation_rate=[1, dilation_rate[1]],
            padding=padding,
            kernel_regularizer=self.regularizer,
        )

        return out1, out2

    def _build_layer(
        self,
        out_size,
        k_size=7,
        use_bias=True,
        dilation_rate=1,
        strides=1,
        padding="SAME",
    ):
        out1, out2 = self._gcn_convolution(
            self.input,
            out_size=out_size,
            k_size=k_size,
            use_bias=use_bias,
            dilation_rate=dilation_rate,
            strides=strides,
            padding=padding,
        )
        out = out1 + out2
        out = self.activation(out)
        out = self.normalizer(out, training=self.is_training, name="BN")


class ResnetConv2D(BaseLayer):
    def _build_layer(
        self,
        out_size,
        k_size=3,
        use_bias=True,
        dilation_rate=1,
        strides=1,
        verbose=False,
    ):
        """
        Creates an imporved ResNet layer as described in [1]

        For implementation reasons, this always uses padding.

        [1]: `Identity Mappings in Deep Residual Networks`.

        Parameters
        ----------
        x : tensorflow.Variable
            The input tensor to this layer
        out_size : int
            The shape of the vector out of this layer
        use_bias : bool
            Wether or not this layer should have a bias term
        dilation_rate : int or array_like (length=2)
            The dilation rate used, Assymetric dilation rates are accepted as
            a length two array
        strides : int or array_like (length=2)
            The strides used for this layer. Asymmetric strides are accepted as a
            length two array, where the first number is the vertical strides and 
            the second number is the horizontal strides.
        scope : str
            The scope of this layer (two layers can't share scope).
        verbose : bool
            Wether intermediate steps should be printed in console.
        Returns:
        --------
        out : tensorflow.Variable
            Output tensor of this layer


        Raises:
        -------
        ValueError
            If the initialiser is not valid.
        """
        res_path = self._generate_residual_path(
            out_size, k_size, use_bias, dilation_rate, strides
        )

        skip = self._generate_skip_connection(out_size, strides)

        # Compute ResNet output
        out = skip + res_path

        return out

    def _generate_residual_path(
        self, out_size, k_size=3, use_bias=True, dilation_rate=1, strides=1
    ):
        res_path = self.normalizer(self.input, training=self.is_training, name="BN_1")

        res_path = self.activation(res_path)
        res_path = tf.layers.conv2d(
            res_path,
            self.input.get_shape().as_list()[-1],
            kernel_size=k_size,
            use_bias=use_bias,
            kernel_initializer=self.initializer,
            strides=strides,
            dilation_rate=dilation_rate,
            padding="SAME",
            kernel_regularizer=self.regularizer,
            name="conv2d_1",
        )

        res_path = self.normalizer(res_path, training=self.is_training, name="BN_2")
        res_path = self.activation(res_path)

        res_path = tf.layers.conv2d(
            res_path,
            out_size,
            kernel_size=k_size,
            use_bias=use_bias,
            kernel_initializer=self.initializer,
            padding="SAME",
            kernel_regularizer=self.regularizer,
            name="conv2d_2",
        )

        return res_path

    def _generate_skip_connection(self, out_size, strides):
        shape = self.input.get_shape().as_list()
        if out_size != shape[-1]:
            return tf.layers.conv2d(
                self.input,
                out_size,
                kernel_size=1,
                use_bias=True,
                kernel_initializer=self.initializer,
                strides=strides,
                dilation_rate=1,
                kernel_regularizer=self.regularizer,
                name="conv2d_skip",
            )
        elif strides != 1:
            if isinstance(strides, int):
                new_shape = np.ceil([shape[1] / strides, shape[2] / strides])
            else:
                new_shape = np.ceil([shape[1] / strides[0], shape[2] / strides[1]])

            return tf.image.resize_images(
                self.input,
                new_shape,
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                align_corners=True,
            )
        return self.input

    def _print_info(self, layer_params):
        print(
            "________________ResNet layer________________\n",
            "Variable_scope: {}\n".format(self.vscope.name),
            "Kernel size: {}\n".format(layer_params.get("k_size", 3)),
            "Output filters: {}\n".format(layer_params["out_size"]),
            "Strides: {}\n".format(layer_params.get("strides", 1)),
            "Dilation rate: {}\n".format(layer_params.get("dilation_rate", 1)),
            "Padding: SAME\n",
            "Kernel initialisation: {}\n".format(self._init_str),
            "Activation function: {}\n".format(self._act_str),
            "Kernel regularisation: {}\n".format(self._reg_str),
            "Number of regularizer loss: {}".format(len(self.reg_list)),
            "Use bias: {}\n".format(layer_params.get("use_bias", True)),
            "Use batch normalization: True\n",
            "Input shape: {}\n".format(self.input.get_shape().as_list()),
            "Output shape: {}".format(self.output.get_shape().as_list()),
        )


class ResnetUpconv2D(ResnetConv2D):
    def _build_layer(self, out_size, k_size=3, use_bias=True, strides=1, verbose=False):
        """
        Creates an imporved ResNet layer as described in [1]

        For implementation reasons, this always uses padding.

        [1]: `Identity Mappings in Deep Residual Networks`.

        Parameters
        ----------
        x : tensorflow.Variable
            The input tensor to this layer
        out_size : int
            The shape of the vector out of this layer
        use_bias : bool
            Wether or not this layer should have a bias term
        strides : int or array_like (length=2)
            The strides used for this layer. Asymmetric strides are accepted as a
            length two array, where the first number is the vertical strides and 
            the second number is the horizontal strides.
        scope : str
            The scope of this layer (two layers can't share scope).
        verbose : bool
            Wether intermediate steps should be printed in console.

        Returns:
        --------
        out : tensorflow.Variable
            Output tensor of this layer

        Raises:
        -------
        ValueError
            If the initialiser is not valid.
        """
        res_path = self._generate_residual_path(
            out_size=out_size, k_size=k_size, use_bias=use_bias, strides=strides
        )
        skip = self._generate_skip_connection(out_size, strides)

        # Compute ResNet output
        out = skip + res_path
        return out

    def _generate_residual_path(self, out_size, k_size=3, use_bias=True, strides=1):
        res_path = self.normalizer(self.input, training=self.is_training, name="BN_1")
        res_path = self.activation(res_path)
        res_path = tf.layers.conv2d(
            res_path,
            self.input.get_shape().as_list()[-1],
            kernel_size=k_size,
            use_bias=use_bias,
            kernel_initializer=self.initializer,
            padding="SAME",
            kernel_regularizer=self.regularizer,
            name="conv2d_1",
        )
        res_path = self.normalizer(res_path, training=self.is_training, name="BN_2")
        res_path = self.activation(res_path)

        res_path = tf.layers.conv2d_transpose(
            res_path,
            out_size,
            kernel_size=k_size,
            use_bias=use_bias,
            kernel_initializer=self.initializer,
            strides=strides,
            padding="SAME",
            kernel_regularizer=self.regularizer,
        )

        return res_path

    def _generate_skip_connection(self, out_size, strides=1):
        if isinstance(strides, int):
            strides = [strides, strides]
        skip = self.input
        shape = skip.get_shape().as_list()
        new_size = [shape[1] * strides[0], shape[2] * strides[1]]

        if out_size != shape[-1]:
            skip = tf.layers.conv2d(
                self.input,
                out_size,
                kernel_size=1,
                use_bias=True,
                kernel_initializer=self.initializer,
                kernel_regularizer=self.regularizer,
                name="conv2d_skip",
            )
        return tf.image.resize_images(
            skip, new_size, align_corners=True, method=tf.image.ResizeMethod.BILINEAR
        )


class ResnetLKM2D(ResnetConv2D):
    """The Resnet node described in [1]

    [1]: Large Kernel Matters - Improve Semantic Segmentation by Global
         Convolutional Network.
    """

    def _generate_residual_path(
        self, out_size, k_size=7, use_bias=True, dilation_rate=1, strides=1
    ):
        if isinstance(strides, int):
            strides = [strides, strides]
        if isinstance(dilation_rate, int):
            dilation_rate = [dilation_rate, dilation_rate]

        out1 = tf.layers.conv2d(
            self.input,
            out_size,
            kernel_size=[1, k_size],
            use_bias=use_bias,
            kernel_initializer=self.initializer,
            strides=[1, strides[1]],
            dilation_rate=[1, dilation_rate[1]],
            padding=padding,
            kernel_regularizer=self.regularizer,
        )
        out1 = self.normalizer(out1, training=self.is_training, name="Normalizer1_1")
        out1 = activation(out1)
        out1 = tf.layers.conv2d(
            out1,
            out_size,
            kernel_size=[k_size, 1],
            use_bias=use_bias,
            kernel_initializer=self.initializer,
            strides=[strides[0], 1],
            dilation_rate=[dilation_rate[0], 1],
            padding=padding,
            kernel_regularizer=self.regularizer,
        )
        out1 = self.normalizer(out1, training=self.is_training, name="Normalizer1_2")
        out1 = activation(out1)

        out2 = tf.layers.conv2d(
            self.input,
            out_size,
            kernel_size=[k_size, 1],
            use_bias=use_bias,
            kernel_initializer=self.initializer,
            strides=[strides[0], 1],
            dilation_rate=[dilation_rate[0], 1],
            padding=padding,
            kernel_regularizer=self.regularizer,
        )
        out1 = self.normalizer(out1, training=self.is_training, name="Normalizer2_1")
        out1 = activation(out1)
        out2 = tf.layers.conv2d(
            out2,
            out_size,
            kernel_size=[1, k_size],
            use_bias=use_bias,
            kernel_initializer=self.initializer,
            strides=[1, strides[1]],
            dilation_rate=[1, dilation_rate[1]],
            padding=padding,
            kernel_regularizer=self.regularizer,
        )
        out1 = self.normalizer(out1, training=self.is_training, name="Normalizer2_2")
        out1 = activation(out1)

        out = tf.concat((out1, out2), axis=-1)

        return tf.layers.conv2d(
            out,
            out_size,
            kernel_size=1,
            use_bias=True,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
            name="conv2d_merge",
        )


class BoundaryRefinementLayer(ResnetConv2D):
    """A boundary refinement layer as described in [1].

    [1]: Large Kernel Matters - Improve Semantic Segmentation by Global
         Convolution Network.
    """

    def _generate_residual_path(
        self, out_size, k_size=3, use_bias=True, dilation_rate=1, strides=1
    ):
        res_path = tf.layers.conv2d(
            self.input,
            out_size,
            kernel_size=k_size,
            use_bias=use_bias,
            kernel_initializer=self.initializer,
            strides=strides,
            dilation_rate=dilation_rate,
            padding="SAME",
            kernel_regularizer=self.regularizer,
            name="conv2d_1",
        )
        res_path = self.activation(res_path)
        return tf.layers.conv2d(
            res_path,
            out_size,
            kernel_size=k_size,
            use_bias=use_bias,
            kernel_initializer=self.initializer,
            padding="SAME",
            kernel_regularizer=self.regularizer,
            name="conv2d_2",
        )


class LearnedAveragePool(Conv2D):
    """
    A learned average pool layer is a conv-layer where the kernel and stride
    size is equal to the pool size..
    """

    def _build_layer(self, pool_size):
        """
        Parameters
        ----------
        x : tensorflow.Variable
            The input tensor to this layer
        pool_size : int
            The size of the pooling window.

        Returns:
        --------
        out : tensorflow.Variable
            Output tensor of this layer
        params : dict
            Dictionary with one or two keys;
            - 'W': The weight tensor
            - 'b': The bias tensor (does not exist if `bias` is False).
        reg_list : list
            List containing all the regularization operators for this layer. 
            Should be added to loss during training.
        """
        shape = self.input.get_shape().as_list()
        return super()._build_layer(
            out_size=shape[-1],
            k_size=pool_size,
            strides=pool_size,
            use_bias=False,
            dilation_rate=1,
        )

    def _print_info(self, layer_params):
        print(
            "________________Learned average pool layer________________\n",
            "Variable_scope: {}\n".format(self.vscope.name),
            "Pooling window size: {}\n".format(layer_params["pool_size"]),
            "Kernel initialisation: {}\n".format(self._init_str),
            "Activation function: {}\n".format(self._act_str),
            "Kernel regularisation: {}\n".format(self._reg_str),
            "Normalization: {}\n".format(self._normalizer_str),
            "Input shape: {}\n".format(self.input.get_shape().as_list()),
            "Output shape: {}".format(self.output.get_shape().as_list()),
        )


class MaxPool(BaseLayer):
    """
    A max pooling layer.
    """

    def _build_layer(self, pool_size):
        """

        Parameters
        ----------
        x : tensorflow.Variable
            The input tensor to this layer
        pool_size : int
            The size of the pooling window.

        Returns:
        --------
        out : tensorflow.Variable
            Output tensor of this layer
        params : dict
            Empty dictionary
        reg_list : list
            Empty list
        """
        out = tf.layers.max_pooling2d(
            inputs=self.input,
            pool_size=pool_size,
            strides=pool_size,
            padding="valid",
            name="max_pool",
        )
        return out

    def _print_info(self, layer_params):
        print(
            "________________Learned average pool layer________________\n",
            "Variable_scope: {}\n".format(self.vscope.name),
            "Pooling window size: {}\n".format(layer_params["pool_size"]),
            "Input shape: {}\n".format(self.input.get_shape().as_list()),
            "Output shape: {}".format(self.output.get_shape().as_list()),
        )


class LinearInterpolate(BaseLayer):
    """
    A linear interpolation layer, used for better upsampling than upconvolutions.
    """

    def _build_layer(self, rate=None, out_size=None):
        if rate is not None and out_size is None:
            shape = self.input.get_shape().as_list()[1:-1]
            out_size = tf.multiply(shape, rate, name="out_size")
        elif (rate is None and out_size is None) or (
            rate is not None and out_size is not None
        ):
            raise ValueError(
                "Either the interpolation rate or output size" " must be set."
            )

        out = tf.image.resize_images(
            self.input,
            out_size,
            method=tf.image.ResizeMethod.BILINEAR,
            align_corners=True,
        )
        return out

    def _print_info(self, layer_params):
        print(
            "______________Linear interpolation layer_____________\n",
            "Variable scope: {}\n".format(self.vscope.name),
            "Input tensor: {}\n".format(self.input.get_shape().as_list()),
            "Output shape: {}".format(self.output.get_shape().as_list()),
        )


class BicubicInterpolate(BaseLayer):
    """
    A bicubic interpolation layer, used for better upsampling than upconvolutions.
    """

    def _build_layer(self, rate=None, out_size=None):
        if rate is not None and out_size is None:
            shape = self.input.get_shape().as_list()[1:-1]
            out_size = tf.multiply(shape, rate, name="out_size")
        elif (rate is None and out_size is None) or (
            rate is not None and out_size is not None
        ):
            raise ValueError(
                "Either the interpolation rate or output size" " must be set."
            )

        out = tf.image.resize_images(
            self.input,
            out_size,
            method=tf.image.ResizeMethod.BICUBIC,
            align_corners=True,
        )

        return out

    def _print_info(self, layer_params):
        print(
            "______________Bicubic interpolation layer_____________\n",
            "Variable scope: {}\n".format(self.vscope.name),
            "Input tensor: {}\n".format(self.input.get_shape().as_list()),
            "Output shape: {}".format(self.output.get_shape().as_list()),
        )


class NearestNeighborInterpolate(BaseLayer):
    """
    A neighbor interpolation layer, used for better upsampling than upconvolutions.
    """

    def _build_layer(self, rate=None, out_size=None):
        if rate is not None and out_size is None:
            shape = self.input.get_shape().as_list()[1:-1]
            out_size = tf.multiply(shape, rate, name="out_size")
        elif (rate is None and out_size is None) or (
            rate is not None and out_size is not None
        ):
            raise ValueError(
                "Either the interpolation rate or output size" " must be set."
            )

        out = tf.image.resize_images(
            self.input,
            out_size,
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            align_corners=True,
        )

        return out

    def _print_info(self, layer_params):
        print(
            "______________Nearest interpolation layer_____________\n",
            "Variable scope: {}\n".format(self.vscope.name),
            "Input tensor: {}\n".format(self.input.get_shape().as_list()),
            "Output shape: {}".format(self.output.get_shape().as_list()),
        )


class GlobalAveragePool(BaseLayer):
    """
    Global average pool, used for image classification
    """

    def _build_layer(self):
        out = tf.reduce_mean(
            self.input,
            axis=tf.range(1, len(self.input.get_shape().as_list()) - 1),
            keepdims=True,
        )

        return out

    def _print_info(self, layer_params):
        print(
            "______________Global average pool_____________\n",
            "Variable scope: {}\n".format(self.vscope.name),
            "Input tensor: {}\n".format(self.input.get_shape().as_list()),
            "Output shape: {}".format(self.output.get_shape().as_list()),
        )
