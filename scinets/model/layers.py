__author__ = 'Yngve Mardal Moe'
__email__ = 'yngve.m.moe@gmail.com'


import tensorflow as tf
import numpy as np
from . import activations
from . import regularizers
from . import normalizers
from . import initializers


_ver = [int(v) for v in tf.__version__.split('.')]
keras = tf.keras if _ver[0] >= 1 and _ver[1] >= 4 else tf.contrib.keras


class BaseLayer:
    def __init__(self, x, initializer=None, regularizer=None, activation=None,
                 normalizer=None, is_training=None, scope=None,
                 layer_params=None, verbose=False, *args, **kwargs):
        if normalizer is not None and is_training is None:
            raise ValueError(
                'You have to supply the `is_training` placeholder for batch norm.'
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
            self.output, self.params, self.reg_list = self._build_layer(
                **layer_params
            )

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
        operator_name = operator_dict['operator']
        operator_args = operator_dict.get('arguments', {})
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
        if str(initializer) == 'None':
            initializer = {'operator': 'he_normal'}

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
        if str(activation) == 'None':
            activation = {'operator': 'linear'}
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
        if str(regularizer) == 'None' or regularizer == {}:
            return None, 'No regularization.'
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
        if str(normalizer) == 'None' or normalizer == {}:
            def normalizer(x, *args, **kwargs):
                return x
            return normalizer, 'No normalization'
        else:
            if str(self.is_training) == 'None':
                raise RuntimeError(
                    'The `is_training` placeholder must be set if normalization '
                    'is used.'
                )
            return self._get_operator(normalizers, normalizer)

    def _get_returns(self, scope):
        """Get the parameters to return from a layer
        """
        trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                      scope=scope.name)
        params = {
            '/'.join(var.name.split('/')[-2:][:-2]): var for var in trainable
        }
        reg_list = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 
                                     scope=scope.name)

        return params, reg_list

    def _build_layer(activation, initalizer, regularizer):
        raise RuntimeError('This should be implemented')

    def print_layer_info(self, layer_params):
        self._print_info(layer_params)
        self._print_parameter_shapes()

    def _print_info(self, layer_params):
        raise RuntimeError('This should be implemented!')

    def _print_parameter_shapes(self):
        print(' Parameter shapes:')
        for pname, param in self.params.items():
            print('  {}: {}'.format(pname, param.get_shape().as_list()))


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
            out = tf.reshape(x, shape, name='Flatten')
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
            kernel_regularizer=self.regularizer
        )
        out = self.activation(out)
        out = self.normalizer(out, training=self.is_training, name='BN')

        # Get variables
        params, reg_list = self._get_returns(self.vscope)
        return out, params, reg_list

    def _print_info(self, layer_params):
        print(
            '\n________________Fully connected layer________________\n'
            'Variable_scope: {}\n'.format(self.vscope.name),
            'Flattened input: {}\n'.format(self.flattened),
            'Kernel initialisation: {}\n'.format(self._init_str),
            'Activation function: {}\n'.format(self._act_str),
            'Kernel regularisation: {}\n'.format(self._reg_str),
            'Number of regularizer loss: {}'.format(len(self.reg_list)),
            'Use bias: {}\n'.format(layer_params['use_bias']),
            'Normalization: {}\n'.format(self._normalizer_str),
            'Input shape: {}\n'.format(self.input.get_shape().as_list()),
            'Output shape: {}'.format(self.output.get_shape().as_list())
        )


class Conv2D(BaseLayer):
    def _build_layer(self, out_size, k_size=3, use_bias=True, dilation_rate=1,
                     strides=1, padding='SAME'):
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
        regularizer : str
            What regularizer to use on the weights. Acceptable values are the name 
            of callables in the `regularizers.py` file, or None.
        dilation_rate : int
            The dilation rate of this layer (for atrous convolutions). Setting a 
            dilation rate and stride different from 1 at the same time yields 
            an error.
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
        params : dict
            Dictionary with one or two keys;
            - 'W': The weight tensor
            - 'b': The bias tensor (does not exist if `bias` is False).
        reg_list : list
            List containing all the regularization operators for this layer. 
            Should be added to loss during training.
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
            kernel_regularizer=self.regularizer
        )
        out = self.activation(out)
        out = self.normalizer(out, training=self.is_training, name='BN')

        # Get variables
        params, reg_list = self._get_returns(self.vscope)

        return out, params, reg_list

    def _print_info(self, layer_params):
        print(
            '________________Convolutional layer________________\n',
            'Variable_scope: {}\n'.format(self.vscope.name),
            'Kernel size: {}\n'.format(layer_params.get('k_size', 3)),
            'Output filters: {}\n'.format(layer_params['out_size']),
            'Strides: {}\n'.format(layer_params.get('strides', 1)),
            'Dilation rate: {}\n'.format(layer_params.get('dilation_rate', 1)),
            'Padding: {}\n'.format(layer_params.get('padding', 'SAME')),
            'Kernel initialisation: {}\n'.format(self._init_str),
            'Activation function: {}\n'.format(self._act_str),
            'Kernel regularisation: {}\n'.format(self._reg_str),
            'Number of regularizer loss: {}'.format(len(self.reg_list)),
            'Use bias: {}\n'.format(layer_params.get('use_bias', True)),
            'Normalization: {}\n'.format(self._normalizer_str),
            'Input shape: {}\n'.format(self.input.get_shape().as_list()),
            'Output shape: {}'.format(self.output.get_shape().as_list())
        )

class ResnetConv2D(BaseLayer):
    def _build_layer(self, out_size, k_size=3, use_bias=True, dilation_rate=1,
                    strides=1, verbose=False):
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
        initialiser : str
            The initialiser to use for the weights. Accepted initialisers are
            - 'he': Normally distributed with 
                    :math:`\\sigma^2 = \\frac{2}{n_{in}}`
            - 'glorot': Normally distributed with 
                        :math:`\\sigma^2 = \\frac{2}{n_{in} + n_{out}}`
            - 'normal': Normally distributed with a standard deviation of `std`
        std : float
            Standard deviation used for weight initialisation if `initialiser` is 
            set to 'normal'.
        regularizer : str
            What regularizer to use on the weights. Acceptable values are the name 
            of callables in the `regularizers.py` file, or None.
        dilation_rate : int
            The dilation rate of this layer (for atrous convolutions). Setting a 
            dilation rate and stride different from 1 at the same time yields an
            error.
        strides : int or array_like(length=2)
            The strides used for this layer. Asymmetric strides are accepted as a
            length two array, where the first number is the vertical strides and 
            the second number is the horizontal strides.
        activation : str
            What activation function to use at the end of the layer. Acceptable 
            values are the name of callables in the `activations.py` file, or None.
        scope : str
            The scope of this layer (two layers can't share scope).
        verbose : bool
            Wether intermediate steps should be printed in console.
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


        Raises:
        -------
        ValueError
            If the initialiser is not valid.
        """
        # Create residual path
        res_path = self._generate_residual_path(out_size, k_size, use_bias,
                                                dilation_rate, strides)

        # Create skip connection
        skip = self._generate_skip_connection(out_size, strides)

        # Compute ResNet output
        out = skip + res_path

        # Get variables
        params, reg_list = self._get_returns(self.vscope)

        return out, params, reg_list

    def _generate_residual_path(self, out_size, k_size=3, use_bias=True,
                                dilation_rate=1, strides=1):
        res_path = self.normalizer(self.input, training=self.is_training,
                                   name='BN_1')

        res_path = self.activation(res_path)
        res_path = tf.layers.conv2d(
            res_path,
            out_size,
            kernel_size=k_size,
            use_bias=use_bias,
            kernel_initializer=self.initializer,
            strides=strides,
            dilation_rate=dilation_rate,
            padding='SAME',
            kernel_regularizer=self.regularizer,
            name='conv2d_1'
        )

        res_path = self.normalizer(res_path, training=self.is_training,
                                   name='BN_2')
        res_path = self.activation(res_path)

        res_path = tf.layers.conv2d(
            res_path,
            out_size,
            kernel_size=k_size,
            use_bias=use_bias,
            kernel_initializer=self.initializer,
            strides=strides,
            dilation_rate=dilation_rate,
            padding='SAME',
            kernel_regularizer=self.regularizer,
            name='conv2d_2'
        )

        return res_path

    def _generate_skip_connection(self, out_size, strides):
        return tf.layers.conv2d(
            self.input,
            out_size,
            kernel_size=1,
            use_bias=False,
            kernel_initializer=self.initializer,
            strides=strides,
            dilation_rate=1,
            kernel_regularizer=self.regularizer,
            name='conv2d_skip'
        )
        
    def _print_info(self, layer_params):
        print(
            '________________ResNet layer________________\n',
            'Variable_scope: {}\n'.format(self.vscope.name),
            'Kernel size: {}\n'.format(layer_params.get('k_size', 3)),
            'Output filters: {}\n'.format(layer_params['out_size']),
            'Strides: {}\n'.format(layer_params.get('strides', 1)),
            'Dilation rate: {}\n'.format(layer_params.get('dilation_rate', 1)),
            'Padding: SAME\n',
            'Kernel initialisation: {}\n'.format(self._init_str),
            'Activation function: {}\n'.format(self._act_str),
            'Kernel regularisation: {}\n'.format(self._reg_str),
            'Number of regularizer loss: {}'.format(len(self.reg_list)),
            'Use bias: {}\n'.format(layer_params.get('use_bias', True)),
            'Use batch normalization: True\n',
            'Input shape: {}\n'.format(self.input.get_shape().as_list()),
            'Output shape: {}'.format(self.output.get_shape().as_list())
        )


class LearnedAveragePool(Conv2D):
    def _build_layer(self, pool_size):
        """
        An average pooling layer, essentialy a conv-layer with convolution filter size as stride
        length.

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
            dilation_rate=1
        )
            
    def _print_info(self, layer_params):
        print(
            '________________Learned average pool layer________________\n',
            'Variable_scope: {}\n'.format(self.vscope.name),
            'Pooling window size: {}\n'.format(layer_params['pool_size']),
            'Kernel initialisation: {}\n'.format(self._init_str),
            'Activation function: {}\n'.format(self._act_str),
            'Kernel regularisation: {}\n'.format(self._reg_str),
            'Normalization: {}\n'.format(self._normalizer_str),
            'Input shape: {}\n'.format(self.input.get_shape().as_list()),
            'Output shape: {}'.format(self.output.get_shape().as_list())
        )


class MaxPool(BaseLayer):
    def _build_layer(self, pool_size):
        """
        A max pooling layer.

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
            padding='valid',
            name='max_pool'
        )
            
    def _print_info(self, layer_params):
        print(
            '________________Learned average pool layer________________\n',
            'Variable_scope: {}\n'.format(self.vscope.name),
            'Pooling window size: {}\n'.format(layer_params['pool_size']),
            'Input shape: {}\n'.format(self.input.get_shape().as_list()),
            'Output shape: {}'.format(self.output.get_shape().as_list())
        )


class LinearInterpolate(BaseLayer):
    def _build_layer(self, rate=None, out_size=None):
        if rate is not None and out_size is None:
            shape = self.input.get_shape().as_list()[1:-1]
            out_size = tf.multiply(shape, rate, name="out_size")
        elif ((rate is None and out_size is None) or
              (rate is not None and out_size is not None)):
            raise ValueError("Either the interpolation rate or output size"
                             " must be set.")

        out = tf.image.resize_images(self.input, out_size,
                                     method=tf.image.ResizeMethod.BILINEAR)
        return out, {}, []
    
    def _print_info(self, layer_params):
        print(
            '______________Linear interpolation layer_____________\n',
            'Variable scope: {}\n'.format(self.vscope.name),
            'Input tensor: {}\n'.format(self.input.get_shape().as_list()),
            'Output shape: {}'.format(self.output.get_shape().as_list())
        )

        
class BicubicInterpolate(BaseLayer):
    def _build_layer(self, rate=None, out_size=None):
        if rate is not None and out_size is None:
            shape = self.input.get_shape().as_list()[1:-1]
            out_size = tf.multiply(shape, rate, name="out_size")
        elif ((rate is None and out_size is None) or
              (rate is not None and out_size is not None)):
            raise ValueError("Either the interpolation rate or output size"
                             " must be set.")

        out = tf.image.resize_images(self.input, out_size,
                                     method=tf.image.ResizeMethod.BICUBIC)

        return out, {}, []
    
    def _print_info(self, layer_params):
        print(
            '______________Linear interpolation layer_____________\n',
            'Variable scope: {}\n'.format(self.vscope.name),
            'Input tensor: {}\n'.format(self.input.get_shape().as_list()),
            'Output shape: {}'.format(self.output.get_shape().as_list())
        )


class NearestNeighborInterpolate(BaseLayer):
    def _build_layer(self, rate=None, out_size=None):
        if rate is not None and out_size is None:
            shape = self.input.get_shape().as_list()[1:-1]
            out_size = tf.multiply(shape, rate, name="out_size")
        elif ((rate is None and out_size is None) or
              (rate is not None and out_size is not None)):
            raise ValueError("Either the interpolation rate or output size"
                             " must be set.")

        out = tf.image.resize_images(
            self.input,
            out_size,
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )

        return out, {}, []
    
    def _print_info(self, layer_params):
        print(
            '______________Linear interpolation layer_____________\n',
            'Variable scope: {}\n'.format(self.vscope.name),
            'Input tensor: {}\n'.format(self.input.get_shape().as_list()),
            'Output shape: {}'.format(self.output.get_shape().as_list())
        )

        
def bicubic_interpolate(x, rate=2, out_size=None, axis=None,
                        scope="bicubic_interpolation", verbose=True):
    """Perform linear interpolation of all input images.

    The input must either be a 4D tensor in the case of 2D images or a 
    5D tensor in the case of 3D images. No other tensor dimensions are
    accepted by the TensorFlow interpolation functions.

    Parameters:
    -----------
    x1 : tensorflow.Variable
        The input images.
    rate : float (optional)
        The interpolation rate, if this is equal to two, the image size will
        double, if it is equal to one half, the image size will be halved, etc.
        This is two as default. Either this or `out_size` must be specified.
    out_size : Array like (optional)
        The output size of the images, must have same length as the dimensions
        of the image. 
    scope : str
        The variable scope of this layer.
    verbose : bool
        Wether additional layer information should be printed.
    

    Returns:
    --------
    out : tensorflow.Variable
        Output tensor of this layer
    params : dict
        Empty dictionary.
    reg_list : list
        Empty list.
    """
    with tf.variable_scope(scope) as vscope:
        shape = x.get_shape.as_list()[1:-1]
        if out_size is None:
            out_size = tf.multiply(shape, rate, name="out_size")
        out = tf.image.resize_images(images, out_size,
                                     method=tf.image.ResizeMethod.BICUBIC)

        if verbose:
            print(
                '______________Bicubic interpolation layer_____________\n',
                'Variable scope: {}\n'.format(vscope.name),
                'Input tensor: {}\n'.format(x),
                'Output shape: {}'.format(out.get_shape().as_list())
            )


def nearest_interpolate(x, rate=2, out_size=None, axis=None,
                        scope="nearest_interpolation", verbose=True):
    """Perform linear interpolation of all input images.

    The input must either be a 4D tensor in the case of 2D images or a 
    5D tensor in the case of 3D images. No other tensor dimensions are
    accepted by the TensorFlow interpolation functions.

    Parameters:
    -----------
    x1 : tensorflow.Variable
        The input images.
    rate : float (optional)
        The interpolation rate, if this is equal to two, the image size will
        double, if it is equal to one half, the image size will be halved, etc.
        This is two as default. Either this or `out_size` must be specified.
    out_size : Array like (optional)
        The output size of the images, must have same length as the dimensions
        of the image. 
    scope : str
        The variable scope of this layer.
    verbose : bool
        Wether additional layer information should be printed.
    

    Returns:
    --------
    out : tensorflow.Variable
        Output tensor of this layer
    params : dict
        Empty dictionary.
    reg_list : list
        Empty list.
    """
    with tf.variable_scope(scope) as vscope:
        shape = x.get_shape.as_list()[1:-1]
        if out_size is None:
            out_size = tf.multiply(shape, rate, name="out_size")

        out = tf.image.resize_images(
            images,
            out_size,
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )

        if verbose:
            print(
                '__________Nearest neighbour interpolation layer________\n',
                'Variable scope: {}\n'.format(vscope.name),
                'Input tensor: {}\n'.format(x),
                'Output shape: {}'.format(out.get_shape().as_list())
            )
