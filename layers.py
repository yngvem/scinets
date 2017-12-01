import tensorflow as tf
import numpy as np
import activations
import regularizers


_ver = [int(v) for v in tf.__version__.split('.')]
keras = tf.keras if _ver[0] >= 1 and _ver[1] >= 4 else tf.contrib.keras


def _flatten(x, return_flattened=False):
    """Flattens `x` if `x` is not a batch of 1 dimensional vectors.

    Parameters
    ----------
    x : tf.Variable
    return_flattened : bool
        If True, this function will return whether or not it flattened the input.

    Returns
    -------
    out : tf.Variable
        A batch of 1 dimensional vectors.
    shape : tuple
        The new shape of this tensor.
    flattened : bool
        Whether `x` was flattened or not. Only returned if `return_flattened=False`
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

    return (out, shape, flattened) if return_flattened else (out, shape)


def _generate_initializer(initializer, std=1, generate_string=False):
    """Generates an initializer instance from string.

    Arguments
    ---------
    initializer : str
        What initializer to use. Acceptable values are 'he', 'glorot' (or 'xavier') and
        'normal' (or 'gaussian').
    std : float
        What standard deviation to use (only used if `initializer='normal'` (or 'gaussian')).
    generate_string : bool
        Whether a string describing the initialiser should be returned.

    Returns
    -------
    init_instance : tf.keras.initializer.Initializer
    init_str : str
        A string describing the initialiser returned (only returned if `generate_string=True`).
    """
    if initializer.lower() == 'he':
        init_str = 'He'
        init_instance = keras.initializers.he_normal()
    elif initializer == 'glorot' or initializer == 'xavier':
        init_str = 'Glorot'
        init_instance = keras.initializers.glorot_normal()
    elif initializer == 'normal' or initaliser == 'gaussian':
        init_str = 'Gaussian - std: {}'.format(std)
        init_instance = tf.initializers.random_normal(stddev=std)
    else:
        raise ValueError('`initializer` argument must be the string `he`, `glorot` or `normal`')

    return (init_instance, init_str) if generate_string else init_instance


def _generate_activation(activation, generate_string=False):
    """Generates an activation function from string.

    Arguments
    ---------
    activation : str
        What initializer to use. Acceptable values are the name of callables in the
        `activations.py` file, or None.
    generate_string : bool
        Whether a string describing the activation function should be returned.

    Returns
    -------
    act_func : function
    act_str : str
        A string describing the initialiser returned (only returned if `generate_string=True`).
    """
    activation = 'linear' if activation is None else activation
    act_str = str(activation)
    act_func = getattr(activations, activation)

    return (act_func, act_str) if generate_string else act_func


def _generate_regularizer(regularizer, generate_string=False):
    """Generates an regularization function from string.

    Arguments
    ---------
    regularizer : str
        What regularizer to use. Acceptable values are the name of callables in the
        `regularizers.py` file, or None.
    generate_string : bool
        Whether a string describing the regularization function should be returned.

    Returns
    -------
    reg_func : function
    reg_str : str
        A string describing the initialiser returned (only returned if `generate_string=True`).
    """
    if regularizer is None:
        reg_str = 'No regularization.'
        reg_func = None
    else:
        reg_str = str(regularizer)
        reg_func = getattr(regularizers, regularizer)

    return (reg_func, reg_str) if generate_string else reg_func


def _get_returns(scope):
    """Get the parameters to return from a layer
    """
    trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
    params = {
        '/'.join(var.name.split('/')[-2:][:-2]): var for var in trainable
    }
    reg_list = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope.name)

    return params, reg_list


def fc_layer(x, out_size, use_bias=True, initializer='he', std=0.1, regularizer=None,
             batch_norm=False, is_training=None, activation='linear', scope='fc', verbose=False,
             **kwargs):
    """Creates a fully connected layer with output dimension `out_size`.

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
          - 'he': Normally distributed with :math:`\\sigma^2 = \\frac{2}{n_{in}}`
          - 'glorot': Normally distributed with :math:`\\sigma^2 = \\frac{2}{n_{in} + n_{out}}`
          - 'normal': Normally distributed with standard deviation given by `std`
    std : float
        Standard deviation used for weight initialisation if `initialiser` is set to 'normal'.
    regularizer : str
        What regularizer to use on the weights. Acceptable values are the name of callables in the
        `regularizers.py` file, or None.
    batch_norm : bool
        Wether or not a batch normalization layer should be placed before the activation function.
    is_training : tf.placeholder(tf.bool, [1])
        Used for batch normalization to signal whether the running average should be updated or not.
    activation : str
        What activation function to use. Acceptable values are the name of callables in the
        `activations.py` file, or None.
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
        List containing all the regularization operators for this layer. Should be added to loss
        during training.

    Raises:
    -------
    ValueError
        If the initialiser is not valid.
    """
    if batch_norm == True and is_training is None:
        raise ValueError('You have to supply the `is_training` placeholder for batch norm.')

    initializer, init_str = _generate_initializer(initializer, generate_string=True)
    activation, act_str = _generate_activation(activation, generate_string=True)
    regularizer, reg_str = _generate_regularizer(regularizer, generate_string=True)

    # Build layer
    with tf.variable_scope(scope) as vscope:
        out, shape, flattened = _flatten(x, return_flattened=True)
        with tf.name_scope(scope) as nscope:
            out = tf.layers.dense(out, out_size, use_bias=use_bias, kernel_initializer=initializer,
                                  kernel_regularizer=regularizer)
            if batch_norm:
                out = tf.layers.batch_normalization(out, training=is_training, name='BN')
            out = activation(out)

        # Get variables
        params, reg_list = _get_returns(vscope)

        if verbose:
            print(
                '\n________________Fully connected layer________________\n'
                ' Name scope: {}\n'.format(nscope),
                'Variable_scope: {}\n'.format(vscope.name),
                'Flattened input: {}\n'.format(flattened),
                'Kernel initialisation: {}\n'.format(init_str),
                'Activation function: {}\n'.format(act_str),
                'Kernel regularisation: {}\n'.format(reg_str),
                'Number of regularizer loss: {}'.format(len(reg_list)),
                'Use bias: {}\n'.format(use_bias),
                'Use batch normalization: {}\n'.format(batch_norm),
                'Input shape: {}\n'.format(x.get_shape().as_list()),
                'Output shape: {}'.format(out.get_shape().as_list())
            )
            print(' Parameter shapes:')
            for pname, param in params.items():
                print('  {}: {}'.format(pname, param.get_shape().as_list()))
        return out, params, reg_list


def conv2d(x, out_size, k_size=3, use_bias=True, initializer='he', std=0.1, regularizer=None,
           dilation_rate=1, strides=1, padding='SAME', batch_norm=False, is_training=None,
           activation='linear', scope='conv2d', verbose=False, **kwargs):
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
    initialiser : str
        The initialiser to use for the weights. Accepted initialisers are
          - 'he': Normally distributed with :math:`\\sigma^2 = \\frac{2}{n_{in}}`
          - 'glorot': Normally distributed with :math:`\\sigma^2 = \\frac{2}{n_{in} + n_{out}}`
          - 'normal': Normally distributed with standard deviation given by `std`
    std : float
        Standard deviation used for weight initialisation if `initialiser` is set to 'normal'.
    regularizer : str
        What regularizer to use on the weights. Acceptable values are the name of callables in the
        `regularizers.py` file, or None.
    dilation_rate : int
        The dilation rate of this layer (for atrous convolutions). Setting a dilation rate and
        stride different from 1 at the same time yields an error.
    strides : int or array_like(length=2)
        The strides used for this layer. Asymmetric strides are accepted as a length two array,
        where the first number is the vertical strides and the second number is the horizontal
        strides.
    padding : str
        How to deal with boundary conditions in the convolutions. Accepted values are
        'SAME' and 'VALID'. 'SAME' uses the value of the nearest pixel and 'VALID' crops the
        image so that boundary conditions aren't a problem.
    batch_norm : bool
        Wether or not a batch normalization layer should be placed before the activation function.
    is_training : tf.placeholder(tf.bool, [])
        Used for batch normalization to signal whether the running average should be updated or not.
    activation : str
        What activation function to use. Acceptable values are the name of callables in the
        `activations.py` file, or None.
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
        List containing all the regularization operators for this layer. Should be added to loss
        during training.

    Raises:
    -------
    ValueError
        If the initialiser is not valid.
    """
    if batch_norm == True and is_training is None:
        raise ValueError('You have to supply the `is_training` placeholder for batch norm.')

    initializer, init_str = _generate_initializer(initializer, generate_string=True)
    activation, act_str = _generate_activation(activation, generate_string=True)
    regularizer, reg_str = _generate_regularizer(regularizer, generate_string=True)

    # Build layer
    with tf.variable_scope(scope) as vscope:
        with tf.name_scope(scope) as nscope:
            out = tf.layers.conv2d(
                x,
                out_size,
                kernel_size=k_size,
                use_bias=use_bias,
                kernel_initializer=initializer,
                strides=strides,
                dilation_rate=dilation_rate,
                padding=padding,
                kernel_regularizer=regularizer
            )
            if batch_norm:
                out = tf.layers.batch_normalization(out, training=is_training, name='BN')
            out = activation(out)

        # Get variables
        params, reg_list = _get_returns(vscope)

        if verbose:
            print(
                '________________Convolutional layer________________\n',
                ' Name scope: {}\n'.format(nscope),
                'Variable_scope: {}\n'.format(vscope.name),
                'Kernel size: {}\n'.format(k_size),
                'Output filters: {}\n'.format(out_size),
                'Strides: {}\n'.format(strides),
                'Dilation rate: {}\n'.format(dilation_rate),
                'Padding: {}\n'.format(padding),
                'Kernel initialisation: {}\n'.format(init_str),
                'Activation function: {}\n'.format(act_str),
                'Kernel regularisation: {}\n'.format(reg_str),
                'Number of regularizer loss: {}'.format(len(reg_list)),
                'Use bias: {}\n'.format(use_bias),
                'Use batch normalization: {}\n'.format(batch_norm),
                'Input shape: {}\n'.format(x.get_shape().as_list()),
                'Output shape: {}'.format(out.get_shape().as_list())
            )
            print(' Parameter shapes:')
            for pname, param in params.items():
                print('  {}: {}'.format(pname, param.get_shape().as_list()))

        return out, params, reg_list


def resnet_conv_2d(x, out_size, k_size=3, use_bias=True, initializer='he', std=0.1, regularizer=None,
                   dilation_rate=1, strides=1, is_training=None, activation='linear',
                   scope='resnet_conv2d', verbose=False, **kwargs):
    """
    Creates an imporved ResNet layer as described in `Identity Mappings in Deep Residual Networks`.

    For implementation reasons, this always uses padding.

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
          - 'he': Normally distributed with :math:`\\sigma^2 = \\frac{2}{n_{in}}`
          - 'glorot': Normally distributed with :math:`\\sigma^2 = \\frac{2}{n_{in} + n_{out}}`
          - 'normal': Normally distributed with standard deviation given by `std`
    std : float
        Standard deviation used for weight initialisation if `initialiser` is set to 'normal'.
    regularizer : str
        What regularizer to use on the weights. Acceptable values are the name of callables in the
        `regularizers.py` file, or None.
    dilation_rate : int
        The dilation rate of this layer (for atrous convolutions). Setting a dilation rate and
        stride different from 1 at the same time yields an error.
    strides : int or array_like(length=2)
        The strides used for this layer. Asymmetric strides are accepted as a length two array,
        where the first number is the vertical strides and the second number is the horizontal
        strides.
    activation : str
        What activation function to use at the end of the layer. Acceptable values are the name of
        callables in the `activations.py` file, or None.
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
        List containing all the regularization operators for this layer. Should be added to loss
        during training.


    Raises:
    -------
    ValueError
        If the initialiser is not valid.
    """
    if is_training is None:
        raise ValueError('You have to supply the `is_training` placeholder for batch norm.')

    initializer, init_str = _generate_initializer(initializer, generate_string=True)
    activation, act_str = _generate_activation(activation, generate_string=True)
    regularizer, reg_str = _generate_regularizer(regularizer, generate_string=True)
    with tf.variable_scope(scope) as vscope:
        with tf.name_scope(scope) as nscope:
            # Create residual path
            res_path = tf.layers.batch_normalization(x, training=is_training, name='BN_1')
            res_path = activation(res_path)
            res_path = tf.layers.conv2d(
                res_path,
                out_size,
                kernel_size=k_size,
                use_bias=use_bias,
                kernel_initializer=initializer,
                strides=strides,
                dilation_rate=dilation_rate,
                padding='SAME',
                kernel_regularizer=regularizer,
                name='conv2d_1'
            )
            res_path = tf.layers.batch_normalization(res_path, training=is_training, name='BN_2')
            res_path = activation(res_path)
            res_path = tf.layers.conv2d(
                res_path,
                out_size,
                kernel_size=k_size,
                use_bias=use_bias,
                kernel_initializer=initializer,
                strides=strides,
                dilation_rate=dilation_rate,
                padding='SAME',
                kernel_regularizer=regularizer,
                name='conv2d_2'
            )

            # Create skip connection
            skip = tf.layers.conv2d(
                x,
                out_size,
                kernel_size=1,
                use_bias=False,
                kernel_initializer=initializer,
                strides=strides,
                dilation_rate=dilation_rate,
                padding='SAME',
                kernel_regularizer=regularizer,
                name='conv2d_2'
            )

            # Compute ResNet output
            out = skip + out1

            # Get variables
            params, reg_list = _get_returns(vscope)

            if verbose:
                print(
                    '________________ResNet layer________________\n',
                    ' Name scope: {}\n'.format(nscope),
                    'Variable_scope: {}\n'.format(vscope.name),
                    'Kernel size: {}\n'.format(k_size),
                    'Output filters: {}\n'.format(out_size),
                    'Strides: {}\n'.format(strides),
                    'Dilation rate: {}\n'.format(dilation_rate),
                    'Padding: SAME\n',
                    'Kernel initialisation: {}\n'.format(init_str),
                    'Activation function: {}\n'.format(act_str),
                    'Kernel regularisation: {}\n'.format(reg_str),
                    'Number of regularizer loss: {}'.format(len(reg_list)),
                    'Use bias: {}\n'.format(use_bias),
                    'Use batch normalization: True\n',
                    'Input shape: {}\n'.format(x.get_shape().as_list()),
                    'Output shape: {}'.format(out.get_shape().as_list())
                )
                print(' Parameter shapes:')
                for pname, param in params.items():
                    print('  {}: {}'.format(pname, param.get_shape().as_list()))
    return out, params, reg_list


def stochastic_depth_2d(x, out_size, k_size=3, keep_prob=0.5, use_bias=True, initializer='he',
                        std=0.1, regularizer=None, dilation_rate=1, strides=1, is_training=None,
                        activation='linear', scope='stochastic_depth2d', verbose=False, **kwargs):
    """
    Creates an imporved ResNet layer as described in `Identity Mappings in Deep Residual Networks`.

    For implementation reasons, this always uses padding.

    Parameters
    ----------
    x : tensorflow.Variable
        The input tensor to this layer
    out_size : int
        The shape of the vector out of this layer
    keep_prob : float
        The probability with which computation of this layer will be dropped.
    use_bias : bool
        Wether or not this layer should have a bias term
    initialiser : str
        The initialiser to use for the weights. Accepted initialisers are
          - 'he': Normally distributed with :math:`\\sigma^2 = \\frac{2}{n_{in}}`
          - 'glorot': Normally distributed with :math:`\\sigma^2 = \\frac{2}{n_{in} + n_{out}}`
          - 'normal': Normally distributed with standard deviation given by `std`
    std : float
        Standard deviation used for weight initialisation if `initialiser` is set to 'normal'.
    regularizer : str
        What regularizer to use on the weights. Acceptable values are the name of callables in the
        `regularizers.py` file, or None.
    dilation_rate : int
        The dilation rate of this layer (for atrous convolutions). Setting a dilation rate and
        stride different from 1 at the same time yields an error.
    strides : int or array_like(length=2)
        The strides used for this layer. Asymmetric strides are accepted as a length two array,
        where the first number is the vertical strides and the second number is the horizontal
        strides.
    is_training : tf.placeholder(tf.bool, [])
        Used for batch normalization to signal whether the running average should be updated or not
        and whether the layers should be skipped or not.
    activation : str
        What activation function to use at the end of the layer. Acceptable values are the name
        of callables in the `activations.py` file, or None.
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
        List containing all the regularization operators for this layer. Should be added to loss
        during training.

    Raises:
    -------
    ValueError
        If the initialiser is not valid.
    """
    if is_training is None:
        raise ValueError('You have to supply the `is_training` placeholder for batch norm.')

    initializer, init_str = _generate_initializer(initializer, generate_string=True)
    activation, act_str = _generate_activation(activation, generate_string=True)
    regularizer, reg_str = _generate_regularizer(regularizer, generate_string=True)
    with tf.variable_scope(scope) as vscope:
        with tf.name_scope(scope) as nscope:
            # Create residual path
            res_path = tf.layers.batch_normalization(x, training=is_training, name='BN_1')
            res_path = activation(res_path)
            res_path = tf.layers.conv2d(
                res_path,
                out_size,
                kernel_size=k_size,
                use_bias=use_bias,
                kernel_initializer=initializer,
                strides=strides,
                dilation_rate=dilation_rate,
                padding='SAME',
                kernel_regularizer=regularizer,
                name='conv2d_1'
            )
            res_path = tf.layers.batch_normalization(res_path, training=is_training, name='BN_2')
            res_path = activation(res_path)
            res_path = tf.layers.conv2d(
                res_path,
                out_size,
                kernel_size=k_size,
                use_bias=use_bias,
                kernel_initializer=initializer,
                strides=strides,
                dilation_rate=dilation_rate,
                padding='SAME',
                kernel_regularizer=regularizer,
                name='conv2d_2'
            )

            # Creatwhethere skip connection
            skip = tf.layers.conv2d(
                x,
                out_size,
                kernel_size=1,
                use_bias=False,
                kernel_initializer=initializer,
                strides=strides,
                dilation_rate=dilation_rate,
                padding='SAME',
                kernel_regularizer=regularizer,
                name='conv2d_2'
            )

            # Compute stochastic output
            random_num = tf.random_uniform([])
            out = tf.cond(  # If layer is training and layer should be skipped
                tf.logical_and(is_training, rand_num >= keep_prob),
                true_fn=lambda: skip,
                false_fn=lambda: skip + out1
            )

            # Get variables
            params, reg_list = _get_returns(vscope)

            if verbose:
                print(
                    '________________ResNet layer________________\n',
                    ' Name scope: {}\n'.format(nscope),
                    'Variable_scope: {}\n'.format(vscope.name),
                    'Kernel size: {}\n'.format(k_size),
                    'Output filters: {}\n'.format(out_size),
                    'Probability of skipping layer: {}\n'.format(1 - keep_prob),
                    'Strides: {}\n'.format(strides),
                    'Dilation rate: {}\n'.format(dilation_rate),
                    'Padding: SAME\n',
                    'Kernel initialisation: {}\n'.format(init_str),
                    'Number of regularizer loss: {}'.format(len(reg_list)),
                    'Activation function: {}\n'.format(act_str),
                    'Kernel regularisation: {}\n'.format(reg_str),
                    'Use bias: {}\n'.format(use_bias),
                    'Use batch normalization: True\n',
                    'Input shape: {}\n'.format(x.get_shape().as_list()),
                    'Output shape: {}'.format(out.get_shape().as_list())
                )
                print(' Parameter shapes:')
                for pname, param in params.items():
                    print('  {}: {}'.format(pname, param.get_shape().as_list()))
    return out, params, reg_list


def learned_avg_pool(x, pool_size=2, strides=2, initializer='he', std=0.1, regularizer=None,
                     batch_norm=False, is_training=None, padding='SAME', activation='linear',
                     scope='conv2d', verbose=False, **kwargs):
    """
    Creates a convolution layer with `out_size` different filters.

    Parameters
    ----------
    x : tensorflow.Variable
        The input tensor to this layer
    pool_size : int
        The size of the pooling window. Should normally be the same as strides.
    strides : int
        The strides used for this layer. Should normally be the same as pool_size.
    initialiser : str
        The initialiser to use for the weights. Accepted initialisers are
          - 'he': Normally distributed with :math:`\\sigma^2 = \\frac{2}{n_{in}}`
          - 'glorot': Normally distributed with :math:`\\sigma^2 = \\frac{2}{n_{in} + n_{out}}`
          - 'normal': Normally distributed with standard deviation given by `std`
    std : float
        Standard deviation used for weight initialisation if `initialiser` is set to 'normal'.
    regularizer : str
        What regularizer to use on the weights. Acceptable values are the name of callables in the
        `regularizers.py` file, or None.
    padding : str
        How to deal with boundary conditions in the convolutions. Accepted values are
        'SAME' and 'VALID'. 'SAME' uses the value of the nearest pixel and 'VALID' crops the
        image so that boundary conditions aren't a problem.
    activation : str
        What activation function to use. Acceptable values are the name of callables in the
        `activations.py` file, or None.
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
        List containing all the regularization operators for this layer. Should be added to loss
        during training.

    Raises:
    -------
    ValueError
        If the initialiser is not valid.
    """
    shape = x.get_shape().as_list()
    if batch_norm == True and is_training is None:
        raise ValueError('You have to supply the `is_training` placeholder for batch norm.')

    initializer, init_str = _generate_initializer(initializer, generate_string=True)
    activation, act_str = _generate_activation(activation, generate_string=True)
    regularizer, reg_str = _generate_regularizer(regularizer, generate_string=True)

    # Build layer
    with tf.variable_scope(scope) as vscope:
        with tf.name_scope(scope) as nscope:
            out = tf.layers.conv2d(
                x,
                shape[-1],
                kernel_size=pool_size,
                use_bias=False,
                kernel_initializer=initializer,
                strides=strides,
                dilation_rate=1,
                padding=padding,
                kernel_regularizer=regularizer
            )
            out = activation(out)

        # Get variables
        params, reg_list = _get_returns(vscope)

        if verbose:
            print(
                '________________Learned average pool layer________________\n',
                ' Name scope: {}\n'.format(nscope),
                'Variable_scope: {}\n'.format(vscope.name),
                'Strides: {}\n'.format(strides),
                'Pooling window size: {}\n'.format(pool_size),
                'Padding: {}\n'.format(padding),
                'Kernel initialisation: {}\n'.format(init_str),
                'Activation function: {}\n'.format(act_str),
                'Kernel regularisation: {}\n'.format(reg_str),
                'Use batch normalization: {}\n'.format(batch_norm),
                'Input shape: {}\n'.format(x.get_shape().as_list()),
                'Output shape: {}'.format(out.get_shape().as_list())
            )
            print(' Parameter shapes:')
            for pname, param in params.items():
                print('  {}: {}'.format(pname, param.get_shape().as_list()))

        return out, params, reg_list
