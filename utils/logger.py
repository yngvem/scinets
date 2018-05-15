"""
TODO: Move TensorBoard log methods to separate class.
"""
__author__ = 'Yngve Mardal Moe'
__email__ = 'yngve.m.moe@gmail.com'


import tensorflow as tf
from pathlib import Path
    

class BaseLogger:
    def __init__(self, evaluator, log_dict=None, train_log_dict=None,
                 val_log_dict=None):
        """Superclass for loggers.

        Parameters
        ----------
        network : SciNets.utils.Evaluator
            Network instance to log from
        log_dict : dict
            Dictionary specifying the logs that should be logged using both
            training data and validation data.
        train_log_dict : dict
            Dictionary specifying the logs that should be logged using only
            training data.
        val_log_dict : dict
            Dictionary specifying the logs that should be logged using only
            validation data.
        """
        # Set default values
        log_dict = {} if log_dict is None else log_dict
        train_log_dict = {} if train_log_dict is None else train_log_dict
        val_log_dict = {} if val_log_dict is None else val_log_dict

        # Set logging parameters
        self.evaluator = evaluator
        self.network = evaluator.network
        self.log_dict = log_dict
        self.train_log_dict = train_log_dict
        self.val_log_dict = val_log_dict
    
    def _get_log_var(self, var_name):
        """Gets the TensorFlow variable to log.

        This function firsts checks if the given variable is an attribute
        of the network to be logged. If it is not, it checks in the parameters
        dictionary. An AttributeError is raised if the variable can't be found.

        Attributes
        ----------
        var_name : str
            Name of the variable to get.
        """
        if hasattr(self.evaluator, var_name):
            return getattr(self.evaluator, var_name)
        elif hasattr(self.network, var_name):
            return getattr(self.network, var_name)
        elif var_name in self.network.params:
            return self.network.params[var_name]
        else:
            raise AttributeError(
                f'{var_name} not an attribute of {self.network.name} or ' \
                    'in its parameter dictionary.'
            )

    def _init_logs(self, log_dict):
        """Initiate the logging operators specified in `log_dict`.

        Parameters:
        -----------
        log_dict : dict
            Dictionary specifying the kind of logs to create. See `__init__`
            docstring for examples.

        Returns:
        --------
        list : List with all logging operators.
        """
        log_list = []
        for var_name, logs_params in log_dict.items():
            log_var = self._get_log_var(var_name)

            with tf.name_scope(var_name):
                for log_params in logs_params:
                    if 'kwargs' not in log_params:
                        log_params['kwargs'] = {}

                    log_list.append(
                        self._init_log(log_var=log_var, **log_params)
                    )
        return log_list

    def _init_log(self, log_var, *args, **kwargs):
        """Create a specific log operator.
        """
        raise NotImplementedError('Subclasses must implement this function')

    def log_multiple(self, summaries, it_nums, log_type='train'):
        """Log summaries for several single time steps.

        Parameters
        ----------
        summaries : Array like
            List of summaries to log.
        it_nums : int
            List of iteration numbers for the log.
        log_type : str
            Specify wether the train writer or validation writer should
            be used.
        """
        for summary, it_num in zip(summaries, it_nums):
            self.log(summary[0], it_num, log_type=log_type)

    def log(self, summary, it_num, log_type='train', *args, **kwargs):
        """Log summaries for a single time step.

        Parameters
        ----------
        summary : str
            The output of a summary operator
        it_num : int
            Iteration number.
        log_type : str
            Specify wether the train writer or validation writer should
            be used.
        """
        log_type = log_type.lower()
        if log_type != 'train' and log_type != 'val':
            raise ValueError('`log_type` must be either `train` or `val`.')
        
        self._log(summary, it_num, log_type, *args, **kwargs)
        if it_num % self.save_step == 0:
            self.save_logs()

    def _log(summary, it_num, log_type, *args, **kwargs):
        raise NotImplementedError('This should be overloaded by subclasses')


class TensorboardLogger:
    def __init__(self, evaluator, log_dict=None, train_log_dict=None,
                 val_log_dict=None, log_dir='./logs'):
        """Initiates a network logger.

        Summary operators are created according to the parameters specified
        in the `log_dict`, `train_log_dict` and `val_log_dict` dictionaries.
        The `log_dict` dictionary contains the parameters that should be
        logged both with training and validation data, whereas the 
        `train_log_dict` and the `val_log_dict` specifies the summaries that
        should be created for only the training data and validation data 
        respectively. The structure of the dictionaries are as follows:
            ```
            {
                'first_log_var': {'log_name': 'Name of log 1',
                                  'log_types': ['first_log_type',
                                                'second_log_type']},
                'second_log_var': {'log_name': 'Name of log 2',
                                   'log_types': ['third_log_type']}
            }
            ```
        The keys of the dictionaries are the name of the variables that we
        want to log. For example, if you want to log the loss of the network,
        this should the key should simply be 'loss'. First, the network instance
        is scanned to see if there is a variable with such a name, then,
        if the network has no variable with the specified name, the parameters
        dictionary of the network is checked. Below is an example of how the
        `log_dict` dictionary might look.
            ```
            {
                'loss': {'log_name': 'Loss',
                        'log_types': ['line']},

                'out': {'log_name': 'Predictions',
                        'log_types': ['histogram', 'image']},
                
                'conv1/weights': {'log_name': 'Conv1',
                                'log_types': ['histogram'] },
                
                'conv2/bias': {'log_name': 'Conv2',
                            'log_types': ['histogram', 'image']}
            }
            ```
        
        Parameters
        ----------
        network : model.NeuralNet
            Network instance to log from
        log_dict : dict
            Dictionary specifying the logs that should be logged using both
            training data and validation data.
        train_log_dict : dict
            Dictionary specifying the logs that should be logged using only
            training data.
        val_log_dict : dict
            Dictionary specifying the logs that should be logged using only
            validation data.
        log_dir : str or pathlib.Path
            The directory to store the logs in.
        """
        super().__init__(
            evaluator=evaluator,
            log_dict=log_dict,
            train_log_dict=train_log_dict,
            val_log_dict=val_log_dict
        )

        # Prepare for file writers
        self.log_dir = Path(log_dir)/self.network.name/'tensorboard'
        self.train_writer = None
        self.val_writer = None
        self.save_step = None
        self.log_step = None

        with tf.name_scope('summaries'):
            self.train_summary_op, self.val_summary_op = self._init_merged_logs()

    def _init_merged_logs(self):
        both_summary_ops = self._init_logs(self.log_dict)
        train_summary_ops = self._init_logs(self.train_log_dict)
        val_summary_ops = self._init_logs(self.val_log_dict)
        
        train_summary_op = tf.summary.merge(
            both_summary_ops + train_summary_ops
        )
        val_summary_op = tf.summary.merge(
            both_summary_ops + val_summary_ops
        )
        return train_summary_op, val_summary_op

    def _init_logs(self, log_dict):
        """Initiate the logging operators specified in `log_dict`.

        Parameters:
        -----------
        log_dict : dict
            Dictionary specifying the kind of logs to create. See `__init__`
            docstring for examples.

        Returns:
        --------
        list : List with all logging operators.
        """
        log_list = []
        for var_name, logs_params in log_dict.items():
            log_var = self._get_log_var(var_name)

            with tf.name_scope(var_name):
                for log_params in logs_params:
                    if 'kwargs' not in log_params:
                        log_params['kwargs'] = {}

                    log_list.append(
                        self._init_log(log_var=log_var, **log_params)
                    )
        return log_list

    def _init_log(self, log_var, log_type, log_name, log_kwargs):
        """Create a specific log operator.

        Attributes
        ----------
        log_var : tensorflow.Tensor
        log_type : str
            Which log to create. To list the available log types, print out
            Logger.available_log_types().
        log_name : str
        log_kwargs : dict
            Dictionary with additional keyword arguments for the log function.
        """
        if not hasattr(self, '_create_'+log_type+'_log'):
            available_log_types = '\n'.join(self.available_log_types())
            raise AttributeError(
                f'{log_type} is not a valid logging type, valid log types are:' \
                f'\n {available_log_types}'

            )
        log_function = getattr(self, '_create_'+log_type+'_log')
        return log_function(
            log_name=log_name,
            log_var=log_var,
            **log_kwargs
        )

    def init_file_writers(self, session, save_step=100):
        """Initiate the FileWriters that save the train and validation logs.

        Parameters
        ----------
        session : tensorflow.Session
        save_step : int
            How often results should be stored to disk
        """
        self.session = session
        self.train_writer = tf.summary.FileWriter(str(self.log_dir/'train'),
                                                  self.session.graph)
        self.val_writer = tf.summary.FileWriter(str(self.log_dir/'test'))
        self.save_step  = save_step
    
    def _log(self, summary, it_num, log_type='train'):
        if log_type == 'train':
            writer = self.train_writer
        else:
            writer = self.val_writer

        writer.add_summary(summary, it_num)

    def save_logs(self):
        """Save all logs to disk.
        """
        self.train_writer.flush()
        self.val_writer.flush()

    @classmethod
    def available_log_types(cls):
        """List all available log methods.
        """
        return [
            attribute[9:-4] for attribute in dir(cls) 
                if attribute[:8] == '_create_' and attribute[-4:] == '_log'
        ]

    # ----------------------------- Log methods ----------------------------- #
    def _create_histogram_log(self, log_name, log_var, family=None):
        return tf.summary.histogram(log_name, log_var, family=family)
    
    def _create_scalar_log(self, log_name, log_var, family=None):
        return tf.summary.scalar(log_name, log_var, family=family)
    
    def _create_log_scalar_log(self, log_name, log_var, family=None):
        log_var = tf.log(log_var)
        return tf.summary.scalar(log_name, log_var, family=family)

    def _create_image_log(self, log_name, log_var, max_outputs=3, channel=None,
                          family=None):
        if channel is not None:
            log_var = log_var[..., channel, tf.newaxis]

        return tf.summary.image(log_name, log_var, max_outputs=max_outputs,
                                family=family)

    def _create_gradient_histogram_log(self, log_name, log_var,
                                       family='Gradients'):
        grads = tf.gradients(self.network.loss, log_var)[0]
        return tf.summary.histogram(log_name, grads, family=family)


class SacredLogger
    def __init__(self, evaluator, log_dict=None, train_log_dict=None,
                 val_log_dict=None):
        super().__init__(
            evaluator=evaluator,
            log_dict=log_dict,
            train_log_dict=train_log_dict,
            val_log_dict=val_log_dict
        )
            # TODO: _INIT_LOG!!!
        def _log(summary, it_num, log_type, log_name, _run):
            # TODO: FIX THIS!!! 
            _run.log(
