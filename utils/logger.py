__author__ = 'Yngve Mardal Moe'
__email__ = 'yngve.m.moe@gmail.com'


import tensorflow as tf
from pathlib import Path


    

class TensorboardLogger:
    def __init__(self, evaluator, log_dict=None, train_log_dict=None,
                 val_log_dict=None, log_dir='./logs', train_collection=None,
                 val_collection=None):
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
        train_collection : str (optional)
            The tensorflow collection to place the training variables. Should
            be different from the validation collection. If this is not 
            specified, the collection
                tensorflow.GraphKeys.SUMMARIES+'_train'
            is used
        val_collection : str (optional)
            The tensorflow collection to place the validation variables. Should
            be different from the train collection. If this is not 
            specified, the collection
                tensorflow.GraphKeys.SUMMARIES+'_val'
            is used
        """
        if log_dict is None and train_log_dict is None and val_log_dict is None:
            raise ValueError()
        
        # Set default values
        log_dict = {} if log_dict is None else log_dict
        train_log_dict = {} if train_log_dict is None else train_log_dict
        val_log_dict = {} if val_log_dict is None else val_log_dict
        if train_collection is None:
            train_collection = tf.GraphKeys.SUMMARIES+'_train'
        if val_collection is None:
            val_collection = tf.GraphKeys.SUMMARIES+'_val'

        # Set logging parameters
        self.evaluator = evaluator
        self.network = evaluator.network
        self.train_log_dict = {**log_dict, **train_log_dict}
        self.val_log_dict = {**log_dict, **val_log_dict}
        self.train_collection = train_collection
        self.val_collection = val_collection
        self.collections = None

        # Prepare for file writers
        self.log_dir = Path(log_dir)/self.network.name/'tensorboard'
        self.train_writer = None
        self.val_writer = None
        self.save_step = None
        self.log_step = None

        with tf.name_scope('summaries'):
            self.train_summary_op = self.init_train_logs()
            self.val_summary_op = self.init_val_logs()

    def init_train_logs(self):
        """Create summary operator for all train logs.
        """
        self.collections = [self.train_collection]
        self.init_logs(self.train_log_dict)
        return tf.summary.merge_all(self.train_collection)

    def init_val_logs(self):
        """Create summary operator for all validation logs.
        """
        self.collections = [self.val_collection]
        self.init_logs(self.val_log_dict)
        return tf.summary.merge_all(self.val_collection)

    def init_logs(self, log_dict):
        """Initiate the logging operators specified in `log_dict`.

        The collections used are the ones currently in `self.collections`.
        """
        for var_name, logs_params in log_dict.items():
            log_var = self._get_log_var(var_name)

            with tf.name_scope(var_name):
                for log_params in logs_params:
                    if 'kwargs' not in log_params:
                        log_params['kwargs'] = {}

                    self.init_log(
                        log_name=log_params['log_name'],
                        log_var=log_var,
                        log_type=log_params['log_type'],
                        log_kwargs = log_params['kwargs']
                    )

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

    def init_log(self, log_name, log_var, log_type, log_kwargs):
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
        log_function(
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
            self.log(summary, it_num, log_type=log_type)

    def log(self, summary, it_num, log_type='train'):
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
        
        self._log(summary, it_num, log_type)
        if it_num % self.save_step == 0:
            self.save_logs()

    def _log(self, summary, it_num, log_type='train'):
        writer = self.train_writer if log_type == 'train' else self.val_writer
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
    # TODO: Move this to separate class.
    def _create_histogram_log(self, log_name, log_var, family=None):
        return tf.summary.histogram(log_name, log_var, family=family,
                                    collections=self.collections)
    
    def _create_scalar_log(self, log_name, log_var, family=None):
        return tf.summary.scalar(log_name, log_var, family=family,
                                 collections=self.collections)
    
    def _create_log_scalar_log(self, log_name, log_var, family=None):
        log_var = tf.log(log_var)
        return tf.summary.scalar(log_name, log_var, family=family,
                                 collections=self.collections)

    def _create_image_log(self, log_name, log_var, max_outputs=3, channel=None,
                          family=None):
        if channel is not None:
            log_var = log_var[..., channel, tf.newaxis]

        return tf.summary.image(log_name, log_var, max_outputs=max_outputs,
                                family=family, collections=self.collections)

    def _create_gradient_histogram_log(self, log_name, log_var,
                                       family='Gradients'):
        grads = tf.gradients(self.network.loss, log_var)[0]
        return tf.summary.histogram(log_name, grads, family=family,
                                    collections=self.collections)
