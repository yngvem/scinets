"""
TODO: Move TensorBoard log methods to separate class.
"""
__author__ = "Yngve Mardal Moe"
__email__ = "yngve.m.moe@gmail.com"


import tensorflow as tf
import numpy as np
from pathlib import Path
import h5py
from copy import deepcopy
from collections import ChainMap


class BaseLogger:
    def __init__(
        self,
        evaluator,
        additional_vars=None,
        log_dicts=None,
        train_log_dicts=None,
        val_log_dicts=None,
    ):
        """Superclass for loggers.

        Parameters
        ----------
        network : SciNets.utils.Evaluator
            Network instance to log from
        log_dicts : list
            List of dictionaries specifying the logs that should be logged using
            only training data and validation data.
        train_log_dicts : list
            List of dictionaries specifying the logs that should be logged using
            only training data.
        val_log_dicts : list
            List of dictionaries specifying the logs that should be logged using
            only validation data.
        additional_vars : dict
            Dictionary of additional variables that can be logged.
        """
        if additional_vars is None:
            additional_vars = {}
        self.additional_vars = additional_vars

        # Set default values
        log_dicts = [] if log_dicts is None else log_dicts
        train_log_dicts = [] if train_log_dicts is None else train_log_dicts
        val_log_dicts = [] if val_log_dicts is None else val_log_dicts

        # Set logging parameters
        self.evaluator = evaluator
        self.network = evaluator.network
        self.log_dicts = log_dicts
        self.train_log_dicts = train_log_dicts
        self.val_log_dicts = val_log_dicts

    def _get_log_var(self, var_name):
        """Gets the TensorFlow variable to log.

        This function firsts checks the `additional_vars` dict for the given
        variable, next it checks the evaluator if it is an attribute before
        checking the network that is logged. Finally, it checks in the parameters
        dictionary of the network.
        
        An AttributeError is raised if the variable can't be found.

        Attributes
        ----------
        var_name : str
            Name of the variable to get.
        """
        if var_name in self.additional_vars:
            return self.additional_vars[var_name]
        elif hasattr(self.evaluator, var_name):
            return getattr(self.evaluator, var_name)
        elif hasattr(self.network, var_name):
            return getattr(self.network, var_name)
        elif var_name in self.network.params:
            return self.network.params[var_name]
        else:
            raise AttributeError(
                f"{var_name} not an attribute of {self.network.name} or "
                "in its parameter dictionary."
            )

    def _init_logs(self, log_dicts):
        """Initiate the logging operators specified in `log_dictsj`.

        Parameters:
        -----------
        log_dicts : list
            List of dictionaries specifying the kind of logs to create.
            See `__init__` docstring for examples.

        Returns:
        --------
        list : List with all logging operators.
        """
        log_list = []
        for log_dict in log_dicts:
            log_dict = deepcopy(log_dict)
            log_dict["var_name"] = log_dict["log_var"]
            log_dict["log_var"] = self._get_log_var(log_dict["log_var"])

            log_list.append(self._init_log(**log_dict))
        return log_list

    def _init_log(self, log_var, var_name, *args, **kwargs):
        """Create a specific log operator.
        """
        raise NotImplementedError("Subclasses must implement this function")

    def log_multiple(self, summaries, it_nums, log_type="train", *args, **kwargs):
        """Log summaries for several single time steps.

        Parameters
        ----------
        summaries : Array like
            List of summaries to log.
        it_nums : int
            List of iteration numbers for the log.
        log_type : str
            Specify wether the train writer or validation writer should
            be used, must be either 'train' or 'val'.
        """
        for summary, it_num in zip(summaries, it_nums):
            self.log(summary, it_num, log_type=log_type, **kwargs)

    def log(self, summary, it_num, log_type="train", *args, **kwargs):
        """Log summaries for a single time step.

        Parameters
        ----------
        summary : str
            The output of a summary operator
        it_num : int
            Iteration number.
        log_type : str
            Specify wether the train writer or validation writer should
            be used, must be either 'train' or 'val'.
        """
        log_type = log_type.lower()
        if log_type != "train" and log_type != "val":
            raise ValueError("`log_type` must be either `train` or `val`.")

        self._log(summary, it_num, log_type, **kwargs)

    def _log(summary, it_num, log_type, *args, **kwargs):
        raise NotImplementedError("This should be overloaded by subclasses")


class TensorboardLogger(BaseLogger):
    def __init__(
        self,
        evaluator,
        log_dicts=None,
        train_log_dicts=None,
        val_log_dicts=None,
        log_dir="./logs",
        additional_vars=None,
    ):
        """Initiates a TensorBoard logger.

        Summary operators are created according to the parameters specified
        in the `log_dict`, `train_log_dict` and `val_log_dict` dictionaries.
        The `log_dict` dictionary contains the parameters that should be
        logged both with training and validation data, whereas the
        `train_log_dict` and the `val_log_dict` specifies the summaries that
        should be created for only the training data and validation data
        respectively. The structure of the dictionaries are as follows:
            ```
            [
                {'log_name': 'Name of log 1',
                 'log_var': 'first_log_var',
                 'log_types': 'first_log_type'},

                {'log_name': 'Name of log 2',
                 'log_var': 'first_log_var',
                 'log_types': 'second_log_type'},

                {'log_name': 'Name of log 3',
                 'log_var': 'second_log_var',
                 'log_types': 'third_log_type'}
            ]
            ```
        The keys of the dictionaries are the name of the variables that we
        want to log. For example, if you want to log the loss of the network,
        this should the key should simply be 'loss'. First, the evaluator
        instance is scanned for variable with the specified name (in this case,
        `loss`), then, if no variable with that name is found the network
        instance is scanned. Finally, if there is no variable with the
        specified name in the network instance the trainable parameters of the
        network is scanned.

        Below is an example of how the
        `log_dict` dictionary might look.
            ```
            [
                {'log_name': 'Loss',
                 'log_var': 'loss',
                 'log_type': 'line'},

                {'log_name': 'Predictions',
                 'log_var': 'out',
                 'log_type': 'histogram'},

                {'log_name': 'Predictions',
                 'log_var': 'out',
                 'log_type': 'image'},

                {'log_name': 'Conv1',
                 'log_var': 'conv1/weights',
                 'log_type': 'histogram'},

                {'log_name': 'Conv2',
                 'log_var': 'conv2/bias',
                 'log_type': 'histogram'}

                {'log_name': 'Conv2',
                 'log_var': 'conv2/bias',
                 'log_type': 'image'}
            ]
            ```

        Parameters
        ----------
        network : model.NeuralNet
            Network instance to log from
        log_dicts : list
            List of dictionaries specifying the logs that should be logged using
            both training data and validation data.
        train_log_dicts : list
            List of dictionaries specifying the logs that should be logged using
            only training data.
        val_log_dicts : list
            List of dictionaries specifying the logs that should be logged using
            only validation data.
        log_dir : str or pathlib.Path
            The directory to store the logs in.
        """
        super().__init__(
            evaluator=evaluator,
            additional_vars=additional_vars,
            log_dicts=log_dicts,
            train_log_dicts=train_log_dicts,
            val_log_dicts=val_log_dicts,
        )

        # Prepare for file writers
        self.log_dir = Path(log_dir) / self.network.name / "tensorboard"
        self.train_writer = None
        self.val_writer = None
        self.save_step = None
        self.log_step = None

        with tf.name_scope("summaries"):
            self.train_summary_op, self.val_summary_op = self._init_merged_logs()

    def _init_merged_logs(self):
        both_summary_ops = self._init_logs(self.log_dicts)
        train_summary_ops = self._init_logs(self.train_log_dicts)
        val_summary_ops = self._init_logs(self.val_log_dicts)

        train_summary_op = tf.summary.merge(both_summary_ops + train_summary_ops)
        val_summary_op = tf.summary.merge(both_summary_ops + val_summary_ops)
        return train_summary_op, val_summary_op

    def _init_log(self, log_var, log_type, log_name, var_name, log_kwargs=None):
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
        if not hasattr(self, "_create_" + log_type + "_log"):
            available_log_types = "\n".join(self.available_log_types())
            raise AttributeError(
                f"{log_type} is not a valid logging type, valid log types are:"
                f"\n {available_log_types}"
            )
        if log_kwargs is None:
            log_kwargs = {}

        log_function = getattr(self, "_create_" + log_type + "_log")
        with tf.name_scope(var_name.replace(":", "_")):
            return log_function(log_name=log_name, log_var=log_var, **log_kwargs)

    def init_file_writers(self, session, save_step=100):
        """Initiate the FileWriters that save the train and validation logs.

        Parameters
        ----------
        session : tensorflow.Session
        save_step : int
            How often results should be stored to disk
        """
        self.session = session
        self.train_writer = tf.summary.FileWriter(
            str(self.log_dir / "train"), self.session.graph
        )
        self.val_writer = tf.summary.FileWriter(str(self.log_dir / "test"))
        self.save_step = save_step

    def log(self, summary, it_num, log_type="train"):
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
        super().log(summary=summary, it_num=it_num, log_type=log_type)

        if it_num % self.save_step == 0:
            self.save_logs()

    def _log(self, summary, it_num, log_type="train"):
        if log_type == "train":
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
            attribute[9:-4]
            for attribute in dir(cls)
            if attribute[:8] == "_create_" and attribute[-4:] == "_log"
        ]

    # ----------------------------- Log methods ----------------------------- #
    def _create_histogram_log(self, log_name, log_var, family=None, ignore_nan=False):
        if ignore_nan:
            log_var = tf.boolean_mask(log_var, tf.logical_not(tf.is_nan(log_var)))
        return tf.summary.histogram(log_name, log_var, family=family)

    def _create_scalar_log(self, log_name, log_var, family=None):
        return tf.summary.scalar(log_name, tf.reduce_mean(log_var), family=family)

    def _create_log_scalar_log(self, log_name, log_var, family=None):
        log_var = tf.log(log_var)
        return tf.summary.scalar(log_name, log_var, family=family)

    def _create_image_log(
        self, log_name, log_var, max_outputs=3, channel=None, family=None
    ):
        if channel is not None:
            log_var = log_var[..., channel, tf.newaxis]

        return tf.summary.image(
            log_name, log_var, max_outputs=max_outputs, family=family
        )

    def _create_gradient_histogram_log(self, log_name, log_var, family="Gradients"):
        grads = tf.gradients(self.network.loss, [log_var])[0]
        return tf.summary.histogram(log_name, grads, family=family)


class HDF5Logger(BaseLogger):
    def __init__(
        self,
        evaluator,
        log_dicts=None,
        train_log_dicts=None,
        val_log_dicts=None,
        log_dir="./logs",
        filename="logs.h5",
    ):
        """Initiate a HDF5 logger.


        Summary operators are created according to the parameters specified
        in the `log_dict`, `train_log_dict` and `val_log_dict` dictionaries.
        The `log_dict` dictionary contains the parameters that should be
        logged both with training and validation data, whereas the
        `train_log_dict` and the `val_log_dict` specifies the summaries that
        should be created for only the training data and validation data
        respectively. The structure of the dictionaries are as follows:
            ```
            [
                {
                    'log_name': 'Name of log 1'
                    'log_var': first_log_var'
                },
                {
                    'log_name': 'Name of log 2'
                    'log_var': 'second_log_var'
                }
            }
            ```
        The keys of the dictionaries are the name of the variables that we
        want to log. For example, if you want to log the loss of the network,
        this should the key should simply be `'loss'`. First, the evaluator
        instance is scanned for variable with the specified name (in this case,
        `loss`), then, if no variable with that name is found the network
        instance is scanned. Finally, if there is no variable with the
        specified name in the network instance the trainable parameters of the
        network is scanned.

        Below is an example of how the
        `log_dict` dictionary might look.
            ```
            [
                {
                    'log_name': 'Loss'
                    'log_var': loss'
                },
                {
                    'log_name': 'Accuracy'
                    'log_var': 'accuracy'
                }
            ]
            ```

        Parameters:
        -----------
        evaluator : utils.Evaluator
            The network evaluator to log from.
        log_dict : dict
            Logging dictionary used for both training and validation logs.
        train_log_dict: dict
            Logging dictionary used for training logs.
        val_log_dict: dict
            Logging dictionary used for validation logs.
        """
        super().__init__(
            evaluator=evaluator,
            log_dicts=log_dicts,
            train_log_dicts=train_log_dicts,
            val_log_dicts=val_log_dicts,
        )

        self.log_dir = Path(log_dir) / self.network.name
        self.filename = filename
        self.filepath = self.log_dir / filename
        self._init_logfile()
        both_summary_ops = self._init_logs(self.log_dicts)
        self.train_summary_op = self._join_summaries(
            self._init_logs(self.train_log_dicts), both_summary_ops
        )
        self.val_summary_op = self._join_summaries(
            self._init_logs(self.val_log_dicts), both_summary_ops
        )

    def _init_logfile(self):
        """Initiate an empty h5 file with the correct groups to write the logs in.
        """
        if not self.log_dir.is_dir():
            self.log_dir.mkdir(parents=True)

        with h5py.File(self.filepath, "w") as h5:
            h5.create_group("train")
            h5.create_group("val")
            h5["train"].create_dataset(
                "it", dtype=np.int32, shape=(0,), maxshape=(None,)
            )
            h5["val"].create_dataset("it", dtype=np.int32, shape=(0,), maxshape=(None,))

    def _join_summaries(self, *args):
        """Join the summaries to one summary list with one dict.

        The input is a series of lists containing one dictionary,
        and the output is a single list with one element which is a joined
        version of all input dictionaries.
        """
        return dict(ChainMap(*args))

    def _init_logs(self, log_dict):
        """Initiate the logging operators specified in `log_dicts`.

        The logging operator is a single dictionary with variable name as keys
        and the corresponding tensorflow operators as values.

        Parameters:
        -----------
        log_dicts : list
            List of dictionaries specifying the kind of logs to create.
            See `__init__` docstring for examples.

        Returns:
        --------
        dict : The logging operator
        """
        logs = tuple(super()._init_logs(log_dict))
        return dict(ChainMap(*logs))

    def _init_log(self, log_var, var_name, *args, **kwargs):
        """Create a specific log operator.
        
        `*args` and `**kwargs` are ignored.

        Attributes
        ----------
        log_var : tensorflow.Tensor
        log_name : str
        """
        with h5py.File(self.filepath, "a") as h5:
            for group in h5:
                group = h5[group]
                group.create_dataset(
                    var_name, dtype=np.float32, shape=(0,), maxshape=(None,)
                )
        return {var_name: log_var}

    def _log(self, summaries, it_num, log_type):
        """Logs a single time step.
        """
        with h5py.File(self.filepath, "a") as h5:
            group = h5[log_type]
            group["it"].resize((group["it"].shape[0] + 1,))
            group["it"][-1] = it_num
            for name, s in summaries.items():
                dataset = group[name]
                dataset.resize((dataset.shape[0] + 1,))
                dataset[-1] = np.mean(s)


class SacredLogger(BaseLogger):
    def __init__(
        self, evaluator, log_dicts=None, train_log_dicts=None, val_log_dicts=None
    ):
        """Initiate a Sacred logger.


        Summary operators are created according to the parameters specified
        in the `log_dict`, `train_log_dict` and `val_log_dict` dictionaries.
        The `log_dict` dictionary contains the parameters that should be
        logged both with training and validation data, whereas the
        `train_log_dict` and the `val_log_dict` specifies the summaries that
        should be created for only the training data and validation data
        respectively. The structure of the dictionaries are as follows:
            ```
            [
                {
                    'log_name': 'Name of log 1'
                    'log_var': first_log_var'
                },
                {
                    'log_name': 'Name of log 2'
                    'log_var': 'second_log_var'
                }
            }
            ```
        The keys of the dictionaries are the name of the variables that we
        want to log. For example, if you want to log the loss of the network,
        this should the key should simply be `'loss'`. First, the evaluator
        instance is scanned for variable with the specified name (in this case,
        `loss`), then, if no variable with that name is found the network
        instance is scanned. Finally, if there is no variable with the
        specified name in the network instance the trainable parameters of the
        network is scanned.

        Below is an example of how the
        `log_dict` dictionary might look.
            ```
            [
                {
                    'log_name': 'Loss'
                    'log_var': loss'
                },
                {
                    'log_name': 'Accuracy'
                    'log_var': 'accuracy'
                }
            ]
            ```

        Parameters:
        -----------
        evaluator : utils.Evaluator
            The network evaluator to log from.
        log_dict : dict
            Logging dictionary used for both training and validation logs.
        train_log_dict: dict
            Logging dictionary used for training logs.
        val_log_dict: dict
            Logging dictionary used for validation logs.
        """
        super().__init__(
            evaluator=evaluator,
            log_dicts=log_dicts,
            train_log_dicts=train_log_dicts,
            val_log_dicts=val_log_dicts,
        )

        both_summary_ops = self._init_logs(self.log_dicts)
        self.train_summary_op = self._join_summaries(
            self._init_logs(self.train_log_dicts), both_summary_ops
        )
        self.val_summary_op = self._join_summaries(
            self._init_logs(self.val_log_dicts), both_summary_ops
        )

    def _join_summaries(self, *args):
        """Join the summaries to one summary list with one dict.

        The input is a series of lists containing one dictionary,
        and the output is a single list with one element which is a joined
        version of all input dictionaries.
        """
        return [dict(ChainMap(*map(lambda x: x[0], args)))]

    def _init_logs(self, log_dict):
        """Initiate the logging operators specified in `log_dictsj`.

        Parameters:
        -----------
        log_dicts : list
            List of dictionaries specifying the kind of logs to create.
            See `__init__` docstring for examples.

        Returns:
        --------
        list : List with one logging operators.
        """
        logs = tuple(super()._init_logs(log_dict))
        return [dict(ChainMap(*logs))]

    def _init_log(self, log_var, log_name, *args, **kwargs):
        """Create a specific log operator.
        
        `*args` and `**kwargs` are ignored.

        Attributes
        ----------
        log_var : tensorflow.Tensor
        log_name : str
        """
        return {log_name: log_var}

    def log_multiple(self, summaries, it_nums, log_type="train", _run=None):
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
        _run : sacred.Run
            The sacred instance to use for logging.
        """
        if _run is None:
            raise ValueError("Run instance must be provided.")

        summary = {log_var: 0 for log_var in summaries[0][0]}
        for log in summaries:
            for log_var, log_value in log[0].items():
                summary[log_var] += np.mean(log_value) / len(it_nums)
        self.log([summary], it_nums[-1], log_type=log_type, _run=_run)

    def log(self, summary, it_num, log_type="train", _run=None):
        """Log summaries for a single time step.

        Parameters
        ----------
        summary : dict
            A dictionary where the keys are the name of the variables
            and the values are the variables' values.
        it_num : int
            Iteration number.
        log_type : str
            Specify wether the train writer or validation writer should
            be used.
        _run : sacred.Run
            The sacred instance to use for logging.
        """
        if _run is None:
            raise ValueError("Run instance must be provided.")
        super().log(summary=summary, it_num=it_num, _run=_run, log_type=log_type)

    def _log(self, summary, it_num, log_type, _run):
        """Logs a single time step.
        """
        it_num = int(it_num)
        for s_dicts in summary:
            for name, s in s_dicts.items():
                name = "{}_{}".format(name, log_type)
                s = np.mean(s)
                if np.isnan(s):
                    s = -1
                _run.log_scalar(name, s, it_num)
