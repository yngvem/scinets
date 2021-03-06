import tensorflow as tf
import numpy as np
from .optimizers import get_optimizer
from .lr_schedulers import get_lr_scheduler
from pathlib import Path
from collections import Iterable
import json


class NetworkTrainer:
    """Class used to train a network instance.
    """

    def __init__(
        self,
        model,
        steps_per_epoch,
        log_dir="./logs",
        train_op=None,
        learning_rate_scheduler=None,
        max_checkpoints=10,
        save_step=2000,
        verbose=True,
    ):
        """Trainer class for neural networks.

        Parameters
        ----------
        model : segmentation_nets.model.NeuralNet
            model to train.
        steps_per_epoch : int
        train_op : dict
            Dictionary that parametrise the training operator. The key
            'operator' should map to the name of the training operator
            (which must be defined in `scinets.trainer.optimizers`)
            and the 'arguments' key should map to a kwargs dict that is
            supplied to the training operator.
        learning_rate_scheduler : dict
            Dictionary specifying how the learning rate should be computed.
            If 'learning_rate' is a key in the `train_op['arguments']` dict
            then this should be None.

            The 'operator' key should be the name of a learning rate scheduler,
            e.g. a function in the `scinets.trainer.lr_schedulers` module.
            Additionally, the key 'arguments' can map to a kwargs dict that
            is supplied to the learning rate scheduler.

            Finally, we note that the `self.global_step` is supplied as the
            keyword argument `global_step=self.global_step` to the learning rate
            scheduler.
        max_checkpoints : int
            Maximum number of checkpoints to store.
        save_step : int
            How often to store checkpoints
        verbose : bool
            Whether additional printout should be produced.
        """
        self.model = model
        self.num_steps = 0
        self.steps_per_epoch = steps_per_epoch
        self.save_step = save_step

        self.train_op = train_op

        self.log_dir = Path(log_dir) / model.name / "checkpoints"
        self.verbose = verbose

        with tf.variable_scope("trainer"):
            self.init_global_step()
            self.learning_rate = self.get_learning_rate(learning_rate_scheduler)
            self._optimizer, self._train_step = self.init_train_op()
            self._checkpoint_saver = self.init_saver(max_checkpoints)

    def train(self, session, num_steps, additional_ops=None):
        """Trains the model for a given number of steps.

        Parameters
        ----------
        session : tensorflow.Session
        num_steps : int
            Number of training steps to perform
        additional_ops : list
            List of tensorflow operators to run.
        
        Returns
        -------
        list
            List of tuples containing the output of the operators in
            `additional_ops`.
        list
            List where the i-th element is the iteration number.
        """
        additional_ops = [] if additional_ops is None else additional_ops

        output = [None] * num_steps
        iterations = np.arange(self.num_steps, self.num_steps + num_steps + 1)
        for i in range(num_steps):
            output[i], _ = self.train_step(session, additional_ops=additional_ops)
        return output, iterations

    def train_step(self, session, additional_ops=None, feed_dict=None):
        """Performs one training step of the specified model.

        Parameters
        ----------
        session : tensorflow.Session
        additional_ops : list
            List of additional tensorflow operators to run.
        feed_dict : dict
            The inputs to the model.
        
        Returns
        -------
        tuple
            Tuple containing the output of the operators in `additional_ops`.
        int
            Current iteration number.
        """
        additional_ops = [] if additional_ops is None else additional_ops
        run_list = [self._train_step] + additional_ops

        feed_dict = {} if feed_dict is None else feed_dict
        feed_dict = {self.model.is_training: True, **feed_dict}

        output = session.run(run_list, feed_dict=feed_dict)[1:]
        self.num_steps += 1
        if self.should_save:
            self.save_state(session)

        return output, self.num_steps

    def save_state(self, session):
        """Save a checkpoint of the model.
        """
        if not self.log_dir.is_dir():
            self.log_dir.mkdir(parents=True)
        file_name = str(self.log_dir / "checkpoint")
        self._checkpoint_saver.save(session, file_name, global_step=self.num_steps)

    def load_state(self, session, step_num=None):
        """Load specified checkpoint, latest is used if `step_num` isn't given.
        """
        if step_num == None:
            with (self.log_dir / "latest_step.json") as f:
                step_num = json.load(f)
        if self.verbose:
            print("Loading model checkpoint {}".format(step_num))
        log_file = str(self.log_dir / "checkpoint-{}".format(step_num))
        self._checkpoint_saver.restore(session, log_file)
        self.num_steps = step_num
        if self.verbose:
            print("Model loaded")

    def init_saver(self, max_checkpoints):
        """Create an operator for saving and loading model state.
        """
        return tf.train.Saver(max_to_keep=max_checkpoints)

    def init_global_step(self):
        """Create the global step variable and its update operator.

        The global step variable counts the number of training steps
        performed and is automatically updated each iteration.
        """
        self.global_step = tf.Variable(self.num_steps, name="global_step")
        self.update_global_step = self.global_step.assign(self.global_step + 1)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, self.update_global_step)

    def get_learning_rate(self, lr_op_dict):
        """Creates a learning rate operator if `lr_op_dict` is given.

        The `lr_op_dict` argument should either be `None` or a dictionary where
        the key 'operator' maps to the name of the learning rate operator
        and the key 'arguments' maps to a kwargs dict to be supplied with the
        operator.

        In addition to the 'arguments' kwargs dict, the keyword argument
        `global_step=self.global_step` is supplied.
        """
        if str(lr_op_dict) == "None":
            return None
        LRScheduler = get_lr_scheduler(lr_op_dict["operator"])
        lr_scheduler = LRScheduler(
            global_step=self.global_step,
            steps_per_epoch=self.steps_per_epoch,
            **lr_op_dict["arguments"]
        )
        return lr_scheduler.build_lr_scheduler()

    def init_train_op(self):
        """Set the operator used for weight updates.

        Returns
        -------
        optimizer : tensorflow.Operator
            The optimizer operator
        train_step : tensorflow.Operator
            The train step operator, evaluating this perform one train step.
        """
        if self.model.loss is None:
            raise RuntimeError("The model instance has no loss function.")

        if (
            self.learning_rate is not None
            and "learning_rate" in self.train_op["arguments"]
        ):
            raise ValueError(
                "The learning rate cannot be set with both a "
                "learning rate operator and in the optimizer "
                "argument dictionary"
            )

        lr = self.train_op["arguments"].get("learning_rate", self.learning_rate)
        self.train_op["arguments"]["learning_rate"] = lr

        Optimizer = get_optimizer(self.train_op["operator"])
        optimizer = Optimizer(**self.train_op["arguments"])

        UPDATE_OPS = tf.GraphKeys.UPDATE_OPS
        with tf.control_dependencies(tf.get_collection(UPDATE_OPS)):
            train_step = optimizer.minimize(self.model.loss)

        return optimizer, train_step

    @property
    def num_epochs(self):
        return self.num_steps // self.steps_per_epoch

    @property
    def should_save(self):
        if self.save_step is None:
            return False
        return self.num_steps % self.save_step == 0
