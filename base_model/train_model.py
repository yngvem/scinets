import tensorflow as tf
from .model import *
from . import optimizers
from . import regularizers
from pathlib import Path
import json


class NetworkTrainer:
    def __init__(self, network, dataset, logger=None, train_op='AdamOptimizer',
                 max_checkpoints=10, verbose=True):
        self.network = network
        self.dataset = dataset
        self.num_steps = 0
        self.epoch_size = dataset.epoch_size
        self.save_step = None

        self.logger = logger
        self.log_folder = Path('./logs')/network.name/checkpoints
        self.verbose = verbose

        self._optimizer, self._train_step = self.init_train_op(train_op)
        self._saver = self.init_saver(max_checkpoints)
    
    def train(self, session, num_steps, save_step):
        """Trains the network for a given number of steps.

        Arguments
        ---------
        session : tensorflow.Session
        num_steps : int
            Number of training steps to perform
        save_step : int
            How often the model should be saved to disk
            """
        self.save_step = save_step
        # if self.logger is None:
        #     raise RuntimeWarning(
        #         'No logger is set, thus output will be stored.'
        #     )
        for _ in range(num_steps):
            self.train_step(session)
    
    def train_step(self, session):
        print('Step {}'.format(self.num_steps))
        batch_x, batch_y = self.dataset.train.next_batch(100)
        batch_x = batch_x.reshape(-1, 28, 28, 1)
        feed_dict = {
            self.network.input: batch_x,
            self.network.true_labels: batch_y,
            self.network.is_training: True
        }
        session.run(self._train_step, feed_dict=feed_dict)
        self.num_steps += 1
        if self.logger is not None: self.logger.log(session, self.num_steps)
        if self.should_save:
            self.save_state(session)

    def save_state(self, session):
        """Save a checkpoint of the model.
        """
        if self.verbose:
            print('Saving model')
        if not self.log_folder.is_dir():
            self.log_folder.mkdir(parents=True)
        file_name = str(self.log_folder/'checkpoint')
        self._saver.save(session, file_name,
                         global_step=self.num_steps)
        if self.verbose:
            print('Model saved')
    
    def load_state(self, session, step_num=None):
        """Load specified checkpoint, latest is used if `step_num` isn't given.
        """
        if step_num==None:
            with (self.log_folder/'latest_step.json') as f:
                step_num = json.load(f)
        if self.verbose:
            print('Loading model checkpoint {}'.format(step_num))
        log_file = str(self.log_folder/'checkpoint-{}'.format(step_num))
        self._saver.restore(session, log_file)
        self.num_steps = step_num
        if self.verbose:
            print('Model loaded')

    def init_saver(self, max_checkpoints):
        """Create an operator for saving and loading model state.
        """
        return tf.train.Saver(max_to_keep=max_checkpoints)

    def init_train_op(self, train_op, **kwargs):
        """Set the operator used for weight updates.

        Parameters
        ----------
        train_op : str
            The optimizer to use for weight updates. Must be the name of an 
            element of the `optimizers.py` file.
        
        Returns
        -------
        optimizer : tensorflow.Operator
            The optimizer operator
        train_step : tensorflow.Operator
            The train step operator, evaluating this perform one train step.
        """
        if self.network.loss is None:
            raise RuntimeError(
                'The network instance has no loss function.'
            )

        Optimizer = getattr(optimizers, train_op)
        optimizer = Optimizer(**kwargs)

        UPDATE_OPS = tf.GraphKeys.UPDATE_OPS
        with tf.control_dependencies(tf.get_collection(UPDATE_OPS)):
            train_step = optimizer.minimize(self.network.loss)
        
        return optimizer, train_step

    @property
    def num_epochs(self):
        return self.num_steps//self.epoch_size
    
    @property
    def should_save(self):
        if self.save_step is None:
            return False
        return self.num_steps % self.save_step == 0
