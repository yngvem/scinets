import numpy as np
import tensorflow as tf
import sacred
from pathlib import Path
from tqdm import trange
from ..model import model
from ..trainer import NetworkTrainer
from ..utils import TensorboardLogger, SacredLogger
from ..utils import evaluator
from ..data import HDFDataset, MNISTDataset


class NetworkExperiment:
    def __init__(self, experiment_params, model_params, dataset_params,
                 trainer_params, log_params):
        """
        experiment_parms = {
            'log_dir': './',
            'name': 'test_experiment',
            'continue_old': False,
            'verbose': True
        }

        model_params = {
            'type': 'NeuralNet',
            'model_params: {
                'loss_function': 'sigmoid_cross_entropy_with_logits',
                'loss_kwargs': {},
                'architecture': [
                    {
                        'layer': 'conv2d',
                        'scope': 'conv1',
                        'out_size': 8,
                        'k_size': (5, 1),
                        'batch_norm': True,
                        'activation': 'relu',
                        'regularizer': {
                            'function': 'weight_decay',
                            'arguments': {
                                'amount': 0.5,
                                'name': 'weight_decay'
                            }
                        }
                    },
                    {
                        'layer': 'conv2d',
                        'scope': 'conv2',
                        'out_size': 16,
                        'k_size': 5,
                        'strides': 2,
                        'batch_norm': True,
                        'activation': 'relu',
                    },
                    {
                        'layer': 'conv2d',
                        'scope': 'conv3',
                        'out_size': 16,
                    }
                ]
            }
        }

        log_params = {
            'val_interval': 1000,
            'evaluator': BinaryClassificationEvaluator,
            'tb_params':
                {
                    'log_dict': None,
                    'train_log_dict': None,         (optional)
                    'val_log_dict': None,           (optional)
                    'train_collection': None,       (optional)
                    'val_collection': None          (optional)
                },
            'sacredboard_params':
                {
                    'log_dict': None,
                    'train_log_dict': None,         (optional)
                    'val_log_dict': None,           (optional)
                }
        }

        trainer_params = {
                'train_op': 'AdamOptimizer',
                'train_op_kwargs': None,
                'max_checkpoints': 10,
                'save_step': 10,
            }
        """
        # Set experiment properties
        self.log_dir = self._get_logdir(experiment_params)
        self.continue_old = self._get_continue_old(experiment_params)
        self.name = self._get_name(experiment_params)
        self.val_interval = log_params['val_log_frequency']
        self.verbose = experiment_params['verbose']

        # Create TensorFlow objects
        self.dataset, self.epoch_size = self._get_dataset(dataset_params)
        self.model = self._get_model(model_params)
        self.trainer = self._get_trainer(trainer_params)
        self.evaluator = self._get_evaluator(log_params['evaluator'])
        self.tb_logger = self._get_tensorboard_logger(log_params['tb_params'])
        self.network_tester = self._get_network_tester(log_params['network_tester'])

    def _get_continue_old(self, experiment_params):
        """Extract whether an old experiment should be continued.
        """
        if 'continue_old' in experiment_params:
            return experiment_params['continue_old']
        return False

    def _get_logdir(self, experiment_params):
        """Extract the log directory from the experiment parameters.
        """
        if 'log_dir' in experiment_params:
            return Path(experiment_params['log_dir'])
        return Path('./')

    def _get_name(self, experiment_params):
        """Enumerate the network name.
        """
        name = experiment_params['name']
        if self.continue_old:
            return name

        i = 0
        while self._name_taken(f'{name}_{i:02d}'):
            i += 1
        return f'{name}_{i:02d}'

    def _name_taken(self, name):
        """Checks if the given name is taken.
        """
        return (self.log_dir/name).is_dir()

    def _get_dataset(self, dataset_params):
        dataset = HDFDataset(**dataset_params)
        epoch_size = len(dataset.train_data_reader)
        return dataset, epoch_size

    def _get_model(self, model_params):
        model_type = getattr(model, model_params['type'])
        return model_type(
            input_var=self.dataset.data,
            true_out=self.dataset.target,
            is_training=self.dataset.is_training,
            name=self.name,
            verbose=self.verbose,
            **model_params['network_params']
        )

    def _get_trainer(self, trainer_params):
        return NetworkTrainer(
            self.model,
            epoch_size=self.epoch_size,
            log_dir=self.log_dir,
            verbose=self.verbose,
            **trainer_params
        )

    def _get_evaluator(self, evaluator_type):
        evaluator_type = getattr(evaluator, evaluator_type)
        return evaluator_type(self.model)

    def _get_tensorboard_logger(self, tensorboard_params):
        return TensorboardLogger(
            self.evaluator,
            log_dir=self.log_dir,
            additional_vars={'learning_rate': self.trainer.learning_rate},
            **tensorboard_params
        )

    def _get_network_tester(self, network_tester_params):
        return evaluator.NetworkTester(
            evaluator=self.evaluator,
            dataset=self.dataset,
            is_training=self.dataset.is_training,
            is_testing = self.dataset.is_testing,
            **network_tester_params
        )

    def _init_session(self, sess, continue_old=None, step_num=None):
        """Initialise the session. Must be run before any training iterations.
        """
        if continue_old is None:
            continue_old = self.continue_old

        sess.run([tf.global_variables_initializer(), self.dataset.initializers])
        if continue_old:
            self.trainer.load_state(sess, step_num=step_num)
        self.tb_logger.init_file_writers(sess)

    def _train_its(self, sess):
        """Perform `self.val_interval` train steps and log validation metrics.
        """
        train_summaries, it_num = self.trainer.train(
            session=sess,
            num_steps=self.val_interval,
            additional_ops=[self.tb_logger.train_summary_op, self.model.loss]
        )
        train_summaries = [s[0] for s in train_summaries]
        self.tb_logger.log_multiple(train_summaries, it_num)

        val_summaries = sess.run(
            self.tb_logger.val_summary_op,
            feed_dict={self.model.is_training: False}
            )
        self.tb_logger.log(val_summaries, it_num[-1], log_type='val')

    def train(self, num_steps):
        """Train the specified model for the given number of steps.
        """
        iterator = trange if self.verbose else range
        num_vals = num_steps//self.val_interval
        with tf.Session() as sess:
            self._init_session(sess)
            for i in iterator(num_vals):
                self._train_its(sess)

    def evaluate_model(self, dataset_type, step_num=None):
        with tf.Session() as sess:
            self._init_session(sess, continue_old=True, step_num=step_num)
            return self.network_tester.test_model(dataset_type, sess)

    def get_all_checkpoint_its(self):
        checkpoint_dir = self.trainer.log_dir
        checkpoints = checkpoint_dir.glob('checkpoint-*.index')

        def checkpoint_to_it(checkpoint):
            checkpoint = str(checkpoint)
            it_num = checkpoint.split('-')[1].split('.')[0]
            return int(it_num)
        return [checkpoint_to_it(ch) for ch in checkpoints]

    def evaluate_all_checkpointed_models(self, dataset_type):
        """Returns the performance for all models.
        """
        checkpoint_its = self.get_all_checkpoint_its()

        return {it: self.evaluate_model(dataset_type, it)
                    for it in checkpoint_its}

    def _find_best_checkpoint(self, performances, metric):
        """Find the best checkpoint from a dictionary of performance dicts.

        The keys of the input dictionary should be iteration numbers and
        the values should be dictionaries whose keys are metrics and values
        are mean-std pairs corresponding to the specified metric.
        """
        from operator import itemgetter
        _performance = [(it, *performance[metric]) 
                            for it, performance in performances.items()]
        best_it = sorted(_performance, key=itemgetter(1))
        return best_it[0]

    def find_best_model(self, dataset_type, performance_metric):
        """Returns the iteration number and performance of the best model
        """
        performances = evaluate_all_checkpointed_models(dataset_type)
        
        performances_means = [p[0] for p in performances]

        best_it_idx = np.argmax(performances_means)
        return checkpoint_its[best_it_idx], performances[best_it_idx]

    def save_outputs(self, dataset_type, filename, step_num):
        filename = self.log_dir / f'{filename}_{step_num}'
        with tf.Session() as sess:
            self._init_session(sess, continue_old=True, step_num=step_num)
            self.network_tester.save_outputs(dataset_type, filename, sess)


class SacredExperiment(NetworkExperiment): 
    def __init__(self, _run, experiment_params, model_params, dataset_params,
                 trainer_params, log_params):
        """
        experiment_parms = {
            'log_dir': './',
            'name': 'test_experiment',
            'continue_old': False,
            'num_steps': 10000
        }

        model_params = {
            'type': 'NeuralNet',
            'model_params: {
                'loss_function': 'sigmoid_cross_entropy_with_logits',
                'loss_kwargs': {},
                'architecture': [
                    {
                        'layer': 'conv2d',
                        'scope': 'conv1',
                        'out_size': 8,
                        'k_size': (5, 1),
                        'batch_norm': True,
                        'activation': 'relu',
                        'regularizer': {
                            'function': 'weight_decay',
                            'arguments': {
                                'amount': 0.5,
                                'name': 'weight_decay'
                            }
                        }
                    },
                    {
                        'layer': 'conv2d',
                        'scope': 'conv2',
                        'out_size': 16,
                        'k_size': 5,
                        'strides': 2,
                        'batch_norm': True,
                        'activation': 'relu',
                    },
                    {
                        'layer': 'conv2d',
                        'scope': 'conv3',
                        'out_size': 16,
                    }
                ]
                'verbose': True,
            }
        }

        log_params = {
            'val_interval': 1000,
            'evaluator': BinaryClassificationEvaluator,
            'tb_params':
                {
                    'log_dict': None,
                    'train_log_dict': None,         (optional)
                    'val_log_dict': None,           (optional)
                    'train_collection': None,       (optional)
                    'val_collection': None          (optional)
                },
            'sacredboard_params':
                {
                    'log_dict': None,
                    'train_log_dict': None,         (optional)
                    'val_log_dict': None,           (optional)
                }
        }

        trainer_params = {
                'train_op': 'AdamOptimizer',
                'train_op_kwargs': None,
                'max_checkpoints': 10,
                'save_step': 10,
                'verbose': True
            }
        """
        self._run = _run
        

        # Set experiment properties
        self.log_dir = self._get_logdir(experiment_params)
        self.continue_old = self._get_continue_old(experiment_params)
        self.name = self._get_name(experiment_params)
        self.val_interval = log_params['val_log_frequency']
        self.verbose = experiment_params['verbose']

        # Create TensorFlow objects
        self.dataset, self.epoch_size = self._get_dataset(dataset_params)
        self.model = self._get_model(model_params)
        self.trainer = self._get_trainer(trainer_params)
        self.evaluator = self._get_evaluator(log_params['evaluator'])
        self.tb_logger = self._get_tensorboard_logger(log_params['tb_params'])
        self.sacred_logger = self._get_sacred_logger(log_params['sacred_params'])
        self.network_tester = self._get_network_tester(log_params['network_tester'])


    def _train_steps(self, sess):
        """Perform `self.val_interval` train steps.
        """
        train_ops = [self.tb_logger.train_summary_op,
                     self.sacred_logger.train_summary_op]

        summaries, it_nums = self.trainer.train(
            session=sess,
            num_steps=self.val_interval,
            additional_ops=train_ops
        )

        tb_summaries = [s[0] for s in summaries]
        sacred_summaries = [s[1] for s in summaries]
        return tb_summaries, sacred_summaries, it_nums

    def _val_logs(self, sess):
        """Create validation logs.
        """
        val_ops = [self.tb_logger.train_summary_op,
                   self.sacred_logger.train_summary_op]

        tb_summaries, sacred_summaries = sess.run(
            val_ops,
            feed_dict={self.model.is_training: False}
            )

        return tb_summaries, sacred_summaries

    def _train_its(self, sess):
        """Perform `self.val_interval` train steps and log validation metrics.
        """

        tb_summaries, sacred_summaries, it_nums = self._train_steps(sess)
        self.tb_logger.log_multiple(tb_summaries, it_nums)
        self.sacred_logger.log_multiple(sacred_summaries, it_nums=it_nums,
                                        _run=self._run)

        val_tb_summaries, val_sacred_summaries = self._val_logs(sess)
        self.tb_logger.log(val_tb_summaries, it_nums[-1], log_type='val')
        self.sacred_logger.log(val_sacred_summaries, it_num=it_nums[-1],
                               log_type='val', _run=self._run)

    def _get_sacred_logger(self, sacred_dict):
        return SacredLogger(self.evaluator, **sacred_dict)


class MNISTExperiment(NetworkExperiment):
    def __init__(self, experiment_params, model_params, dataset_params,
                 trainer_params, log_params):

        # Set experiment properties
        self.log_dir = self._get_logdir(experiment_params)
        self.continue_old = self._get_continue_old(experiment_params)
        self.name = self._get_name(experiment_params)
        self.val_interval = log_params['val_log_frequency']

        # Create TensorFlow objects
        self.dataset = MNISTDataset(name='MNIST')
        self.epoch_size = 40000
        self.model = self._get_model(model_params)
        self.trainer = self._get_trainer(trainer_params)
        self.evaluator = self._get_evaluator(log_params['evaluator'])
        self.tb_logger = self._get_tensorboard_logger(log_params['tb_params'])

        self.train(experiment_params['num_steps'])


