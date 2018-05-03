import tensorflow as tf
from pathlib import Path
from ..model import model
from ..trainer import trainer
from ..utils import logger
from ..utils import HDFReader


class NetworkExperiment:
    def __init__(self, experiment_params, network_params, dataset_params, 
                 trainer_params, log_params=None):
        """
        experiment_parms={
            'log_dir': './',
            'name': 'test_experiment',
            'continue_old': False
        }

        model_params={
            'type': 'NeuralNet',
            'network_params: {
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
                'kwargs': {}
            }
        }

        log_params={
            'val_log_frequency': 1000,
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

        trainer_params={
            train_op: 'AdamOptimizer',
            train_op_kwargs: None,
            max_checkpoints: 10,
            save_step: 10,
            verbose: True
        }
        """
        if 'log_dir' in experiment_params:
            self.log_dir = Path(experiment_params['log_dir'])
        else:
            self.log_dir = Path('./')

        if 'continue_old' in experiment_params:
            self.continue_old = experiment_params['continue_old']
        else:
            self.continue_old = False

        if self.continue_old:
            self.name = experiment_params['name']
        else:
            self.name = self.create_name(experiment_params['name'])

    
    def create_name(self,    self.model,
            epoch_size=len(self.epoch_size),,
            log_dir
            **trainer_params
        ) name):
        """Enumerate the network name.
        """
        i = 0
        while _name_taken(f'{name}_{i:02d}'):
            i += 1
        return f'{name}_{i:02d}'
    
    def _name_taken(self, name):
        """Checks if the given name is taken.
        """
        return (self.log_dir/name).is_dir()

    def create_dataset(self, dataset_params):
        self.dataset = HDFReader(**dataset_params)
        self.epoch_length = len(self.dataset.train_data_reader)
        self.batch_size = batch_size

    def create_model(self, model_params):
        self.model_type = getattr(models, model_params['type'])
        self.model = model_type(
            input_var=self.dataset.data,
            true_out=self.dataset.target,
            is_training=self.is_training,
            **model_params['network_params']
        )
    
    def create_trainer(self, trainer_params):
        self.trainer = NetworkTrainer(
            self.model,
            epoch_size=len(self.epoch_size),,
            log_dir=self.log_dir,
            **trainer_params
        )

    def create_tensorboard_logger(self, tensorboard_params):
        self.tb_logger = TensorboardLogger(
            self.model,
            log_dir=self.log_dir,
            **tensorboard_params
        )

    