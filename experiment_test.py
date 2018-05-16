import tensorflow as tf
from scinets.model import NeuralNet
from scinets.utils import TensorboardLogger, HDFReader, BinaryClassificationEvaluator
from scinets.utils.experiment import NetworkExperiment
from scinets.trainer import NetworkTrainer
from tqdm import trange
import sys



if __name__ == '__main__':
    experiment_params={
        'log_dir': './logs/',
        'name': 'test_experiment',
        'continue_old': False,
        'num_steps': 1000
    }

    dataset_params = {
        'data_path': '/home/yngve/dataset_extraction/val_split_2d.h5',
        'batch_size': [64, 64, 1],
        'val_group': 'val'
    }
    model_params={
        'type': 'NeuralNet',
        'network_params': {
            'loss_function': 'sigmoid_cross_entropy_with_logits',
            'loss_kwargs': {},
            'architecture': [
                {
                    'layer': 'conv2d',
                    'scope': 'conv1',
                    'out_size': 1,
                    'k_size': 5,
                    'batch_norm': True,
                    'activation': 'linear',
                }
            ],
            'verbose': True,
        }
    }

    trainer_params = {}

    log_params={
        'val_log_frequency': 10,
        'evaluator': 'BinaryClassificationEvaluator',
        'sacred_params': {
            'log_dict': {
                'loss': 'Loss'
            }
        }
        'tb_params': {
            'log_dict': {
                'loss': [
                    {
                        'log_name': 'Loss',
                        'log_type': 'scalar'
                    }
                ],
                'probabilities': [
                    {
                        'log_name': 'Probability_map',
                        'log_type': 'image',
                        'log_kwargs': {'max_outputs':1}
                    }
                ],
                'accuracy': [
                    {
                        'log_name': 'Accuracy',
                        'log_type': 'scalar'
                    }
                ],
                'dice': [
                    {
                        'log_name': 'Dice',
                        'log_type': 'scalar'
                    }
                ],
                'true_out': [
                    {
                        'log_name': 'Mask',
                        'log_type': 'image',
                        'log_kwargs': {'max_outputs':1}
                    }
                ],
                'input': [
                    {
                        'log_name': 'CT',
                        'log_type': 'image',
                        'log_kwargs': {'max_outputs': 1,
                                'channel': 0}
                    },
                    {
                        'log_name': 'PET',
                        'log_type': 'image',
                        'log_kwargs': {'max_outputs': 1,
                                'channel': 1}
                    }
                ]
            }
        }
    }

    experiment = NetworkExperiment(
        experiment_params=experiment_params,
        model_params=model_params,
        dataset_params=dataset_params,
        trainer_params=trainer_params,
        log_params=log_params
    )

    experiment.train(1000, 10)
