import tensorflow as tf
import sacred
from scinets.utils.experiment import SacredExperiment, NetworkExperiment




if __name__ == '__main__':
    ex = sacred.Experiment()
    ex.observers.append(
        sacred.observers.MongoObserver.create(
            db_name='sacred',
            port=27017,
            url='yngvem.no',
            username='sacredWriter',
            password='LetUsUseSacredForLogging',
            )
    )

    @ex.config
    def cfg():
        experiment_params = {
            'log_dir': './logs/',
            'name': 'test_experiment',
            'continue_old': False,
            'num_steps': 10000
        }
        dataset_params = {
            'data_path': '/home/yngve/dataset_extraction/val_split_2d.h5',
            'batch_size': [256, 256, 1],
            'val_group': 'val'
        }
        model_params = {
            'type': 'UNet',
            'network_params': {
                'loss_function': 'sigmoid_cross_entropy_with_logits',
                'loss_kwargs': {},
                'skip_connections': [
                    ('conv1', 'linear_upsample_2'),
                    ('conv2', 'linear_upsample_1')
                ],
                'architecture': [
                        {
                            'layer': 'Conv2D',
                            'scope': 'conv1',
                            'layer_params': {
                                'out_size': 8,
                                'k_size': 5,
                                'strides': 2
                            },
                            'normalization': {
                                'operator': 'batch_normalization'
                            },
                            'activation': {
                                'operator': 'relu'
                            },
                            'initializer': {
                                'operator': 'he_normal'
                            }
                        },
                        {
                            'layer': 'Conv2D',
                            'scope': 'conv2',
                            'layer_params': {
                                'out_size': 16,
                                'k_size': 5,
                                'strides': 2
                            },
                            'normalization': {
                                'operator': 'batch_normalization'
                            },
                            'activation': {
                                'operator': 'relu'
                            },
                            'initializer': {
                                'operator': 'he_normal'
                            }
                        },
                        {
                            'layer': 'Conv2D',
                            'scope': 'conv3',
                            'layer_params': {
                                'out_size': 16,
                                'k_size': 5,
                                'strides': 2
                            },
                            'normalization': {
                                'operator': 'batch_normalization'
                            },
                            'activation': {
                                'operator': 'relu'
                            },
                            'initializer': {
                                'operator': 'he_normal'
                            }
                        },
                        {
                            'layer': 'Conv2D',
                            'scope': 'conv4',
                            'layer_params': {
                                'out_size': 32,
                                'k_size': 5,
                            },
                            'normalization': {
                                'operator': 'batch_normalization'
                            },
                            'activation': {
                                'operator': 'relu'
                            },
                            'initializer': {
                                'operator': 'he_normal'
                            }
                        },
                        {
                            'layer': 'LinearInterpolate',
                            'scope': 'linear_upsample_1',
                            'layer_params': {'rate': 2}
                        },
                        {
                            'layer': 'Conv2D',
                            'scope': 'conv5',
                            'layer_params': {
                                'out_size': 32,
                                'k_size': 5,
                            },
                            'normalization': {
                                'operator': 'batch_normalization'
                            },
                            'activation': {
                                'operator': 'relu'
                            },
                            'initializer': {
                                'operator': 'he_normal'
                            }
                        },
                        {
                            'layer': 'LinearInterpolate',
                            'scope': 'linear_upsample_2',
                            'layer_params': {'rate': 2}
                        },
                        {
                            'layer': 'Conv2D',
                            'scope': 'conv6',
                            'layer_params': {
                                'out_size': 64,
                                'k_size': 5,
                            },
                            'normalization': {
                                'operator': 'batch_normalization'
                            },
                            'activation': {
                                'operator': 'relu'
                            },
                            'initializer': {
                                'operator': 'he_normal'
                            }
                        },
                ],
                'verbose': True,
            }
        }

        trainer_params = {
            'train_op': 'GradientDescentOptimizer',
            'train_op_kwargs': {'learning_rate': 0.00003}
        }

        log_params={
            'val_log_frequency': 10,
            'evaluator': 'BinaryClassificationEvaluator',
            'tb_params': {
                'log_dicts': [
                    {
                        'log_name': 'Loss',
                        'log_var': 'loss',
                        'log_type': 'scalar'
                    },
                    {
                        'log_name': 'Probability_map',
                        'log_var':'probabilities',
                        'log_type': 'image',
                        'log_kwargs': {'max_outputs':1}
                    },
                    {
                        'log_name': 'Accuracy',
                        'log_var': 'accuracy',
                        'log_type': 'scalar'
                    },
                    {
                        'log_name': 'Dice',
                        'log_var': 'dice',
                        'log_type': 'scalar'
                    },
                    {
                        'log_name': 'Mask',
                        'log_var': 'true_out',
                        'log_type': 'image',
                        'log_kwargs': {'max_outputs':1}
                    },
                    {
                        'log_name': 'CT',
                        'log_var': 'input',
                        'log_type': 'image',
                        'log_kwargs': {'max_outputs': 1,
                                    'channel': 0}
                    },
                    {
                        'log_name': 'PET',
                        'log_var': 'input',
                        'log_type': 'image',
                        'log_kwargs': {'max_outputs': 1,
                                    'channel': 1}
                    },
                    {
                        'log_name': 'Probability_map',
                        'log_var':'probabilities',
                        'log_type': 'histogram',
                    }
                ]
            },
            'sacred_params': {
                'log_dicts': [
                    {
                        'log_name': 'Loss',
                        'log_var': 'loss',
                    },
                    {
                        'log_name': 'Accuracy',
                        'log_var': 'accuracy',
                    },
                    {
                        'log_name': 'Dice',
                        'log_var': 'dice',
                    },
                    {
                        'log_name': 'Precision',
                        'log_var': 'precision',
                    },
                    {
                        'log_name': 'Recall',
                        'log_var': 'recall',
                    },
                    {
                        'log_name': 'True positives',
                        'log_var': 'true_positives',
                    },
                    {
                        'log_name': 'True negatives',
                        'log_var': 'true_negatives',
                    }
                ]
            }
        }

    @ex.automain
    def main(_run, experiment_params, model_params, dataset_params,
             trainer_params, log_params):
        experiment = SacredExperiment(
            _run=_run,
            experiment_params=experiment_params,
            model_params=model_params,
            dataset_params=dataset_params,
            trainer_params=trainer_params,
            log_params=log_params,
        )
