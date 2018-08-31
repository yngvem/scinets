import tensorflow as tf
import sacred
from pprint import pprint
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
            'name': 'espen_test',
            'continue_old': False,
            'verbose': True
        }
        dataset_params = {
            'data_path': '/home/yngve/Dropbox/dataset_extraction/val_split_2d.h5',
            'batch_size': [32, 256, 1],
            'val_group': 'val'
        }
        model_params = {
            'type': 'UNet',
            'network_params': {
                'loss_function': 'sigmoid_cross_entropy_with_logits',
                'loss_kwargs': {},
                'skip_connections': [
                    ('input', 'linear_upsample_2'),
                    ('conv1', 'linear_upsample_1')
                ],
                'architecture': [
                        {
                            'layer': 'Conv2D',
                            'scope': 'conv1',
                            'layer_params': {
                                'out_size': 64,
                                'k_size': 7,
                                'strides': 2
                            },
                            'normalizer': {
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
                            'layer': 'ResnetConv2D',
                            'scope': 'conv2',
                            'layer_params': {
                                'out_size': 16,
                                'k_size': 5,
                                'strides': 2
                            },
                            'normalizer': {
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
                            'layer': 'ResnetConv2D',
                            'scope': 'conv3',
                            'layer_params': {
                                'out_size': 16,
                                'k_size': 5
                            },
                            'normalizer': {
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
                            'layer': 'ResnetConv2D',
                            'scope': 'conv4',
                            'layer_params': {
                                'out_size': 32,
                                'k_size': 5,
                            },
                            'normalizer': {
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
                            'layer': 'ResnetConv2D',
                            'scope': 'conv5',
                            'layer_params': {
                                'out_size': 32,
                                'k_size': 5,
                            },
                            'normalizer': {
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
                            'layer': 'ResnetConv2D',
                            'scope': 'conv6',
                            'layer_params': {
                                'out_size': 64,
                                'k_size': 5,
                            },
                            'normalizer': {
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
                            'layer': 'ResnetConv2D',
                            'scope': 'conv7',
                            'layer_params': {
                                'out_size': 1,
                                'k_size': 5,
                            },
                            'normalizer': {
                                'operator': 'batch_normalization'
                            },
                            'activation': {
                                'operator': 'linear'
                            },
                            'initializer': {
                                'operator': 'he_normal'
                            }
                        }
                ],
            }
        }

        trainer_params = {
            'train_op': {
                'operator': 'AdamOptimizer',
                'arguments': {
                    'learning_rate': 0.0001
                }
            },
            # 'learning_rate_op': {
            #     'operator': 'cosine_decay_restarts',
            #     'arguments': {
            #         'learning_rate': 0.0000001,
            #         'first_decay_steps': 100,
            #         't_mul': 2,
            #         'm_mul': 1,
            #         'alpha': 0.01
            #     }
            # }
        }

        log_params={
            'val_log_frequency': 100,
            'evaluator': 'BinaryClassificationEvaluator',
            'tb_params': {
                'log_all_params': False,
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
                    },
                    {
                        'log_name': 'Precision',
                        'log_var': 'precision',
                        'log_type': 'scalar'
                    },
                    {
                        'log_name': 'Recall',
                        'log_var': 'recall',
                        'log_type': 'scalar'
                    },
                    {
                        'log_name': 'True positives',
                        'log_var': 'true_positives',
                        'log_type': 'scalar'
                    },
                    {
                        'log_name': 'True negatives',
                        'log_var': 'true_negatives',
                        'log_type': 'scalar'
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
        experiment.train(50000)
