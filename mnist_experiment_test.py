import tensorflow as tf
from scinets.model import NeuralNet
from scinets.utils import TensorboardLogger, MNISTDataset, BinaryClassificationEvaluator
from scinets.trainer import NetworkTrainer
from tqdm import trange
import sys



if __name__ == '__main__':
    dataset = MNISTDataset()
    is_training = dataset.is_training
    x = dataset.data

    architecture = [
        {
            'layer': 'ResnetConv2D',
            'scope': 'resnet1',
            'layer_params': {
                'out_size': 1,
                'k_size': 3,
            },
            'normalization': {
                'operator': 'batch_normalization'
            },
            'activation': {
                'operator': 'relu'
            }
        },
        {
            'layer': 'ResnetConv2D',
            'scope': 'resnet2',
            'layer_params': {
                'out_size': 1,
                'k_size': 3,
                'strides': 2
            },
            'normalization': {
                'operator': 'batch_normalization'
            },
            'activation': {
                'operator': 'relu'
            }
        },
        {
            'layer': 'ResnetConv2D',
            'scope': 'resnet3',
            'layer_params': {
                'out_size': 1,
                'k_size': 3,
            },
            'normalization': {
                'operator': 'batch_normalization'
            },
            'activation': {
                'operator': 'relu'
            }
        },
        {
            'layer': 'ResnetConv2D',
            'scope': 'resnet4',
            'layer_params': {
                'out_size': 1,
                'k_size': 3,
                'strides': 2
            },
            'normalization': {
                'operator': 'batch_normalization'
            },
            'activation': {
                'operator': 'relu'
            }
        },
        {
            'layer': 'ResnetConv2D',
            'scope': 'resnet5',
            'layer_params': {
                'out_size': 10,
                'k_size': 3,
            },
            'normalization': {
                'operator': 'batch_normalization'
            },
            'activation': {
                'operator': 'linear'
            }
        },
        {
            'layer': 'GlobalAveragePool',
            'scope': 'global_average'
        }
    ]

    name = sys.argv[1] if len(sys.argv) > 1 else 'testetest2'
    network = NeuralNet(x, architecture, verbose=True, name=name,
                        is_training=dataset.is_training)
    network.set_loss(
        true_out=dataset.target,
        loss_function='sigmoid_cross_entropy_with_logits'
    )
    trainer = NetworkTrainer(network, epoch_size=len(dataset.train_data_reader))

    log_dicts = [
            {
                'log_name': 'Log loss',
                'log_var': 'loss',
                'log_type': 'log_scalar'
            },
            {
                'log_name': 'Loss',
                'log_var': 'loss',
                'log_type': 'scalar'
            },
            {
                'log_name': 'Probability_map',
                'log_var': 'probabilities',
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
            }
    ]

    evaluator = BinaryClassificationEvaluator(network)
    logger = TensorboardLogger(evaluator, log_dicts)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        logger.init_file_writers(sess)

        # Train the model
        for i in trange(1000):
            train_summaries, steps = trainer.train(
                sess,
                10,
                additional_ops=[logger.train_summary_op]
            )
            train_summaries = [s[0] for s in train_summaries]
            logger.log_multiple(train_summaries, steps)
            val_summaries = sess.run(logger.val_summary_op,
                                     feed_dict={is_training:False})
            logger.log(val_summaries, steps[-1], log_type='val')
