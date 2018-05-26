import tensorflow as tf
from scinets.model import NeuralNet
from scinets.utils import TensorboardLogger, HDFReader, BinaryClassificationEvaluator
from scinets.trainer import NetworkTrainer
from tqdm import trange
import sys



if __name__ == '__main__':
    dataset = HDFReader(
        data_path='/home/yngve/dataset_extraction/val_split_2d.h5',
        batch_size=[64, 64, 1],
        val_group='val'
    )
    is_training = dataset.is_training
    x = dataset.data

    architecture = [
            {
                'layer': 'conv2d',
                'scope': 'conv1',
                'out_size': 1,
                'k_size': 5,
                'batch_norm': True,
                'activation': 'linear',
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
