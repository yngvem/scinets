import tensorflow as tf
from model import NeuralNet
from utils import TensorboardLogger, HDFReader, BinaryClassificationEvaluator
from trainer import NetworkTrainer
import sys



if __name__ == '__main__':
    dataset = HDFReader(
        data_path='/home/yngve/dataset_extraction/val_split_2d.h5',
        batch_size=[5, 5, 1],
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

    name = sys.argv[1] if len(sys.argv) > 1 else 'test_log'
    network = NeuralNet(x, architecture, verbose=True, name=name,
                        is_training=dataset.is_training)
    network.set_loss(
        true_out=dataset.target,
        loss_function='sigmoid_cross_entropy_with_logits'
    )
    trainer = NetworkTrainer(network, epoch_size=len(dataset.train_data_reader))
    
    log_dict = {
        'loss': [
            {
                'log_name': 'Log loss',
                'log_type': 'log_scalar'
            },
            {
                'log_name': 'Loss',
                'log_type': 'scalar'
            }
        ],
        'probabilities': [
            {
                'log_name': 'Probability_map',
                'log_type': 'image',
                'kwargs': {'max_outputs':1}
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
                'kwargs': {'max_outputs':1}
            }
        ],
        'input': [
            {
                'log_name': 'CT',
                'log_type': 'image',
                'kwargs': {'max_outputs': 1,
                           'channel': 0}
            },
            {
                'log_name': 'PET',
                'log_type': 'image',
                'kwargs': {'max_outputs': 1,
                           'channel': 1}
            }
        ]
    }

    evaluator = BinaryClassificationEvaluator(network)
    logger = TensorboardLogger(evaluator, log_dict)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        logger.init_file_writers(sess)
        
        # Train the model
        for i in range(100):
            summaries, steps = trainer.train(sess, 10, additional_ops=[logger.train_summary_op])
            summaries = [summary[0] for summary in summaries]
            logger.log_multiple(summaries, steps)
