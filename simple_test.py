import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model import NeuralNet
from utils import TensorboardLogger, HDFReader
from trainer import NetworkTrainer
import sys


def get_channel(image, channel):
    return image[..., channel, tf.newaxis]


def get_pet(image):
    with tf.variable_scope('get_pet'):
        return get_channel(image, 1)


def get_ct(image):
    with tf.variable_scope('get_ct'):
        return get_channel(image, 0)


if __name__ == '__main__':
    dataset = HDFReader(
        data_path='/home/yngve/dataset_extraction/val_split_2d.h5',
        batch_size=[5, 5, 1],
        val_group='val'
    )
    is_training = dataset.is_training
    x = dataset.data
    print('Getting shape')
    target_shape = dataset.target.get_shape().as_list()
    print(f'Target shape: {target_shape}')
    target_shape[0] = -1
    true_y = tf.reshape(dataset.target, (*target_shape, 1))
    target_shape = true_y.get_shape().as_list()
    print(f'Target shape: {target_shape}')



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

    name = sys.argv[1] if len(sys.argv) > 1 else 'test_net'
    network = NeuralNet(x, architecture, verbose=True, name=name, is_training=dataset.is_training)
    network.set_loss(
        true_out=true_y,
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
        'out': [
            {
                'log_name': 'Logit output',
                'log_type': 'image',
                'kwargs': {'max_outputs':1}
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
                           'transform': get_ct}
            },
            {
                'log_name': 'PET',
                'log_type': 'image',
                'kwargs': {'max_outputs': 1,
                           'transform': get_pet}
            }
        ]
    }

    logger = TensorboardLogger(network, log_dict)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        logger.init_file_writers(sess)
        
        # Train the model
        for i in range(100):
            summaries, steps = trainer.train(sess, 10, additional_ops=[logger.train_summary_op])
            summaries = [summary[0] for summary in summaries]
            logger.log_multiple(summaries, steps)
