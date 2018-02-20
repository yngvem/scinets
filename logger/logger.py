import tensorflow as tf
from pathlib import Path
from ..base_model import NeuralNet


class NetworkLogger:
    def __init__(self, session, network):
        self.session = session
        self.network = network

        self.log_dir = Path('./log')/network.name/'tensorboard'
        self.train_writer = tf.summary.FileWriter(str(logdir/'train'),
                                                  self.session.graph)
        self.test_writer = tf.summary.FileWriter(str(self.logdir/'test'))
    
    
        