import tensorflow as tf

class HyperParams:
    def __init__(self, learning_rate, beta=0.9, 
                batch_size=128, batch_norm=False, 
                num_epochs=100, initializer=tf.contrib.layers.xavier_initializer()):
        self.learning_rate = learning_rate
        self.beta = beta
        self.batch_norm = batch_norm
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.initializer = initializer

net1_hyperparams = HyperParams(learning_rate=0.01, num_epochs=1)