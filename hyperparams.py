import tensorflow as tf

class HyperParams:
    def __init__(self, learning_rate, beta=0.9, 
                batch_size=128, batch_norm=False, 
                num_epochs=100, initializer=tf.contrib.layers.xavier_initializer(),
                decay_rate = 0.96, decay_steps = 1000):
        self.learning_rate = learning_rate
        self.beta = beta
        self.batch_norm = batch_norm
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.initializer = initializer
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

net1_hyperparams = HyperParams(learning_rate=0.01, 
                               num_epochs=5, 
                               batch_size=10, 
                               decay_steps=100, 
                               decay_rate=0.9, 
                               initializer=tf.initializers.truncated_normal(mean=0.0, stddev=0.01))