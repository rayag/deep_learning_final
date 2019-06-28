import tensorflow as tf
from utils import dense_relu, conv2d, maxpool2d
from config import NetworkConfig
from hyperparams import HyperParams
import numpy as np

width = 48
height = 48
num_channels = 1
num_classes = 7

class model_net():
    def __init__(self, config, hyperparams):
        self.config = config
        self.hyperparams = hyperparams
        self.init_global_step()
        self.build_model()
        self.init_saver()
    

    def build_model(self):
        print("Building model")
        self.train = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=[None, 48, 48, 1])
        self.y = tf.placeholder(tf.float32, shape=[None, 7])
        
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            self.weights = {
                'conv1' : tf.get_variable('w_conv1', shape=(3,3,1,64), initializer=tf.contrib.layers.xavier_initializer()),
                'conv2' : tf.get_variable('w_conv2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()),
                'conv3' : tf.get_variable('w_conv3', shape=(3,3,128,256), initializer=tf.contrib.layers.xavier_initializer()),
                'fc1' : tf.get_variable('w_fc1', shape=(9216, 4096), initializer=tf.contrib.layers.xavier_initializer()),
                'fc2' : tf.get_variable('w_fc2', shape=(4096, 1024), initializer=tf.contrib.layers.xavier_initializer()),
                'fc3' : tf.get_variable('w_fc3', shape=(1024, 256), initializer=tf.contrib.layers.xavier_initializer()),
                'fc4' : tf.get_variable('w_fc4', shape=(256, 7), initializer=tf.contrib.layers.xavier_initializer())
            }
            self.biases = {
                'conv1' : tf.get_variable('b_conv1', shape=(64), initializer=tf.zeros_initializer()),
                'conv2' : tf.get_variable('b_conv2', shape=(128), initializer=tf.zeros_initializer()),
                'conv3' : tf.get_variable('b_conv3', shape=(256), initializer=tf.zeros_initializer()),
                'fc1' : tf.get_variable('b_fc1', shape=(4096), initializer=tf.zeros_initializer()),
                'fc2' : tf.get_variable('b_fc2', shape=(1024), initializer=tf.zeros_initializer()),
                'fc3' : tf.get_variable('b_fc3', shape=(256), initializer=tf.zeros_initializer()),
                'fc4' : tf.get_variable('b_fc4', shape=(7), initializer=tf.zeros_initializer())
            }


            conv1 = conv2d(self.x, self.weights['conv1'], self.biases['conv1'], strides=1)
            conv1 = maxpool2d(conv1, k=3, strides=2)
            
            conv2 = conv2d(conv1, self.weights['conv2'], self.biases['conv2'], strides=1)
            conv2 = maxpool2d(conv2, k=3, strides=2)

            conv3 = conv2d(conv2, self.weights['conv3'], self.biases['conv3'], strides=1)
            conv3 = maxpool2d(conv3, k=3, strides=2)

            conv3_reshape = tf.reshape(conv3, [-1, self.weights['fc1'].get_shape().as_list()[0]])
            fc1 = dense_relu(conv3_reshape, self.weights['fc1'], self.biases['fc1'])

            fc2 = dense_relu(fc1, self.weights['fc2'], self.biases['fc2'])

            fc3 = dense_relu(fc2, self.weights['fc3'], self.biases['fc3'])

            logits = dense_relu(fc3, self.weights['fc4'], self.biases['fc4'])
            self.prediction = tf.argmax(tf.nn.softmax(logits))
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits))
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(self.loss, global_step=self.global_step_tensor)
                correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.y, 1))
                self.train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    def init_saver(self):
        self.saver = tf.train.Saver()

    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, self.config.model_path)
        print("Model saved")

    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint('net1/')
        print(self.config.model_path)
        print(latest_checkpoint)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")

    def predict(self, sess, batch):
        pred = sess.run(self.prediction, feed_dict={self.x: batch, self.train:False})
        pred = np.argmax(pred)
        return pred


