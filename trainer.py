from data import Data
from models.model import model_net  
import tensorflow as tf

'''
Class performing training of a model
'''
class Trainer:
    def __init__(self, sess, model, logger, data, verbose=True):
        self.session = sess
        self.model = model
        self.data = data
        self.logger = logger
        self.verbose = verbose
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.session.run(self.init)

    def train(self):
        num_iterations_per_epoch = int(self.data.train_size()/self.model.hyperparams.batch_size) + 1

        if self.verbose:
            print("Training started")
            print("Number of epochs {}".format(self.model.hyperparams.num_epochs))
            print("Number of steps per epoch {}".format(num_iterations_per_epoch))

        for cur_epoch in range(self.model.hyperparams.num_epochs):
            for _ in range(num_iterations_per_epoch):
                loss, accuracy = self.train_step()
                cur_step = self.model.global_step_tensor.eval(self.session)
                sum_dict = {
                    'loss': loss,
                    'accuracy': accuracy
                }
                # Add entry to tensorboard graphs
                self.logger.summarize(cur_step, sum_dict)
            if self.verbose:
                print('Minibatch loss on epoch {}: {}'.format(cur_epoch, loss))
                print('Minibatch accuracy: {}'.format(accuracy))
        # Save model
        self.model.save(self.session)          


    def train_step(self):
        # Generate next batch
        batch_x, batch_y = next(self.data.next_batch(self.model.hyperparams.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.train: True}
        _, l, accuracy = self.session.run([self.model.optimizer, self.model.loss, self.model.train_accuracy],
                                            feed_dict=feed_dict)
        return l, accuracy
