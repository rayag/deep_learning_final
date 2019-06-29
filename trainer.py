from data import Data
from models.model import model_net  
import tensorflow as tf
import numpy as np
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
            print("Model trained {}".format(self.model.config.name))
            print("Starter learning rate {}".format(self.model.hyperparams.learning_rate))
            print("Batch size {}".format(self.model.hyperparams.batch_size))
            print("Initializer {}".format(self.model.hyperparams.initializer))

        for cur_epoch in range(self.model.hyperparams.num_epochs):
            losses = np.zeros(shape=(num_iterations_per_epoch))
            accuracies = np.zeros(shape=(num_iterations_per_epoch))
            for i in range(num_iterations_per_epoch):
                loss, accuracy = self.train_step()
                losses[i] = loss
                accuracies[i] = accuracy
                cur_step = self.model.global_step_tensor.eval(self.session)
                sum_dict = {
                    'loss': loss,
                    'accuracy': accuracy
                }
                # Add entry to tensorboard graphs
                self.logger.summarize(cur_step, train=True, summaries_dict=sum_dict)
            # Evaluate validation accuracy and loss
            val_loss, val_acc = self.eval_model()
            val_sum_dict = {
                'valid_loss': val_loss,
                'valid_accuracy': val_acc
            }
            # Add scalar summary
            self.logger.summarize(cur_step, train=True, summaries_dict=sum_dict)
            if self.verbose:
                print('Average Minibatch loss on epoch {}: {:0.6f}'.format(cur_epoch, np.mean(losses)))
                print('Average Minibatch accuracy: {:10.6f}'.format(np.mean(accuracies)))
                print('Validation accuracy: {:10.6f}, Validation loss {:10.6f}'.format(val_acc, val_loss))           
            
        # Save model
        self.model.save(self.session)          


    def train_step(self):
        # Generate next batch
        batch_x, batch_y = next(self.data.next_batch(self.model.hyperparams.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.train: True}
        _, l, accuracy = self.session.run([self.model.optimizer, self.model.loss, self.model.train_accuracy],
                                            feed_dict=feed_dict)
        return l, accuracy
   
    def eval_model(self, validation=True):
        x = self.data.valid_dataset if validation else self.data.test_dataset
        y = self.data.valid_labels if validation else self.data.test_labels
        size = self.data.valid_dataset.shape[0]
        losses = np.zeros(shape=(size))
        accuracies = np.zeros(shape=(size))
        for i in range(size):
            batch = x[i]
            batch = np.expand_dims(batch, axis=0)
            labels = y[i]
            labels = np.expand_dims(labels, axis=0)
            feed_dict = {self.model.x: batch, 
                         self.model.y: labels, 
                         self.model.train: False}
            _, l, acc = self.session.run([self.model.optimizer, self.model.loss, self.model.train_accuracy],
                                        feed_dict=feed_dict)
            losses[i] = l
            accuracies[i] = acc
        return np.mean(losses), np.mean(accuracies)
        