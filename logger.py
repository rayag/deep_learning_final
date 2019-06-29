import tensorflow as tf
import os

class Logger:
    def __init__(self, sess, path):
        self.path = path
        self.sess = sess
        self.train_summary_writer = tf.summary.FileWriter(os.path.join(path, "train"))
        self.valid_summary_writer = tf.summary.FileWriter(os.path.join(path, "valid"))

    
    def summarize(self, step, train=True, summaries_dict=None):
        summary_writer = self.train_summary_writer if train else self.valid_summary_writer
        for tag,value in summaries_dict.items():
            self.summary = tf.summary.scalar(tag, value)
            sum = self.sess.run(self.summary)
            summary_writer.add_summary(sum, step)
        summary_writer.flush()