import tensorflow as tf
from datetime import datetime

class Logger:
    def __init__(self, sess, path):
        self.path = path
        self.sess = sess
        self.train_summary_writer = tf.summary.FileWriter(path + "/train/train-"+ datetime.now().strftime("%Y-%m-%d-%H%M%S"))
        self.valid_summary_writer = tf.summary.FileWriter(path + "/valid/valid-"+ datetime.now().strftime("%Y-%m-%d-%H%M%S"))

    
    def summarize(self, step, train=True, summaries_dict=None):
        summary_writer = self.train_summary_writer if train else self.valid_summary_writer
        self.summary_list = []
        for tag,value in summaries_dict.items():
            self.summary_list.append(tf.summary.scalar(tag, value))
        
        for s in self.summary_list:
            sum = self.sess.run(s)
            summary_writer.add_summary(sum, step)
        summary_writer.flush()