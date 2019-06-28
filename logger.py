import tensorflow as tf
import os

class Logger:
    def __init__(self, sess, path):
        self.path = path
        self.sess = sess
        self.train_summary_writer = tf.summary.FileWriter(os.path.join(path, "train"), 
                                                          self.sess.graph)
        self.valid_summary_writer = tf.summary.FileWriter(os.path.join(path, "valid"))

    
    def summarize(self, step, train=True, summaries_dict=None):
        summary_writer = self.train_summary_writer if train else self.valid_summary_writer

        if summaries_dict is not None:
            summary_list = []
            for tag, value in summaries_dict.items():
                if tag not in self.summary_ops:
                    if len(value.shape) <= 1:
                        self.summary_placeholders[tag] = tf.placeholder('float32', value.shape, name=tag)
                    else:
                        self.summary_placeholders[tag] = tf.placeholder('float32', [None] + list(value.shape[1:]), name=tag)
                    if len(value.shape) <= 1:
                        self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
                    else:
                        self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag])

                summary_list.append(self.sess.run(self.summary_ops[tag], {self.summary_placeholders[tag]: value}))

            for summary in summary_list:
                summary_writer.add_summary(summary, step)
            summary_writer.flush()