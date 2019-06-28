import tensorflow as tf

def conv2d(x, W, b, strides=1, padding='SAME', batch_normalization=False, train=True):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    if(batch_normalization):
        x = tf.layers.batch_normalization(x, training=train, momentum=0.9)
    return tf.nn.relu(x)

def maxpool2d(x, k=2, strides=1, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], 
                          strides=[1, strides, strides, 1], padding=padding)

def dense_relu(x, w, b, batch_normalization=False):
    x = tf.matmul(x, w) + b
    if batch_normalization:
        x = tf.layers.batch_normalization(x)
    return tf.nn.relu(x)

def averagepool2d(x, k=2, strides=1, padding='SAME'):
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], 
                          strides=[1, strides, strides, 1], padding=padding)