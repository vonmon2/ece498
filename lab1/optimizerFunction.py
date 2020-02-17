import tensorflow as tf

def function(loss, lr):
    n = (tf.compat.v1.train.AdamOptimizer(lr))
    return n.minimize(loss)