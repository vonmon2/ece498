import numpy as np
import tensorflow as tf

def function(shape):
    X = tf.Variable(np.random.random(shape), dtype=tf.float32) #init X
    return X