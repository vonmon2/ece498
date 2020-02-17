import tensorflow as tf

def function(a, X, b, y):
    Xt = tf.transpose(X)
    bt = tf.transpose(b)
    
#   Z = (a(X^t*X) + b^t*X)

    Z = tf.matmul(Xt, X)
    Z = tf.scalar_mul(a, Z)
    
    tmp = tf.matmul(bt, X)
    
    Z = tf.add(Z, tmp)
    Z = tf.subtract(Z, y) 
    
    sq = tf.constant(2, dtype = tf.float32)
    
    ans = tf.pow(Z, sq)
    return ans