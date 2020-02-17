import tensorflow as tf

def function(session, loss):
        return loss.eval(session=session)[0][0]