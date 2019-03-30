import tensorflow as tf

def calc_acc(Z_L, y):
    
    correct_prediction = tf.equal(tf.argmax(Z_L, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    return accuracy