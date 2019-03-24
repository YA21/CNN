from utils.preprocess import preprocess

import tensorflow as tf

def conv2d(x, W):
    Z = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")
    return tf.nn.relu(Z)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")

def compute_cost(X, Y, output_layer_num):
    # extract features
    (m, n_H0, n_W0, n_C0) = X.shape
    n_y = Y_.shape[1]

    # create placeholder
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, [None, n_y])

    # define weights
    W1 = tf.get_variable("W1", [3, 3, 3, 32], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    # 2 conv and maxpool layer
    A1 = conv2d(X, W1)
    A1 = max_pool_2x2(A1)

    A2 = conv2d(A1, W2)
    A2 = max_pool_2x2(A2)

    # flattern
    A2 = tf.contrib.layers.flatten(A2)

    # fully connected layers
    Z3 = tf.contrib.layers.fully_connected(P2, output_layer_num)

    # compute cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))
    
    return cost