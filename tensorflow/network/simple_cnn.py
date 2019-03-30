import tensorflow as tf

def conv2d(x, W):
    Z = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")
    return tf.nn.relu(Z)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")

def forward_propagation(x, output_layer_num):
    # define weights
    W1 = tf.get_variable("W1", [3, 3, 3, 32], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    # 2 conv and maxpool layer
    A1 = conv2d(x, W1)
    P1 = max_pool_2x2(A1)

    A2 = conv2d(P1, W2)
    P2 = max_pool_2x2(A2)

    # flatten
    P2 = tf.contrib.layers.flatten(P2)

    # fully connected layers
    Z3 = tf.contrib.layers.fully_connected(P2, output_layer_num)

    return Z3

def compute_cost(Z_L, y):
    # compute cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = Z_L, labels = y))
    
    return cost