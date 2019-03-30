from utils.preprocess import preprocess
from utils.evaluate_metrics import calc_acc
from network.simple_cnn import forward_propagation, compute_cost
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import tensorflow as tf
import argparse
import yaml
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='runner for cnn')

    parser.add_argument('image_dir', help='directory path for training')
    parser.add_argument('output_dir', help='directory to output models and logs')
    parser.add_argument('model_path', help='model path with extension')
    parser.add_argument('--config_path', default='./config/config.yaml')

    # read commandline option
    args = parser.parse_args()
    train_image_dir = args.image_dir
    output_dir = args.output_dir
    model_path = args.model_path
    config_path = args.config_path
    os.makedirs(output_dir, exist_ok=True)

    # read config
    with open(config_path, 'r') as f:
        config = yaml.load(f)

    # preprocess image
    X, Y, class_index = preprocess(train_image_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    output_layer_num = len(class_index)

    # create placeholder
    (m, n_H0, n_W0, n_C0) = X.shape
    n_y = Y.shape[1]
    x = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    y = tf.placeholder(tf.float32, [None, n_y])

    # create compute graphs for sessions
    Z3 = forward_propagation(x, output_layer_num)
    cost = compute_cost(Z3, y)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    accuracy = calc_acc(Z3, y)

    # start session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in tqdm(range(config['train']['epochs'])):
            _, computed_cost = sess.run([optimizer, cost], feed_dict={x:X_train, y:y_train})

        test_accuracy = sess.run(accuracy, feed_dict={x:X_test, y:y_test})
        print("accuracy for test data: ", test_accuracy)

        # save model
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(output_dir, model_path))
        print("model is saved")
