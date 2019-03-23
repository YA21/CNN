from network.simple_cnn import cnn_network
from utils.preprocess import preprocess
from utils.manage_callbacks import manage_callbacks
from sklearn.model_selection import train_test_split

import argparse
import os
import yaml

import hog

if __name__ == '__main__':
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

    # read_config
    with open(config_path, 'r') as f:
        config = yaml.load(f)

    # preprocess image
    X, Y, class_index = preprocess(train_image_dir)
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2)

    # define network
    model = cnn_network(input_shape=X_train.shape[1:],
                        output_layer_num=y_train.shape[1],
                        metrics=config['model']['metrics'],
                        loss=config['model']['loss'],
                        optimizer=config['model']['optimizer'])

    # add callbacks
    callbacks = manage_callbacks(config, output_dir)

    # train model
    model.fit(X_train, y_train,
              batch_size=config['train']['batch_size'],
              epochs=config['train']['epochs'],
              verbose=1,
              validation_data=(X_valid, y_valid),
              callbacks=callbacks)

    # evaluate model
    scores = model.evaluate(X_valid, y_valid, verbose=1)
    print('valid loss:', scores[0])
    print('valid accuracy', scores[1])
    print('model summary:')
    model.summary()
    
    # save model
    model.save(os.path.join(output_dir, model_path))
