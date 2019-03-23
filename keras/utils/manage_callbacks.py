from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from datetime import datetime
import os

def manage_callbacks(config, output_dir):

    callbacks = []

    if config['train']['checkpoint_interval'] != -1:
            checkpoint = ModelCheckpoint(os.path.join(output_dir, 'checkpoint.h5'),
                                        monitor='val_loss',
                                        verbose=1,
                                        period=config['train']['checkpoint_interval'])
            callbacks.append(checkpoint)

    if config['train']['early_stopping'] is True:
            callbacks.append(EarlyStopping(monitor='val_loss', patience=0))

    if config['train']['tensorboard'] is True:
            log_dir = os.path.join('./tflog', datetime.today().strftime('%Y%m%d_%H%M%S'))
            tensorboard = TensorBoard(log_dir=log_dir, write_images=True)
            callbacks.append(tensorboard)
    
    return callbacks