import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten

def cnn_network(input_shape, output_layer_num, metrics="accuracy", loss="categorical_crossentropy", optimizer="SGD"):
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=64, kernel_size=(3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(output_layer_num))
    model.add(Activation("softmax"))

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model