import keras
from keras.models import Sequential
from keras.layers import Dense
from typing import Tuple
import numpy as np
import tensorflow as tf


np.random.seed(1)


def load_data(fd:str) -> Tuple[np.array, np.array]:
    # Load data
    with open(fd, 'r') as f:
        data = f.readlines()
    # load csv data into respective arrays
    labels, images = [], []
    for i, s in enumerate(data):
        # seperate elements by comma and remove \n char
        s = s.split(',')
        s[784] = s[784].replace('\n', '')
        # append image of number to array images
        images.append(s[1:785])
        # create label/Y arrays
        labels.append([0.01 for x in range(10)])
        labels[i][int(s[0])] = 0.99
    # convert arrays to numpy array
    x = np.asarray(images, dtype='int16')
    y = np.asarray(labels, dtype='float32')
    return x, y


def build_model() -> keras.Sequential:
    # model layers and build
    model = Sequential()
    model.add(Dense(units=784, activation='sigmoid', input_dim=784))
    model.add(Dense(units=500, activation='sigmoid'))
    model.add(Dense(units=10, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam',
                  metrics=['accuracy'])
    # store checkpoints of the model's weights
    return model


def trained_model() -> keras.Sequential:
    # load training data & build model
    train_x, train_y = load_data("mnist_data/mnist_train.csv")
    model = build_model()
    # store model weights as checkpoints
    checkpoint_path = "model_weights_test/cp.cpkt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1
    )
    # fit model
    model.fit(train_x, train_y, batch_size=16,
              epochs=5, verbose=1, callbacks=[cp_callback])
    keras.models.save_model(model=model, filepath="model_test")
    return model


def test_model(model) -> float:
    test_x, test_y = load_data("mnist_data/mnist_test.csv")
    rv = round(model.evaluate(test_x, test_y)[1] * 100, 4)
    print(f"\nTest Accuracy: {rv}%")
    return rv


if __name__ == '__main__':
    model = trained_model()
    test_model(model)
