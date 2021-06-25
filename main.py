from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import tensorflow as tf
import numpy as np

np.random.seed(1)

# Load data
# 785 chars per image 0th-i = label
with open("mnist_train.csv", 'r') as f:
    data = f.readlines()

# load csv data into respective arrays
labels, images = [], []
for i,s in enumerate(data):
    # images
    s = s.split(',')
    s[784] = s[784].replace('\n', '')
    labels.append([0 for x in range(10)])
    labels[i][int(s[0])] = int(s[0])
    images.append(s[1:785])

# convert arrays to numpy array
train_X = np.asarray(images, dtype='int16')
train_y = np.asarray(labels, dtype='int16')

with open("mnist_test.csv", 'r') as f:
    data = f.readlines()

# load csv data into respective arrays
labels, images = [], []
for i,s in enumerate(data):
    # images
    s = s.split(',')
    s[784] = s[784].replace('\n', '')
    labels.append([0 for x in range(10)])
    labels[i][int(s[0])] = int(s[0])
    images.append(s[1:785])

# convert arrays to numpy arrays
test_X = np.asarray(images, dtype='int16')
test_y = np.asarray(labels, dtype='int16')

# model layers and build
model = Sequential()
model.add(Dense(units=784, activation='sigmoid', input_dim=784))
model.add(Dense(units=250, activation='sigmoid'))
model.add(Dense(units=10, activation='sigmoid'))
opt = optimizers.gradient_descent_v2.SGD()
model.compile(loss='mean_squared_error', optimizer=opt,
              metrics=['accuracy'])

# fit and evaluate model
model.fit(train_X, train_y, epochs=10, verbose=1, batch_size=16)
print()
print(f"\nTest Accuracy: {round(model.evaluate(test_X, test_y)[1] * 100, 3)}%")
