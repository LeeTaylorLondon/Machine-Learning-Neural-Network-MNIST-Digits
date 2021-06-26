from keras.models import Sequential
from keras.layers import Dense
import numpy as np

np.random.seed(1)

# Load data
with open("mnist_train.csv", 'r') as f:
    data = f.readlines()

# load csv data into respective arrays
labels, images = [], []
for i,s in enumerate(data):
    # seperate elements by comma and remove \n char
    s = s.split(',')
    s[784] = s[784].replace('\n', '')
    # append image of number to array images
    images.append(s[1:785])
    # create label/Y arrays
    labels.append([0.01 for x in range(10)])
    labels[i][int(s[0])] = 0.99

# convert arrays to numpy array
train_X = np.asarray(images, dtype='int16')
train_y = np.asarray(labels, dtype='float32')

with open("mnist_test.csv", 'r') as f:
    data = f.readlines()

# load csv data into respective arrays
labels, images = [], []
for i,s in enumerate(data):
    # seperate elements by comma and remove \n char
    s = s.split(',')
    s[784] = s[784].replace('\n', '')
    # append image of number to array images
    images.append(s[1:785])
    # create label/Y arrays
    labels.append([0.01 for x in range(10)])
    labels[i][int(s[0])] = 0.99

# convert arrays to numpy arrays
test_X = np.asarray(images, dtype='int16')
test_y = np.asarray(labels, dtype='float32')

# model layers and build
model = Sequential()
model.add(Dense(units=784, activation='sigmoid', input_dim=784))
model.add(Dense(units=500, activation='sigmoid'))
model.add(Dense(units=10, activation='sigmoid'))
#opt = optimizers.gradient_descent_v2.SGD()
model.compile(loss='mean_squared_error', optimizer='adam',
              metrics=['accuracy'])

# fit and evaluate model
model.fit(train_X, train_y, batch_size=16, epochs=5, verbose=1)
print()
print(f"\nTest Accuracy: {round(model.evaluate(test_X, test_y)[1] * 100, 4)}%")
