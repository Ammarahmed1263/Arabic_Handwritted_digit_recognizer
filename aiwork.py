import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout


dataset = tf.keras.datasets.mnist

#### train - test - split ####
(X_train, y_train), (X_test, y_test) = dataset.load_data()


#### normalize value to b/w 0and1 ###
X_train= X_train/255.0
X_test= X_test/255.0


### CNN (BATCH , HEIGHT, WIDTH, 1)
#### ANN (BATCH_SIZE, FEATURES)
#### FEATURES = WIDTH * HEIGHT
#### reshape array to fit in network ####
#flatten each 28x28 image into 28^2=784

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
print(X_train.shape) #(60000, 784)
print(X_test.shape) #(10000, 784)

# (batch_size, height, width, 1)
#### ANN ########
# 0-1
model = Sequential()
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

## [0-9] ##
model.add(Dense(10, activation='softmax'))

model.compile('adam', 'sparse_categorical_crossentropy', metrics=['acc'])

model.fit(X_train, y_train, epochs=3, batch_size=12, validation_split=0.1)
model.save('digit_trained.h5')

loss, accuracy= model.evaluate(X_test,y_test)
print(loss)
print(accuracy)

#### making prediction #######
plt.imshow(X_test[255].reshape(28,28), cmap='gray')
plt.xlabel(y_test[255])
plt.ylabel(np.argmax(model.predict(X_test)[255]))
plt.show()


