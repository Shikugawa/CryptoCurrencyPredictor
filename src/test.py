from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.utils import np_utils
from keras.datasets import mnist
from keras.callbacks import TensorBoard
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import numpy as np

def convert_one_hot(label):
    label = np.reshape(np.array(label), (-1, 1))
    label = np_utils.to_categorical(label)
    return label

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

y_train = convert_one_hot(y_train)
y_test = convert_one_hot(y_test)

# --for using tensorboard--
old_session = KTF.get_session()

session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)
# --------------------------

model = Sequential()
model.add(Convolution2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# --for using tensorboard--
tb_cb = TensorBoard(log_dir="tflog/", histogram_freq=1)
cbks = [tb_cb]
# --------------------------

result = model.fit(x_train, y_train, epochs=10, verbose=1, callbacks=cbks, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# --for using tensorboard--
KTF.set_session(old_session)
# --------------------------