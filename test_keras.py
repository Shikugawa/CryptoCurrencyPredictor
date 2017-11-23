from sklearn import preprocessing
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

SERIES_LENGTH = 1000

# Prepare dataset
dataset = np.arange(1, SERIES_LENGTH+1, 1).reshape(SERIES_LENGTH, 1).astype(np.float)
# print(dataset)
# Transform
scaler = preprocessing.MinMaxScaler()
dataset = scaler.fit_transform(dataset)
# print(dataset)
# Split dataset into train and test subsets
train_dataset = dataset[0:int(len(dataset)*0.8), :]
test_dataset =  dataset[len(train_dataset):len(dataset), :]

look_back = 1
x, y = [], []
for i in range(len(dataset) - look_back):
    a = i + look_back
    x.append(dataset[i:a, 0])
    y.append(dataset[a, 0])

x = np.array(x)
y = np.array(y)
x = np.reshape(x, (x.shape[0], 1, x.shape[1]))

old_session = KTF.get_session()

with tf.Graph().as_default():
    session = tf.Session('')
    KTF.set_session(session)
    KTF.set_learning_phase(1)

    model = Sequential()
    model.add(LSTM(128, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()

    tb_cb = TensorBoard(log_dir='./', histogram_freq=1)
    cbks = [tb_cb]

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=0)

    model.fit(x_train, y_train, callbacks=cbks, batch_size=2, epochs=5, verbose=1, validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)


KTF.set_session(old_session)