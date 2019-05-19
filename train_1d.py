from keras import Sequential
from keras.layers import Conv1D, BatchNormalization, MaxPool1D, Flatten, Dense, Dropout
from read_data_1d import read_data
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import random
from sklearn.model_selection import train_test_split

def define_model():
    model = Sequential()
    model.add(Conv1D(input_shape=(300, 1), filters=128, kernel_size=10, strides=3, activation='relu', kernel_initializer='glorot_uniform', data_format="channels_last"))
    model.add(BatchNormalization())
    model.add(MaxPool1D(pool_size=2, strides=3))

    model.add(Conv1D(filters=32, kernel_size=7, strides=1, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(MaxPool1D(pool_size=2, strides=2))

    model.add(Conv1D(filters=32, kernel_size=10, strides=1, activation='relu', kernel_initializer='glorot_uniform'))

    model.add(Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(MaxPool1D(pool_size=1, strides=2))

    model.add(Conv1D(filters=256, kernel_size=1, strides=1, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(MaxPool1D(pool_size=1, strides=2))

    model.add(Conv1D(filters=512, kernel_size=1, strides=1, activation='relu', kernel_initializer='glorot_uniform'))

    model.add(Conv1D(filters=128, kernel_size=1, strides=1, activation='relu', kernel_initializer='glorot_uniform'))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(17, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def run():
    X, y = read_data()
    y = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    y = onehot_encoder.fit_transform(integer_encoded)
    X = np.array(X)
    y = np.array(y)
    print('dfghj')
    combined = list(zip(X, y))
    random.shuffle(combined)
    X[:], y[:] = zip(*combined)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
    model = define_model()
    history = model.fit(X_train, y_train, batch_size=64, epochs=100)
    eval_model = model.evaluate(X_test, y_test, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], eval_model[1] * 100))
    print("%s: %.2f%%" % (model.metrics_names[0], eval_model[0]))
    model.save_weights('model_1d_cnn.h5')

if __name__ == "__main__":
    run()