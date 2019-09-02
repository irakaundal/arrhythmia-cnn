from keras import Sequential
from keras.layers import Conv1D, BatchNormalization, MaxPool1D, Flatten, Dense, Dropout
from read_data_1d_fragments import read_data
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import f1_score
from keras.callbacks import ModelCheckpoint

def define_model():
    model = Sequential()
    model.add(Conv1D(input_shape=(3600, 1), filters=128, kernel_size=50, strides=3, activation='relu', kernel_initializer='glorot_uniform', data_format="channels_last"))
    model.add(BatchNormalization())
    model.add(MaxPool1D(pool_size=2, strides=3))

    model.add(Conv1D(filters=32, kernel_size=7, strides=1, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(MaxPool1D(pool_size=2, strides=2))

    model.add(Conv1D(filters=32, kernel_size=10, strides=1, activation='relu', kernel_initializer='glorot_uniform'))

    model.add(Conv1D(filters=128, kernel_size=5, strides=2, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(MaxPool1D(pool_size=2, strides=2))

    model.add(Conv1D(filters=256, kernel_size=15, strides=1, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(MaxPool1D(pool_size=2, strides=2))

    model.add(Conv1D(filters=512, kernel_size=5, strides=1, activation='relu', kernel_initializer='glorot_uniform'))

    model.add(Conv1D(filters=128, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform'))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(17, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def run():
    X_train ,y_train, X_validation, y_validation, X_test, y_test = read_data()
    y_train = np.array(y_train)
    y_validation = np.array(y_validation)
    y_test = np.array(y_test)
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = label_encoder.fit_transform(y_train)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    y_train = onehot_encoder.fit_transform(integer_encoded)
    integer_encoded = label_encoder.fit_transform(y_validation)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    y_validation = onehot_encoder.fit_transform(integer_encoded)
    integer_encoded = label_encoder.fit_transform(y_test)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    y_test = onehot_encoder.fit_transform(integer_encoded)
    combined = list(zip(X_train, y_train))
    random.shuffle(combined)
    X_train[:], y_train[:] = zip(*combined)
    combined = list(zip(X_validation, y_validation))
    random.shuffle(combined)
    X_validation[:], y_validation[:] = zip(*combined)
    combined = list(zip(X_test, y_test))
    random.shuffle(combined)
    X_test[:], y_test[:] = zip(*combined)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
    model = define_model()
    filepath = "model_1d_fragments_balanced_class_balanced_validation_without_mean-{epoch:02d}-{val_acc:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    history = model.fit(X_train, y_train, batch_size=16, epochs=50, validation_data=(X_validation, y_validation), callbacks= callbacks_list)
    eval_model = model.evaluate(X_test, y_test, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], eval_model[1] * 100))
    print("%s: %.2f%%" % (model.metrics_names[0], eval_model[0]))
    #model.save_weights('model_1d_fragments_type2_withoutconst_new.h5')
    y_pred = model.predict(X_test, verbose=1)
    predicted_class1 = np.zeros(y_pred.shape)
    predicted_class1[y_pred > 0.5] = 1
    y_pred = np.argmax(predicted_class1, axis=1)
    y_test = np.argmax(y_test, axis=1)
    acc = accuracy_score(y_test, y_pred)
    cm1 = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix : \n', cm1)
    mcm = multilabel_confusion_matrix(y_test, y_pred)
    print('Multilabel Confusion Matrix : \n', mcm)
    tn = mcm[:, 0, 0]
    tp = mcm[:, 1, 1]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1_scor = f1_score(y_test, y_pred, average='weighted')
    with open('Results_new.txt', 'a+') as f:
        f.write("Validation for Fragments Type 2 & Subtracting Mean Component - " + "%s: %.2f%%" % (model.metrics_names[1], eval_model[1] * 100) + "  " +
                "%s: %.2f%%" % (model.metrics_names[0], eval_model[0]) + ' Testing Accuracy - ' + str(acc*100) +
                ' Sensitivity - '+ str(np.mean(sensitivity)*100) + ' Specificity - '+ str(np.mean(specificity)*100) + ' F1 Score - '+ str(f1_scor*100) +'\n')
    f.close()

if __name__ == "__main__":
    run()