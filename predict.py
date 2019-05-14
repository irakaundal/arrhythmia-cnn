from utils import BS,BASE_DIR
from generator import csv_image_generator
from keras.models import load_model
from keras.metrics import categorical_accuracy
from sklearn.metrics import accuracy_score
from utils import BASE_DIR, encoding
import numpy as np
import cv2
import os
from utils import SAMPLE_DIR
import pickle
from model import define_model

model = define_model()
model.load_weights('model_1.h5')
with open('X_test_1.pkl', 'rb') as f:
    X_test = pickle.load(f)
f.close()
with open('y_test_1.pkl', 'rb') as f:
    y_test = pickle.load(f)
f.close()
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
idx = np.argmax(y_pred, axis=-1)
y_pred = np.zeros(y_pred.shape)
y_pred[np.arange(y_pred.shape[0]), idx] = 1
# acc = categorical_accuracy(y_test, y_pred)
y = []
for i in range(0, len(y_test)):
    lab = [0, 0, 0, 0, 0, 0, 0, 0]
    lab[encoding[y_test[i]] - 1] = 1
    y.append(lab)
acc = accuracy_score(np.array(y), y_pred)
print("Accuracy is: ", acc * 100)
'''X_test = []
y_test = []

with open('test.csv') as f:
    lines = f.readlines()
    for line in lines:
        label = [0, 0, 0, 0, 0, 0, 0, 0]
        if line == "\n":
            continue
        image, category = line.split(',')
        category = category.strip('\n')
        if image == "image_name":
            continue
        im = cv2.imread(os.path.join(os.path.join(SAMPLE_DIR, category), image))
        X_test.append(im)
        label[encoding[category] - 1] = 1
        y_test.append(label)
X_test = np.array(X_test)
y_test = np.array(y_test)
y_pred = model.predict(X_test)
idx = np.argmax(y_pred, axis=-1)
y_pred = np.zeros(y_pred.shape)
y_pred[np.arange(y_pred.shape[0]), idx] = 1
#acc = categorical_accuracy(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print("Accuracy is: ", acc*100)


testGen = csv_image_generator(BASE_DIR, BS, mode="test", aug=None)
predIdxs = model.predict_generator(testGen, steps=(1600 // BS)+1)
predIdxs = np.argmax(predIdxs, axis=1)
print(predIdxs)
'''