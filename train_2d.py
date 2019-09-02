from utils import TRAIN_CSV, TEST_CSV, BS, NUM_EPOCHS, BASE_DIR, SAMPLE_DIR,encoding
from model_2d import define_model
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint
import os
import cv2
import numpy as np
import random
from sklearn.metrics import accuracy_score
import statistics
import pickle
import matplotlib.pyplot as plt

def plot(history, filename):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(filename+'_acc')
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show(filename+'_loss')

X = []
y = []
sample_dir = os.path.join(BASE_DIR, 'samples')
list_samples = os.listdir(sample_dir)
for sam in list_samples:
    print("Loading images for "+sam)
    im_dir = os.path.join(sample_dir, sam)
    list_imgs = os.listdir(im_dir)
    for i in list_imgs:
        image = cv2.imread(os.path.join(im_dir,i))
        if type(X) == list:
            X.append(image)
        else:
            np.append(X, [image], axis=0)
        y.append(sam)
        if sam != 'NOR':
            # apply augmentation and add to training set
            crop = image[:96, :96]
            crop = cv2.resize(crop, (128, 128))
            X.append(crop)
            y.append(sam)

            # Center Top Crop
            crop = image[:96, 16:112]
            crop = cv2.resize(crop, (128, 128))
            X.append(crop)
            y.append(sam)

            # Right Top Crop
            crop = image[:96, 32:]
            crop = cv2.resize(crop, (128, 128))
            X.append(crop)
            y.append(sam)

            # Left Center Crop
            crop = image[16:112, :96]
            crop = cv2.resize(crop, (128, 128))
            X.append(crop)
            y.append(sam)

            # Center Center Crop
            crop = image[16:112, 16:112]
            crop = cv2.resize(crop, (128, 128))
            X.append(crop)
            y.append(sam)

            # Right Center Crop
            crop = image[16:112, 32:]
            crop = cv2.resize(crop, (128, 128))
            X.append(crop)
            y.append(sam)

            # Left Bottom Crop
            crop = image[32:, :96]
            crop = cv2.resize(crop, (128, 128))
            X.append(crop)
            y.append(sam)

            # Center Bottom Crop
            crop = image[32:, 16:112]
            crop = cv2.resize(crop, (128, 128))
            X.append(crop)
            y.append(sam)

            # Right Bottom Crop
            crop = image[32:, 32:]
            crop = cv2.resize(crop, (128, 128))
            X.append(crop)
            y.append(sam)

combined = list(zip(X, y))
random.shuffle(combined)
X[:], y[:] = zip(*combined)
skf = StratifiedKFold(n_splits=10, shuffle=True)
skf.get_n_splits(X, y)
X = np.array(X)
y = np.array(y)
accs = []
for index, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    with open("X_train_"+str(index+1)+".pkl", 'wb') as f:
        pickle.dump(X_train, f , protocol=4)
    f.close()
    with open("X_test_"+str(index+1)+".pkl", 'wb') as f:
        pickle.dump(X_test, f, protocol=4)
    f.close()
    with open("y_train_"+str(index+1)+".pkl", 'wb') as f:
        pickle.dump(y_train, f, protocol=4)
    f.close()
    with open("y_test_"+str(index+1)+".pkl", 'wb') as f:
        pickle.dump(y_test, f, protocol=4)
    f.close()

    '''d = {'X_train': X_train,
         'X_test': X_test,
         'y_train': y_train,
         'y_test': y_test}
    pickle_out = open("training_set_"+str(index+1)+".pickle", "wb")
    pickle.dump(d, pickle_out, protocol=4)
    pickle_out.close()'''

    checkpoint = ModelCheckpoint("model_"+str(index+1)+".h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    model = None
    model = define_model()
    # Debug message I guess
    print ("Training new iteration on " + str(X_train.shape[0]) + " training samples, " + str(X_test.shape[0]) + " validation samples, this may be a while...")
    y_tr = []
    y_te = []
    for i in range(0, len(y_train)):
        lab = [0, 0, 0, 0, 0, 0, 0, 0]
        lab[encoding[y_train[i]] - 1] = 1
        y_tr.append(lab)
    for i in range(0, len(y_test)):
        lab = [0, 0, 0, 0, 0, 0, 0, 0]
        lab[encoding[y_test[i]] - 1] = 1
        y_te.append(lab)
    y_train = np.array(y_tr)
    y_test = np.array(y_te)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    history = model.fit(X_train, y_train, batch_size=64, epochs=30)
    eval_model = model.evaluate(X_test, y_test, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], eval_model[1]*100))
    print("%s: %.2f%%" % (model.metrics_names[0], eval_model[0]))
    accs.append(eval_model[1]*100)
    #plot(history= history, filename="plot_"+str(index+1))
    model.save_weights('model_'+str(index+1)+'.h5')
    with open('results.txt', 'a+') as f:
        f.write("Set "+ str(index+1)+" Loss is: "+str(eval_model[0])+" Accuracy is: " + str(eval_model[1] * 100) +'\n')
    f.close()

print("List of accuracies: ", accs)
print("Final accuracy: ", str(statistics.mean(accs)))
with open('results.txt', 'a+') as f:
    f.write("Final List of accuracies: " + str(accs) +"Final Mean accuracy: "+str(statistics.mean(accs)) + '\n')
f.close()
'''
f = open(TRAIN_CSV, "r")
labels = set()
testLabels = []
NUM_TRAIN_IMAGES = 0
NUM_TEST_IMAGES = 0

# loop over all rows of the CSV file
for line in f:
    # extract the class label, update the labels list, and increment
    # the total number of training images
    if line == '\n':
        continue
    label = line.strip().split(",")[0]
    img_path = line[0]
    if img_path == 'image_name':
        continue
    labels.add(label)
    NUM_TRAIN_IMAGES += 1

# close the training CSV file and open the testing CSV file
f.close()
f = open(TEST_CSV, "r")

# loop over the lines in the testing file
for line in f:
    # extract the class label, update the test labels list, and
    # increment the total number of testing images
    if line == '\n':
        continue
    label = line.strip().split(",")[0]
    img_path = line[0]
    if img_path == 'image_name':
        continue
    testLabels.append(label)
    NUM_TEST_IMAGES += 1

# close the testing CSV file
f.close()

lb = LabelBinarizer()
lb.fit(list(labels))
testLabels = lb.transform(testLabels)

# construct the training image generator for data augmentatione

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
    horizontal_flip=True, fill_mode="nearest")


trainGen = csv_image_generator(BASE_DIR, BS,
    mode="train")
testGen = csv_image_generator(BASE_DIR, BS,
    mode="eval")

checkpoint = ModelCheckpoint('model_new.h5',
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True,
                            save_weights_only=True,
                            mode='min')

model.fit_generator(
generator=trainGen,
steps_per_epoch=NUM_TRAIN_IMAGES // BS,
validation_data=testGen,
validation_steps=NUM_TEST_IMAGES,
epochs=NUM_EPOCHS,
verbose=1,
workers=1,
max_queue_size=8,
callbacks=[checkpoint]
)

X_train = []
y_train = []
with open('train.csv', 'r') as csvFile:
    reader = csv.DictReader(csvFile)
    for row in reader:
        img_path = row['image_name']
        sample = row['key']
        lab = [0, 0, 0, 0, 0, 0, 0, 0]
        lab[encoding[sample] - 1] = 1
        y_train.append(lab)
        image = cv2.imread(os.path.join(os.path.join(SAMPLE_DIR, sample), img_path))
        if type(X_train) == list:
            X_train = np.array([image])
        else:
            X_train = np.append(X_train, [image], axis=0)

X_train = np.array(X_train)
y_train = np.array(y_train)
print('yolo')
'''