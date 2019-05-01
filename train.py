from keras.preprocessing.image import ImageDataGenerator
from utils import TRAIN_CSV, TEST_CSV, BS, NUM_EPOCHS, BASE_DIR, SAMPLE_DIR,encoding
from sklearn.preprocessing import LabelBinarizer
from callbacks import CustomModelCheckpoint
from generator import csv_image_generator
from model import define_model
from keras.callbacks import ModelCheckpoint


model = define_model()

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

# construct the training image generator for data augmentation
'''
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
    horizontal_flip=True, fill_mode="nearest")
'''

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
'''
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