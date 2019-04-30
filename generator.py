from keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
from utils import BASE_DIR, encoding
import numpy as np
import random
import cv2
import os

def csv_image_generator(inputPath, bs, mode="train"):
    sample_path = os.path.join(inputPath,'samples')
    # open the CSV file for reading
    f = open(os.path.join(inputPath,mode+'.csv'), 'r')
    # loop indefinitely
    while True:
        # initialize our batches of images and labels
        images = []
        labels = []

        # keep looping until we reach our batch size
        while len(images) < bs:
            # attempt to read the next line of the CSV file
            line = f.readline()
            if line == '\n':
                continue
            # check to see if the line is empty, indicating we have
            # reached the end of the file
            if line == "":
                # reset the file pointer to the beginning of the file
                # and re-read the line
                f.seek(0)
                line = f.readline()

                # if we are evaluating we should now break from our
                # loop to ensure we don't continue to fill up the
                # batch from samples at the beginning of the file

            # extract the label and construct the image
            line = line.strip().split(",")
            img_path = line[0]
            label = line[1]
            #print(img_path,label)
            if img_path == 'image_name':
                continue
            image = cv2.imread(os.path.join(os.path.join(sample_path,label), img_path))
            #image = np.array(image)
            # image = np.array([int(x) for x in line[1:]], dtype="uint8")
            #image = image.reshape((64, 64, 3))

            lab = [0,0,0,0,0,0,0,0]
            lab[encoding[label]-1] = 1
            # update our corresponding batches lists
            if type(images) == list:
                images.append(image)
            else:
                np.append(images,[image], axis=0)
            if type(images) == list:
                labels.append(lab)
            else:
                np.append(labels,[lab], axis=0)
            if mode=="eval":
                continue
            # apply augmentation and add to training set
            crop = image[:96, :96]
            crop = cv2.resize(crop, (128, 128))
            images.append(crop)
            labels.append(lab)

            # Center Top Crop
            crop = image[:96, 16:112]
            crop = cv2.resize(crop, (128, 128))
            images.append(crop)
            labels.append(lab)

            # Right Top Crop
            crop = image[:96, 32:]
            crop = cv2.resize(crop, (128, 128))
            images.append(crop)
            labels.append(lab)

            # Left Center Crop
            crop = image[16:112, :96]
            crop = cv2.resize(crop, (128, 128))
            images.append(crop)
            labels.append(lab)

            # Center Center Crop
            crop = image[16:112, 16:112]
            crop = cv2.resize(crop, (128, 128))
            images.append(crop)
            labels.append(lab)

            # Right Center Crop
            crop = image[16:112, 32:]
            crop = cv2.resize(crop, (128, 128))
            images.append(crop)
            labels.append(lab)

            # Left Bottom Crop
            crop = image[32:, :96]
            crop = cv2.resize(crop, (128, 128))
            images.append(crop)
            labels.append(lab)

            # Center Bottom Crop
            crop = image[32:, 16:112]
            crop = cv2.resize(crop, (128, 128))
            images.append(crop)
            labels.append(lab)

            # Right Bottom Crop
            crop = image[32:, 32:]
            crop = cv2.resize(crop, (128, 128))
            images.append(crop)
            labels.append(lab)

            # shuffle the new data
        try:
            combined = list(zip(images, labels))
            random.shuffle(combined)
            images[:], labels[:] = zip(*combined)
        except:
            pass
            print('Error will occur now')
        # yield the batch to the calling function
        yield (np.array(images), np.array(labels))