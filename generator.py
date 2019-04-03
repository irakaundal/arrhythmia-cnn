from keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
from utils import BASE_DIR, encoding
import numpy as np
import cv2
import os

def csv_image_generator(inputPath, bs, lb, mode="train", aug=None):
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
                if mode == "eval":
                    break

            # extract the label and construct the image
            line = line.strip().split(",")
            img_path = line[0]
            label = line[1]
            #print(img_path,label)
            if img_path == 'image_name':
                continue
            image = cv2.imread(os.path.join(os.path.join(sample_path, label), img_path))
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

            # if the data augmentation object is not None, apply it
        if aug is not None:
            (images, labels) = next(aug.flow(np.array(images),
                                             labels, batch_size=bs))

        # yield the batch to the calling function
        yield (np.array(images), np.array(labels))