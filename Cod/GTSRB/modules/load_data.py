import os
from typing import Tuple

import numpy as np
import pandas as pd
from keras.utils import img_to_array, load_img
from modules.config import IMG_SIZE, NUM_CLASSES
from sklearn.preprocessing import LabelBinarizer


# Function to load the images and labels from the dataset
def load_training_data(data_dir):
    images = []
    labels = []
    bboxes = []
    image_paths = []

    # loop over all 42 classes
    for c in range(0, NUM_CLASSES):
        prefix = os.path.join(data_dir, format(c, "05d"))  # subdirectory for class
        with open(os.path.join(prefix, "GT-" + format(c, "05d") + ".csv")) as gtFile:
            annotations = pd.read_csv(gtFile, sep=";")
            # loop over all images in current annotations file
            for _, row in annotations.iterrows():
                impath = os.path.join(prefix, row[0])
                image = img_to_array(load_img(impath, target_size=IMG_SIZE))
                label = row[7]
                w = int(row[1])
                h = int(row[2])
                xmin = int(row[3]) / w
                ymin = int(row[6]) / h
                xmax = int(row[5]) / w
                ymax = int(row[4]) / h
                # print("Loading image {} with label {}".format(row[0], label))
                images.append(image)  # the 1st column is the filename
                labels.append(label)  # the 8th column is the label
                bboxes.append((xmin, ymin, xmax, ymax))
                image_paths.append(impath)

    # one-hot encoding
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    # normalize -> from [0-255] to [0-1]
    images = np.array(images, dtype="float32") / 255.0

    # convert to np arrays
    labels = np.array(labels)
    bboxes = np.array(bboxes, dtype="float32")
    image_paths = np.array(image_paths)

    return images, labels, bboxes, image_paths


def load_test_data(data_dir):
    images = []
    bboxes = []
    image_paths = []

    with open(os.path.join(data_dir, "GT-final_test.test.csv")) as csvFile:
        annotations = pd.read_csv(csvFile, sep=";")
        # loop over all images in current annotations file
        for _, row in annotations.iterrows():
            impath = os.path.abspath(os.path.join(data_dir, row[0]))
            image = img_to_array(load_img(impath, target_size=IMG_SIZE))
            w = int(row[1])
            h = int(row[2])
            xmin = int(row[3]) / w
            ymin = int(row[6]) / h
            xmax = int(row[5]) / w
            ymax = int(row[4]) / h
            # print("Loading image {} with label {}".format(row[0], label))
            images.append(image)  # the 1st column is the filename
            bboxes.append((xmin, ymin, xmax, ymax))
            image_paths.append(impath)

    # normalize -> from [0-255] to [0-1]
    images = np.array(images, dtype="float32") / 255.0
    bboxes = np.array(bboxes, dtype="float32")
    image_paths = np.array(image_paths)

    return images, bboxes, image_paths
