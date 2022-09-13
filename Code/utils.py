import pathlib

import numpy as np
import os
import cv2

from pydicom import dcmread, uid

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

IMG_SIZE = 160

def get_labels():
    return ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']


def read_img(path):
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img


def load_training_data(val_size=0.2):
    path = './data/'
    labels = get_labels()

    X = []
    y = []

    # Get the images from both the training and testing directories (we will split them ourselves)
    for label in labels:
        trainingLoc = 'Training/' + label
        trainLabelPath = os.path.join(path, trainingLoc)

        # read all the images from the training directory
        for imgFilename in os.listdir(trainLabelPath):
            X.append(read_img(os.path.join(trainLabelPath, imgFilename)))
            y.append(labels.index(label))


    # Convert everything to numpy arrays
    X = np.array(X)
    X = X / 255.0
    y = np.array(y)

    # Shuffle so that all the same labels aren't next to each other
    X, y = shuffle(X, y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size)

    return X_train, X_val, y_train, y_val 


def load_partition(index):
    num_clients = 4
    part_size = 1 / num_clients
    X_train, X_val, y_train, y_val = load_training_data(0.2)
    return (
        X_train[round(index * len(X_train) * part_size) : round((index+1) * len(X_train) * part_size)],
        X_val[round(index * len(X_val) * part_size) : round((index+1) * len(X_val) * part_size)],
        y_train[round(index * len(y_train) * part_size) : round((index+1) * len(y_train) * part_size)],
        y_val[round(index * len(y_val) * part_size) : round((index+1) * len(y_val) * part_size)]
    )
    

def load_testing_data():
    path = './data/'
    labels = get_labels()

    X_test = []
    y_test = []

    # Get the images from both the training and testing directories (we will split them ourselves)
    for label in labels:
        testingLoc = 'Testing/' + label
        testLabelPath = os.path.join(path, testingLoc)

        # read all the images from the testing directoory 
        for imgFilename in os.listdir(testLabelPath):
            X_test.append(read_img(os.path.join(testLabelPath, imgFilename)))
            y_test.append(labels.index(label))

    # Convert everything to numpy arrays
    X_test = np.array(X_test)
    X_test = X_test / 255.0
    y_test = np.array(y_test)

    return X_test, y_test


def load_data_dicom(test_size=0.2):
    dicom_dir = pathlib.Path(__file__).parent / "dicom"
    X, y = [], []

    for file in dicom_dir.glob("*.dcm"):
        ds = dcmread(file, force=True)
        ds.file_meta.TransferSyntaxUID = uid.ImplicitVRLittleEndian
        X.append(ds.pixel_array)
        y.append(ds.DataSetDescription)

    X = np.array(X)
    X = X / 255.0
    y = np.array(y)

    # Shuffle so that all the same labels aren't next to each other
    X, y = shuffle(X, y)

    # Tuple of X_train, X_test, y_train, y_test
    return train_test_split(X, y, test_size=test_size)
    

if __name__ == '__main__':
    # test to see that function works properly
    # #load_random_subset(0.2)
    # X_train, X_val, y_train, y_val = load_partition(0)
    # X_train1, X_val1, y_train1, y_val1 = load_partition(1)
    # print(X_train.shape)
    # print(X_train1.shape)
    #
    # if (len(sys.argv) > 1):
    #     print("reading from folder")
    #     X_train, X_test, y_train, y_test = load_data_from_folder(sys.argv[1])
    #     print(X_train.shape)
    #     print(y_train.shape)
    # else:
    print("reading all")
    X_train, X_test, y_train, y_test = load_data_dicom(0.2)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    print(X_train[1].shape)

