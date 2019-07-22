from pathlib import Path
from os import listdir
from os.path import isdir
from PIL import Image

from numpy import asarray
from numpy import savez_compressed
from numpy import load
from numpy import expand_dims
from random import choice
import numpy as np


import mtcnn
from mtcnn.mtcnn import MTCNN
import face_recognition


DATA_PATH = Path("./data/custom")

# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = np.array(image)
    # create the detector, using default weights
    detector = face_recognition

    face_locations = detector.face_locations(pixels, number_of_times_to_upsample=0, model="cnn")
    top, right, bottom, left = face_locations[0]

    # extract the face
    face = pixels[top:bottom, left:right]

    # resize pixels to the model size
    image = Image.fromarray(face)

    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

# load images and extract faces for all images in a directory
def load_faces(directory):
    faces = []
    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + filename
        # get face
        face = extract_face(path)
        # store
        faces.append(face)
    return faces

def load_dataset(directory):
    X, y = [], []
    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        #print(f'{subdir}')
        path = directory + subdir + '/'
        #print(f'path is: {path}')
        # skip any files that might be in the dir
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)


def main():
    # Load the training dataset
    x_train, y_train = load_dataset(str(DATA_PATH) + '/train/')
    print(f'Loaded training data - x_shape: {x_train.shape}, y_shape: {y_train.shape}')
    savez_compressed('scientist-faces-dataset.npz', x_train, y_train)
    print('Data saved.')

if __name__ == '__main__':
    main()

