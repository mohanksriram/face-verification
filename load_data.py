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

import os
import mtcnn
from mtcnn.mtcnn import MTCNN
import face_recognition
import cv2
import time
from enum import Enum
import matplotlib.pyplot as plt

class EDetectorType(Enum):
    MTCNN = 0
    FACE_RECOGNITION = 1

model_file = "opencv_face_detector_uint8.pb"
config_file = "opencv_face_detector.pbtxt"
dnn_weights_path = 'models'
modelFile = os.path.join(dnn_weights_path, model_file)
configFile = os.path.join(dnn_weights_path, config_file)


DATA_PATH = Path("./data/custom")

def get_face_locations(image):
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    frame_width = image.shape[1]
    frame_height = image.shape[0]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            area = width * height

            bboxes.append(([x1, y1, x2, y2], area))
    bboxes.sort(key=lambda x: x[1], reverse=True)
    return bboxes

def extract_cv_face(image, required_size=(160, 160)):
    start_time = time.time()
    
    image = np.array(image, dtype='uint8')
    face_locations = get_face_locations(image)

    end_time = time.time() - start_time
    #print('Time for dnn Detector: {}'.format(end_time))

    if len(face_locations) > 0:

        x1, y1, x2, y2 = face_locations[0][0]
        face_rect = face_locations[0][0]
        
        full_pixels = image

        # face = full_pixels[top:bottom, left:right]

        face = full_pixels[y1:y2, x1:x2]


        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)


    else:
        return []
    
    return face_array


# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160), det_type=EDetectorType.FACE_RECOGNITION):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = np.array(image)
    # create the detector, using default weights

    if det_type == EDetectorType.MTCNN:
        detector = MTCNN()
        # detect faces in the image
        results = detector.detect_faces(pixels)
        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
    
    else:
        print(f'loading face reg detector')
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
        img = cv2.imread(path)
        # get face
        face = extract_cv_face(img)
        # store
        faces.append(face)
        plt.imshow(face)
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

    # load test dataset
    x_test, y_test = load_dataset(str(DATA_PATH) + '/val/')
    savez_compressed('faces-dataset.npz', x_train, y_train, x_test, y_test)
    
    print('Data saved.')
    print('showing data!')
    plt.imshow(x_train[0])
    plt.show()


if __name__ == '__main__':
    main()

