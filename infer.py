# develop a classifier for the 5 Celebrity Faces Dataset
from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
import numpy
import pickle as p
import numpy as np
from scipy.stats import mode
from sklearn.cluster import KMeans

def main():
    # load faces
    data = load('scientist-faces-dataset.npz')
    testX_faces = data['arr_0']
    # load face embeddings
    data = load('scientist-faces-embeddings.npz')
    trainX, trainy = data['arr_0'], data['arr_1']
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    numpy.save('classes.npy', out_encoder.classes_)

    trainy = out_encoder.transform(trainy)
    # fit model
    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainy)
    # test model on a random example from the test dataset
    selection = choice([i for i in range(trainX.shape[0])])
    random_face_pixels = testX_faces[selection]
    random_face_emb = trainX[selection]
    random_face_class = trainy[selection]
    random_face_name = out_encoder.inverse_transform([random_face_class])
    # prediction for the face
    samples = expand_dims(random_face_emb, axis=0)
    yhat_class = model.predict(samples)
    yhat_prob = model.predict_proba(samples)
    # get name
    class_index = yhat_class[0]
    class_probability = yhat_prob[0,class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_class)
    print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
    print('Expected: %s' % random_face_name[0])
    # plot for funt
    pyplot.imshow(random_face_pixels)
    title = '%s (%.3f)' % (predict_names[0], class_probability)
    pyplot.title(title)
    pyplot.show()

if __name__ == "__main__":
    main()