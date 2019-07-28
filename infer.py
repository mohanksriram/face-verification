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

def distance(emb1, emb2):
        return np.sum(np.square(emb1 - emb2))

def get_most_similar(new_embedding, avg_embeddings):
    sims = []
    dists = [distance(new_embedding, emb2) for emb2 in avg_embeddings]

    print(f'smallest: {np.argmin(dists)}')
    return np.argmin(dists)

def main():
    # load faces
    data = load('faces-dataset.npz')
    testX_faces = data['arr_2']
    # load face embeddings
    classes = np.load('models/classes.npy')
    print(classes)

    data = load('faces-embeddings.npz')
    testX, testy = data['arr_2'], data['arr_3']
    in_encoder = Normalizer(norm='l2')
    testX = in_encoder.transform(testX)


    # test model on a random example from the test dataset
    selection = choice([i for i in range(testX.shape[0])])
    print(selection)
    random_face_pixels = testX_faces[selection]
    random_face_emb = testX[selection]
    random_face_name = testy[selection]
    
    
    # fit model
    avg_embeddings = numpy.load('models/avg_embeddings.npy')

    yhat_class = classes[get_most_similar(random_face_emb, avg_embeddings)]

    print('Predicted: %s' % (yhat_class))
    print('Expected: %s' % random_face_name)
    # plot for fun
    pyplot.imshow(random_face_pixels)
    title = '%s' % (yhat_class)
    pyplot.title(title)
    pyplot.show()

if __name__ == "__main__":
    main()