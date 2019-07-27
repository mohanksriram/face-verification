# develop a classifier for the 5 Celebrity Faces Dataset
from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle
import numpy
import pickle as p
import numpy as np
from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import time

def create_model(trainX, n_classes):
    # cluster feature vectors
    kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(trainX)
    model = kmeans
    return model

def gen_avg_embedding(trainX, trainy):
    num_examples = len(list(set(trainy)))
    all_embedding = [None]*num_examples

    for idx, label in enumerate(trainy):
        try:
            all_embedding[label].append(trainX[idx])
        except:
            all_embedding[label] = [trainX[idx]]

    avg_embedding = [np.average(ele, axis=0) for ele in all_embedding]

    return np.array(avg_embedding)

def distance(emb1, emb2):
        return np.sum(np.square(emb1 - emb2))

def get_most_similar(new_embeddings, avg_embeddings):
    sims = []
    for new_embedding in new_embeddings:
        dists = [distance(new_embedding, emb2) for emb2 in avg_embeddings]
        ans = np.argmin(dists)
        sims.append(ans)
    return sims

def main():
    # load dataset
    data = load('scientist-faces-embeddings.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print(trainy)
    print('Dataset: train=%d' % (trainX.shape[0]))
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    numpy.save('models/classes.npy', out_encoder.classes_)

    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)

    avg_embeddings = gen_avg_embedding(trainX, trainy)
    numpy.save('models/avg_embeddings.npy', avg_embeddings)

    yhat_train = get_most_similar(trainX, avg_embeddings)
    score_train = accuracy_score(trainy, yhat_train)

    yhat_test = get_most_similar(testX, avg_embeddings)
    score_test = accuracy_score(testy, yhat_test)

    # summarize
    print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

if __name__ == "__main__":
    main()