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

def create_model(trainX, n_classes):
    # cluster feature vectors
    kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(trainX)
    model = kmeans
    return model

def sim_predict(model, new_img_f, orig_classes, top_n=1, n_classes=5):
    # cluster labels do not match with actual order of train data. so find indices to reorder cluster centres
    kmeans = model
    steps = np.linspace(0, len(kmeans.labels_), num=n_classes+1)
    orig_labels = []
    last_val = 0
    for i in steps[1:]:
        cluster_labels = kmeans.labels_[last_val:int(i)]
        last_val = int(i)
        orig_labels += [mode(cluster_labels)[0][0]]

    # new_map = {}
    # for i, label in enumerate(encode_labels):
    #     new_map[]


    relabeled = kmeans.cluster_centers_[orig_labels]
    sims = np.array([])
    for i in range(relabeled.shape[0]):
        sim = np.dot(relabeled[i],new_img_f)
        sims = np.append(sims,sim)
    sims_top_n = sims.argsort()[-top_n:][::-1]
    classes = sims_top_n

    classes = [orig_classes[val] for val in classes]
    
    #print(f'new_classes: {classes}')
    probs = sims[sims_top_n]
    #print(f'classes: {classes}')
    return classes, probs

def main():
    # load dataset
    data = load('scientist-faces-embeddings.npz')
    trainX, trainy = data['arr_0'], data['arr_1']
    print(trainy)
    print('Dataset: train=%d' % (trainX.shape[0]))
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    numpy.save('models/svc_classes.npy', out_encoder.classes_)
    
    orig_classes = [x for i, x in enumerate(trainy) if i == list(trainy).index(x)]
    with open("models/kmeans_classes.txt", "wb") as fp:   #Pickling
        pickle.dump(orig_classes, fp)

    trainy = out_encoder.transform(trainy)

    print(f'transformed trainy = {trainy}')
    
    
    # SVC model
    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainy)

    # predict
    yhat_train = model.predict(trainX)
    # score
    score_train = accuracy_score(trainy, yhat_train)
    # summarize
    print('SVC Model Accuracy: train=%.3f' % (score_train*100))

    filename = 'models/svc_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    print(f'saved svc model')

    # Kmeans model
    # testing the above methods
    classifier_model = create_model(trainX, 5)

    # save the model to disk
    # predict
    yhat_train =  [sim_predict(classifier_model, trainX[i], orig_classes,  1)[0][0] for i in range(trainX.shape[0])]
    yhat_train = out_encoder.transform(yhat_train) 

    # score
    score_train = accuracy_score(trainy, yhat_train)
    # summarize
    print('Accuracy: train=%.3f' % (score_train*100))

    filename = 'models/kmeans_model.sav'
    pickle.dump(classifier_model, open(filename, 'wb'))
    print('saved kmeans model')

if __name__ == "__main__":
    main()