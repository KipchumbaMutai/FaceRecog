from numpy import load
from sklearn.metrics import accuracy_score
import numpy as np
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from faceNet_embeddings import faceEmbeddings

# load dataset


class Classifier:

    faceEmb = faceEmbeddings()

    embedding = faceEmb.get_trainEmbeddings()

    embedded_faces = faceEmb.get_embedding()

    def __init__(self, encoder, model, normalizer):

        self.model = SVC(kernel='linear', probability=True)
        self.encoder = LabelEncoder()
        self.normalizer = Normalizer()

    def svmTrain(self):

        # data = load('5-celebrity-faces-embeddings.npz')

        data = load(Classifier.embedding)

        trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
        print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
        # normalize input vectors
        in_encoder = Normalizer(norm='l2')
        trainX = in_encoder.transform(trainX)
        testX = in_encoder.transform(testX)
        # label encode targets
        out_encoder = LabelEncoder()
        out_encoder.fit(trainy)
        trainy = out_encoder.transform(trainy)
        testy = out_encoder.transform(testy)
        # fit model
        # self.model = SVC(kernel='linear', probability=True)
        self.model.fit(trainX, trainy)
        # predict
        yhat_train = model.predict(trainX)
        yhat_test = model.predict(testX)
        # score
        score_train = accuracy_score(trainy, yhat_train)
        score_test = accuracy_score(testy, yhat_test)
        # summarize
        print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

    """
    a function to make predictions of every face in the video
    """
    def predictFace(self):
        faces = Classifier.embedded_faces
        face_norm = self.normalizer(faces)
        samples = np.expand_dims(face_norm, axis=0)
        predClass = self.model.predict(samples)
        return predClass

