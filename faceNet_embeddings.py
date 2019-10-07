# calculate a face embedding for each face in the dataset using facenet
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model


class faceEmbeddings:

    """"
    get the face embedding for one face

    """

    def get_embedding(self, model, face_pixels):
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        yhat = model.predict(samples)
        return yhat[0]
    def load_data(self, file_name):
        self.data = load(file_name)


    def get_trainEmbeddings(self):

        # data = load('5-celebrity-faces-dataset.npz')
        # data1 = self.data
        """
        use self.data['arr-i'] to access the properties from the load_data function
        """
        trainX, trainy, testX, testy = self.data['arr_0'], self.data['arr_1'], self.data['arr_2'], self.data['arr_3']
        print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
        # load the facenet model
        model = load_model('facenet_keras.h5')
        print('Loaded Model')
        # convert each face in the train set to an embedding
        newTrainX = list()
        for face_pixels in trainX:
            embedding = self.get_embedding(model, face_pixels)
            newTrainX.append(embedding)
        newTrainX = asarray(newTrainX)
        print(newTrainX.shape)
        # convert each face in the test set to an embedding
        newTestX = list()
        for face_pixels in testX:
            embedding = self.get_embedding(model, face_pixels)
            newTestX.append(embedding)
        newTestX = asarray(newTestX)
        print(newTestX.shape)
        # save arrays to one file in compressed format
        return savez_compressed('5-celebrity-faces-embeddings.npz', newTrainX, trainy, newTestX, testy)
