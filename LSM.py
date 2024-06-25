from sklearn.neural_network import *
import numpy as np
from embedding import *
import random


class LSM():
    def __init__(self, modl):
        self.model = modl
        self.embeddings = None

    def load_embeddings(self, filePath, nDim):
        self.embeddings = GloVeEmbedding(filePath,nDim)

    def train(self, text):
        X = np.array([self.embeddings.get_embeddings_from_sentence(text[:-1])])
        y = np.array([self.embeddings.get_embedding_from_word(text[-1])])
        print(X)
        print(y)
        self.model.fit(X, y)
        
    def predict(self, input):
        prediction = self.model.predict(input)

        result = self.embeddings.get_word_from_embedding(prediction)
        return result
    


