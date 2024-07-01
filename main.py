import LSM
from sklearn.neural_network import MLPClassifier
from embedding import *
from keras.src.models import Sequential
from keras.src.layers.core import *
from keras.src.optimizers import SGD
import keras
import numpy as np
import matplotlib.pyplot as plt

# a = "hello world hey"
# model = keras.Sequential()
# model.add(keras.Input(shape=(50,50)))
# model.add(keras.layers.Dense(8))

# model.compile(optimizer = SGD(), loss = 'mse')

# # classifier = MLPClassifier(hidden_layer_sizes=(64,64),activation='relu',solver='adam',max_iter=3,verbose=True,)

# model = LSM.LSM(modl=model)

# model.load_embeddings("./embeddings/glove.6B.50d.txt",50)
# # tex = "are you ok cause you are the one who needed space now finally i am doing fine you would rather see me cry"

embedding = GloVeEmbedding("./embeddings/glove.6B.50d.txt",50)


embedding.get_word_from_embedding
# model.predict("hello world")