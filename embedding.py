import numpy as np
from scipy.spatial.distance import cdist

class GloVeEmbedding:
    def __init__(self, path, num_dimensions):
        self.word_to_index = {}
        self.word_vectors = []

        self.load_embeddings(path, num_dimensions)

    def load_embeddings(self, path, num_dimensions):
        with open(path, 'r', encoding='utf-8') as file:
            for index, line in enumerate(file):
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                self.word_to_index[word] = index
                self.word_vectors.append(vector)

                if len(vector) != num_dimensions:
                    raise ValueError(f"Expected {num_dimensions} dimensions, but found {len(vector)} dimensions for word '{word}'.")

        self.word_vectors = np.array(self.word_vectors)

    def get_embedding_from_word(self, word):
        index = self.word_to_index.get(word.lower())
        if index is None:
            raise ValueError(f"Word '{word}' not found in the vocabulary.")

        return self.word_vectors[index]

    def get_words(self):
        return list(self.word_to_index.keys())

    def get_embeddings_from_sentence(self, sentence):
        words = sentence.split()
        embeddings = []

        for word in words:
            embedding = self.get_embedding_from_word(word)
            embeddings.append(embedding)

        return np.array(embeddings)

    def get_word_from_embedding(self, embedding, top_n=1):
        distances = cdist(embedding[np.newaxis, :], self.word_vectors, metric='euclidean')
        indices = np.argsort(distances)
        words = [self.get_words()[i] for i in indices[:, :top_n].flatten()]

        if top_n == 1:
            return words[0]
        else:
            return [words[i:i+top_n] for i in range(0, len(words), top_n)]
    def get_sentence_from_embedding(self,embedding):
        words = []
        for word_embedding in embedding:
            word = self.get_word_from_embedding(word_embedding, top_n=1)
            words.append(word)
        sentence = ' '.join(words)
        return sentence