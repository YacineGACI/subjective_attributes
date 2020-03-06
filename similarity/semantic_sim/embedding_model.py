import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

from features import cosine_similarity


def phrase_embedding(phrase):
    return np.mean([word_vectors[w] for w in phrase.split(' ') if w in word_vectors.vocab], axis=0)


model =  Word2Vec.load("models/word2vec.model")
word_vectors = model.wv
del model


if __name__ == "__main__":

    s1 = "food great"
    s2 = "chicken wings delicious"

    v1 = phrase_embedding(s1)
    v2 = phrase_embedding(s2)

    print(cosine_similarity(v1, v2))