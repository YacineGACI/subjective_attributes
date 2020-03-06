import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

from features import cosine_similarity


def phrase_embedding(phrase):
    return np.mean([word_vectors[w] for w in phrase.split(' ') if w in word_vectors.vocab], axis=0)


model =  Word2Vec.load("models/word2vec_sg_100_5_5.model")
word_vectors = model.wv
del model