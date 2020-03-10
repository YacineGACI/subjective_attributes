import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

from similarity.semantic_sim.features import cosine_sim


def word2vec_embedding(phrase):
    return np.mean([word_vectors[w] for w in phrase.split(' ') if w in word_vectors.vocab], axis=0)


def paragram_embedding(phrase):
    return np.mean([paragram[w] for w in word_tokenize(phrase.lower()) if w in paragram.keys()], axis=0)



model =  Word2Vec.load("similarity/semantic_sim/models/word2vec_sg_100_5_5.model")
#model = Word2Vec.load("models/word2vec_nltk-tokenization_sg_100_5_20.model")
word_vectors = model.wv
del model


with open("similarity/semantic_sim/models/paragram_embed_ws353.pkl", 'rb') as f:
    paragram = pickle.load(f)



if __name__ == "__main__":

    s1 = "food great"
    s2 = "chicken wings delicious"

    v1 = phrase_embedding(s1)
    v2 = phrase_embedding(s2)

    print(cosine_sim(v1, v2))
