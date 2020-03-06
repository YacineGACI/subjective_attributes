import gensim

class Sentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                yield line.split('\t', 1)[1].replace('\t', ' ').split(' ')



sentences = Sentences('../../data/reviews/processed/reviews.txt')
# model = gensim.models.Word2Vec(sentences, size=300, min_count=1, workers=4, sg=1, iter=5)
model = gensim.models.Word2Vec(sentences, size=100, min_count=5, workers=4, sg=1, iter=5)
model.save("models/word2vec_sg_100_5_5.model") 

# Naming convention for the trained models
# word2vec_{sg|cb}_{size}_{min_count}_{epochs}.model