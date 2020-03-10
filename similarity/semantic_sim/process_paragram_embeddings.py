import pickle

from data.data import read_large_file
from similarity.semantic_sim.embedding_model import word_vectors


def get_paragram_embedding(line):
    line = line.split(" ")
    key = line[0]
    embedding = [float(x) for x in line[1:]]
    return key, embedding



if __name__ == "__main__":

    paragram_embeddings = {}

    # Only keep the words available on word2vec trained vocabulary, because the paragram file is huge
    vocab = word_vectors.vocab
    count = 0
    with open("similarity/semantic_sim/data/paragram_300_sl999/paragram_300_sl999.txt", 'r', encoding="utf8", errors='ignore') as f:
        for line in read_large_file(f):
            key, embedding = get_paragram_embedding(line)
            if key.lower() in vocab:
                paragram_embeddings[key.lower()] = embedding

                # Just to follow the computation
                count += 1
                if count % 10000 == 0:
                    print(count)
        

    with open("similarity/semantic_sim/models/paragram_embed_sl999.pkl", "wb") as f:
        pickle.dump(paragram_embeddings, f)