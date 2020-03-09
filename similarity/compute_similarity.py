import pickle
import numpy as np

from similarity.conceptual_sim.sample import compute_conceptual_similarity
from similarity.semantic_sim.embedding_model import phrase_embedding
from similarity.semantic_sim.features import cosine_sim
from similarity.semantic_sim.train_neural_net import get_feature_vector
from similarity.semantic_sim.create_features import get_features


def similarity(s1, s2):
    conceptual = compute_conceptual_similarity(s1, s2)
    print(conceptual)

    cosine = cosine_sim(phrase_embedding(s1), phrase_embedding(s2))
    print(cosine)

    nn_input = get_feature_vector(s1, s2)
    nn_sim = nn_model.predict(nn_input)[0]
    print(nn_sim)

    rf_input = get_features(s1, s2)
    rf_input = np.nan_to_num(rf_input, posinf=np.finfo('float32').max, neginf=np.finfo('float32').min).reshape(1, -1)
    rf_sim = rf_model.predict(rf_input)[0]
    print(rf_sim)

    return (1/4) * (conceptual + cosine + nn_sim + rf_sim)



with open("similarity/semantic_sim/models/rf_model.pkl", 'rb') as f:
    rf_model = pickle.load(f)

with open("similarity/semantic_sim/models/nn_model.pkl", 'rb') as f:
    nn_model = pickle.load(f)
