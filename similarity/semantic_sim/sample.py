import numpy as np

import torch

from similarity.semantic_sim.train_neural_net import get_feature_vector
from similarity.semantic_sim.train_deep_network_softmax import FFNN_Softmax


def neural_net_sim(s1, s2):
    x = torch.tensor(get_feature_vector(s1, s2, embedding="paragram")).unsqueeze(0).float()

    output = nn_model(x)[1] # The output before softmax because somehow the softmax in the model always gives a tensor of zeros

    softmax_vec = softmax(output)
    score = 0
    for i, s in enumerate(softmax_vec[0][0]):
        score += i * s.item()
    score /= 5
    return score



softmax = torch.nn.Softmax(dim=2)

nn_model = FFNN_Softmax([300, 100])
nn_model.load_state_dict(torch.load("similarity/semantic_sim/models/ffnn_softmax_ws353.pt"))
nn_model.eval()



