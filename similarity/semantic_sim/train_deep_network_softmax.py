import random

import torch 
import torch.nn as nn
import numpy as np

from similarity.semantic_sim.train_neural_net import get_feature_vector
from similarity.semantic_sim.data import read_raw_file

class FFNN_Softmax(nn.Module):
    def __init__(self, hidden_sizes, input_dim=600, num_outputs=6, dropout=0.1):
        super(FFNN_Softmax, self).__init__()
        hidden_sizes = [input_dim] + hidden_sizes + [num_outputs]
        self.layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes) - 1)])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        output = x
        for i in range(len(self.layers) - 1):
            output = self.relu(self.dropout(self.layers[i](output)))
        
        output_before_softmax = self.layers[-1](output)
        return self.softmax(output_before_softmax), output_before_softmax




def train(input, target):
    model.train()
    model.zero_grad()
    output = model(input.float())[0]
    loss = criterion(output, target)

    loss.backward()
    optimizer.step()

    return loss.item()


def test(input, target):
    model.eval()
    output = model(input.float())[0]
    loss = criterion(output, target)
    return loss.item()




if __name__ == "__main__":

    Xtrain, Ytrain = read_raw_file("similarity/semantic_sim/data/raw/sts-train.csv")
    Xtest, Ytest= read_raw_file("similarity/semantic_sim/data/raw/sts-test.csv")

    Ytrain = [round(x * 5) for x in Ytrain]
    Ytest = [round(x * 5) for x in Ytest]

    X_train_input = np.concatenate([get_feature_vector(x[0], x[1], embedding="paragram") for x in Xtrain])
    Y_train_input = np.array(Ytrain)

    X_test_input = np.concatenate([get_feature_vector(x[0], x[1], embedding="paragram") for x in Xtest])
    Y_test_input = np.array(Ytest)

    X_train_input = torch.tensor(X_train_input)
    Y_train_input = torch.tensor(Y_train_input)
    X_test_input = torch.tensor(X_test_input)
    Y_test_input = torch.tensor(Y_test_input)

    print("Dataset processed")

    learning_rate = 0.0001
    n_epochs = 30000
    minibatch_size = 200
    weight_decay = 0.0004
    dropout = 0.2
    print_every = 300

    model = FFNN_Softmax([300, 100], dropout=dropout)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)



    train_loss = 0
    test_loss = 0

    model.train()

    for epoch in range(n_epochs):
        boundary = random.randint(0, X_train_input.shape[0] - minibatch_size - 1)
        input = X_train_input[boundary: boundary + minibatch_size]
        target = Y_train_input[boundary: boundary + minibatch_size]

        train_loss += train(input, target)



        boundary = random.randint(0, X_test_input.shape[0] - minibatch_size - 1)
        input = X_test_input[boundary: boundary + minibatch_size]
        target = Y_test_input[boundary: boundary + minibatch_size]

        test_loss += test(input, target)

        if (epoch + 1) % print_every == 0:
            print("Training {}% --> Training Loss = {}".format(round(((epoch + 1) / n_epochs) * 100, 2), train_loss/print_every))
            print("Training {}% --> Evaluation Loss = {}".format(round(((epoch + 1) / n_epochs) * 100, 2), test_loss/print_every))
            print()
            train_loss = 0
            test_loss = 0

    print("Training Complete")
    torch.save(model.state_dict(), "similarity/semantic_sim/models/ffnn_softmax_ws353.pt")
    print("Model saved")