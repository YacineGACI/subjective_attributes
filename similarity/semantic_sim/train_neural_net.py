from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error # For evaluating the trained model
import pickle
import numpy as np

from similarity.semantic_sim.data import read_raw_file
from similarity.semantic_sim.embedding_model import *



def get_feature_vector(s1, s2):
    v1 = phrase_embedding(s1).reshape(1, -1)
    v2 = phrase_embedding(s2).reshape(1, -1)

    diff = np.abs(v1 - v2)
    hadamard = np.multiply(v1, v2)

    return np.concatenate((v1, v2, diff, hadamard), axis=1)



if __name__ == "__main__":
    
    Xtrain, Ytrain = read_raw_file("similarity/semantic_sim/data/raw/sts-train.csv")
    Xtest, Ytest= read_raw_file("similarity/semantic_sim/data/raw/sts-test.csv")

    X_train_input = np.concatenate([get_feature_vector(x[0], x[1]) for x in Xtrain])
    Y_train_input = np.array(Ytrain)

    X_test_input = np.concatenate([get_feature_vector(x[0], x[1]) for x in Xtest])
    Y_test_input = np.array(Ytest)

    print("Dataset processed")

    nn_model = MLPRegressor(hidden_layer_sizes=(300, 200, 100, 50), activation="relu", solver="adam", learning_rate="adaptive", learning_rate_init=0.001, max_iter=500, tol=0.000001)
    nn_model.fit(X_train_input, Y_train_input)

    print("Training complete")

    # Saving the trained models
    with open("similarity/semantic_sim/models/nn_model.pkl", "wb") as f:
        pickle.dump(nn_model, f)

    # Evaluate the trained model
    y_pred_nn = nn_model.predict(X_test_input)

    print("MSE --> ", mean_squared_error(Y_test_input, y_pred_nn))
    print("MAE --> ", mean_absolute_error(Y_test_input, y_pred_nn))
