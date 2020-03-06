import pandas as pd
import numpy as np

def read_raw_file(filename, max_score=5):
    """
        Reads the STS training data
        @max_score: the maximum similarity score --> In the datasets, the similarity score is between 0 and 5
    """
    X = []
    Y = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for l in lines:
            sim, sen1, sen2 = l.split("\t")[4:7]
            X.append((sen1.strip('.'), sen2.strip('\n.')))
            Y.append(float(sim)/max_score)
    return X, Y




def write_to_csv(filename, data):
    """
        @data: list of lists containing the computed features from the raw training/test/dev data
        @filename: path to CSV file
    """
    with open(filename, 'w') as f:
        for d in data:
            f.write(",".join([str(x) for x in d]) + "\n")



def read_processed_data(filename):
    """
        Reads the numerical features into a numpy array, and the similarity labels into a Pandas Series
    """
    X = pd.read_csv(filename)

    last_col_name = X.columns[-1]
    Y = X[last_col_name]
    del X[last_col_name]

    X = X.values
    Y = Y.values

    X = np.nan_to_num(X, posinf=np.finfo('float32').max, neginf=np.finfo('float32').min) #Just to remove Nan and inf

    return X, Y