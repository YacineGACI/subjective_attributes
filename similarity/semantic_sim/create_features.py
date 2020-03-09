import similarity.semantic_sim.features as features
from similarity.semantic_sim.data import read_raw_file, write_to_csv
from similarity.semantic_sim.embedding_model import *



def get_features(s1, s2):
    u = phrase_embedding(s1)
    v = phrase_embedding(s2)

    new_features = []
    new_features.append(features.bleu_score(s1, s2))
    new_features.append(features.levenshtein_distance(s1, s2))
    new_features.append(features.jaccard_distance_word(s1, s2))
    new_features.append(features.jaccard_distance_char(s1, s2))
    new_features.append(features.ngram_overlap(s1, s2, 1))
    new_features.append(features.ngram_overlap(s1, s2, 2))
    new_features.append(features.ngram_overlap(s1, s2, 3))
    new_features.append(features.ngram_overlap(s1, s2, 4))
    new_features.append(features.dice_index(s1, s2))
    new_features.append(features.overlap_index(s1, s2))

    new_features.append(features.cosine_sim(u, v))
    new_features.append(features.euclidean_distance(u, v))
    new_features.append(features.manhattan_distance(u, v))

    new_features.append(features.pearson_correlation(u, v))
    new_features.append(features.spearman_correlation(u, v))
    new_features.append(features.kendall_tau(u, v))

    new_features.append(features.kl_divergence(u, v))
    new_features.append(features.mutual_info(u, v))

    new_features.append(features.kernel(u, v, metric="laplacian"))
    new_features.append(features.kernel(u, v, metric="sigmoid"))
    new_features.append(features.kernel(u, v, metric="rbf"))
    new_features.append(features.kernel(u, v, metric="polynomial"))

    return new_features





if __name__ == "__main__":

    Xtrain, Ytrain = read_raw_file("similarity/semantic_sim/data/raw/sts-train.csv")
    Xtest, Ytest= read_raw_file("similarity/semantic_sim/data/raw/sts-test.csv")
    Xdev, Ydev = read_raw_file("similarity/semantic_sim/data/raw/sts-dev.csv")


    train_processed = []
    test_processed = []
    dev_processed = []

    for i, x in enumerate(Xtrain):
        train_processed.append(get_features(x[0], x[1]) + [Ytrain[i]])
    print("Finished processing the train dataset")

    for i, x in enumerate(Xtest):
        test_processed.append(get_features(x[0], x[1]) + [Ytest[i]])
    print("Finished processing the test dataset")

    for i, x in enumerate(Xdev):
        dev_processed.append(get_features(x[0], x[1]) + [Ydev[i]])
    print("Finished processing the dev dataset")

    write_to_csv("similarity/semantic_sim/data/processed/train.csv", train_processed)
    write_to_csv("similarity/semantic_sim/data/processed/test.csv", test_processed)
    write_to_csv("similarity/semantic_sim/data/processed/dev.csv", dev_processed)
