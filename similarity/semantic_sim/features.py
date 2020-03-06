import nltk
from nltk.metrics import edit_distance
from nltk.metrics import jaccard_distance
from nltk.translate.bleu_score import modified_precision, SmoothingFunction
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from scipy.stats import pearsonr, spearmanr, kendalltau, entropy
from sklearn.metrics import mutual_info_score
from sklearn.metrics.pairwise import sigmoid_kernel, pairwise_kernels



##############  n-gram distance features  ##############

def bleu_score(s1, s2):
    hypothesis = s2.split(' ')
    reference = s1.split(' ')
    cc = SmoothingFunction()
    return nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, smoothing_function=cc.method4)


def levenshtein_distance(s1, s2):
    return edit_distance(s1, s2)


def jaccard_distance_word(s1, s2):
    w1 = set(s1.split(" "))
    w2 = set(s2.split(" "))
    return jaccard_distance(w1, w2)


def jaccard_distance_char(s1, s2):
    w1 = set(s1)
    w2 = set(s2)
    return jaccard_distance(w1, w2)


def ngram_overlap(s1, s2, n=1):
    w1 = s1.split(" ")
    w2 = s2.split(" ")
    return float(modified_precision([w1], w2, n))


def dice_index(s1, s2): 
    a = set(s1.split()) 
    b = set(s2.split())
    c = a.intersection(b)
    return 2*float(len(c)) / (len(a) + len(b))


def overlap_index(s1, s2): 
    a = set(s1.split()) 
    b = set(s2.split())
    c = a.intersection(b)
    return float(len(c)) / min(len(a) , len(b) )

########################################################




##############  linear kernel features  ##############

def cosine_sim(u, v):
    u = u.reshape(1, u.shape[0])
    v = v.reshape(1, v.shape[0])
    return cosine_similarity(u, v)[0][0]


def euclidean_distance(u, v):
    return np.sqrt(np.dot(u-v, u-v))


def manhattan_distance(u, v):
    return distance.cityblock(u, v)


########################################################



##############  statistical measures features  ##############

def pearson_correlation(u, v):
    return pearsonr(u, v)[0]


def spearman_correlation(u, v):
    return spearmanr(u, v)[0]


def kendall_tau(u, v):
    return kendalltau(u, v)[0]


########################################################



##############  probabilistic measures features  ##############

def kl_divergence(u, v):
    return entropy(u, v, base=2)


def mutual_info(u, v):
    return mutual_info_score(u, v)


########################################################


##############  kernel features  ##############

def sigmoid_kernel(u, v):
    return sigmoid_kernel(np.array(u).reshape(1, -1), np.array(v).reshape(1, -1))[0][0]


def kernel(u, v, metric="linear"):
    return pairwise_kernels(np.array(u).reshape(1, -1), np.array(v).reshape(1, -1), metric=metric)[0][0]