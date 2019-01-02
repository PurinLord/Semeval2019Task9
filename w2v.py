import numpy as np
from string import punctuation
from gensim.models import KeyedVectors


def get_w2v():
    w2v = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
    return w2v


def get_sent_vector(sent, w2v):
    vec = np.zeros(300)
    word_list = sent.split()
    for w in word_list:
        w = w.lower().strip(punctuation)
        try:
            vec += w2v.word_vec(w)
        except:
            pass
    return vec/len(word_list)


def sent_vectors(data):
    w2v = get_w2v()
    out = np.ndarray((len(data), 300))
    for i, sent in enumerate(data):
        out[i] = get_sent_vector(sent, w2v)
    return out

