import numpy as np
import pandas

import nltk
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

from w2v import sent_vectors
from select_words import make_model, train_eval, make_input

if __name__ == '__main__':
    o = pandas.read_csv('./Subtask-A/Training_Full_V1.3 .csv', names=['id','x','y'], encoding = "ISO-8859-1")

    X = o['x']
    Y = o['y']
    Y_in = np.array(Y).reshape(len(Y), 1)

    caract = make_input(X)
    tfidf_vect = TfidfVectorizer()
    tf = tfidf_vect.fit_transform(X).toarray()
    w2v = sent_vectors(X)
    X_in = np.hstack((caract, tf, w2v))

    model = make_model(X_in)
    enc = train_eval(model, X_in, Y_in, final=10)

    oo = pandas.read_csv('./Subtask-A/TrialData_SubtaskA_Test.csv', names=['id','x','y'], encoding = "ISO-8859-1")
    X = oo['x']
    caract = make_input(X)
    tf = tfidf_vect.transform(X).toarray()
    w2v = sent_vectors(X)
    X_in = np.hstack((caract, tf, w2v))

    pred = enc.inverse_transform(model.predict(X_in))

    sub = open('submission.csv', 'w')
    data = open('./Subtask-A/TrialData_SubtaskA_Test.csv', 'r')

    for l, p in zip(data, pred):
        sub.write(l.replace('X', str(int(p[0]))))
    sub.close()
    data.close()

