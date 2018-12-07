import numpy as np
from tqdm import tqdm
import pandas
import spacy
from scipy import sparse
from collections import Counter
import re

import nltk
from nltk.tokenize import word_tokenize

# Random Forest
from sklearn.ensemble import RandomForestClassifier
# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
# Suport Vector Machine
from sklearn.svm import SVC

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score 
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_val_score

from nltk import word_tokenize
from nltk.util import ngrams  

from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.callbacks import EarlyStopping

from losses import focal


keywords = ["suggest", "recommend", "hopefully", "go for", "request", "it would be nice", "adding", "should come with", "should be able", "could come with",  "i need" ,  "we need", "needs",  "would like to", "would love to", "allow", "add"]

# Goldberg et al.
pattern_strings = [r'.*would\slike.*if.*', r'.*i\swish.*', r'.*i\shope.*', r'.*i\swant.*', r'.*hopefully.*',
                   r".*if\sonly.*", r".*would\sbe\sbetter\sif.*", r".*should.*", r".*would\sthat.*",
                   r".*can't\sbelieve.*didn't.*", r".*don't\sbelieve.*didn't.*", r".*do\swant.*", r".*i\scan\shas.*"]


def clean_text(text):
    def alfa(l):
        return ''.join([c for c in l if c.isalpha()])
    c = [alfa(l.lower()) for l in text.split() ]
    return c


def keyword_vector(corpus):

    bow = []
    for line in corpus:
        clean = clean_text(line)

        count = Counter(clean)

        word_vec = [count[w] for w in keywords]

        bow.append(word_vec)

    return bow


def patern_vector(corpus):
    compiled_patterns = []
    for patt in pattern_strings:
        compiled_patterns.append(re.compile(patt))
    out = []
    for line in corpus:
        joined_sent = " ".join(clean_text(line))
        pat_found = [len(pat.findall(joined_sent)) > 0 for pat in compiled_patterns]
        out.append(pat_found)
    return out


def tag_vector(corpus):
    out = []
    for line in corpus:
        tokenized_sent = word_tokenize(line)
        tagged_sent = nltk.pos_tag(tokenized_sent)
        tags = [i[1] for i in tagged_sent]
        pos_match = [t in tags for t in ['MD', 'VB']]
        out.append(pos_match)
    return out


def make_input(corpus):
    X_key = keyword_vector(corpus)
    X_pat = patern_vector(corpus)
    X_tag = tag_vector(corpus)
    X_in = np.hstack((X_key, X_pat, X_tag))
    return X_in


def make_model(x_train):
    def f1(y_true, y_pred):
        import keras.backend as K
        def recall(y_true, y_pred):
            """Recall metric.

            Only computes a batch-wise average of recall.

            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision(y_true, y_pred):
            """Precision metric.

            Only computes a batch-wise average of precision.

            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision
        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

    in_shape = (len(x_train[0]), )
    m_input = Input(shape=in_shape)
    m = Dropout(0.0)(m_input)
    m = Dense(80, activation='relu')(m)
    m = Dropout(0.75)(m)
    m = Dense(50, activation='relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(2, activation='softmax')(m)
    model = Model(inputs=m_input, outputs=m)
    loss = focal()
    model.compile(optimizer='rmsprop', loss=loss, metrics=['acc', f1])
    return model


def train_eval(model, X, Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    enc = OneHotEncoder()
    enc.fit(y_train)

    model.fit(x_train, enc.transform(y_train),
            validation_split=0.2,
            epochs=10,
            callbacks=[EarlyStopping(patience=4)])

    final_pred = model.predict(x_test)
    no_one_hot = enc.inverse_transform(final_pred).reshape(len(final_pred))

    print('Final ', f1_score(no_one_hot, y_test))

    return enc


if __name__ == '__main__':
    o = pandas.read_csv('./Subtask-A/Training_Full_V1.3 .csv', names=['id','x','y'], encoding = "ISO-8859-1")
    #oo = pandas.read_csv('./Subtask-A/TrialData_SubtaskA_Test.csv', names=['id','x','y'], encoding = "ISO-8859-1")

    X = o['x']
    Y = o['y']

    bigram = CountVectorizer(ngram_range=(1,3))
    X_in = bigram.fit_transform(X)
    #X_in = make_input(X)
    #tfidf_vect = TfidfVectorizer()
    #X_in = tfidf_vect.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(X_in, Y, test_size=0.33, random_state=420)

    #x_train = np.array(x_train)
    #x_test = np.array(x_test)
    x_train = x_train.toarray()
    x_test = x_test.toarray()
    y_train = np.array(y_train).reshape(len(y_train), 1)
    y_test = np.array(y_test).reshape(len(y_test), 1)

    #svm = SVC(probability=True)
    #svm.fit(x_train, y_train)
    #svm_pred = svm.predict(x_test)
    #print('SVM ', f1_score(y_test, svm_pred))


    ## Random F
    #rfc = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    #rfc.fit(x_train, y_train)
    ##print(rfc.feature_importances_)
    #rfc_pred = rfc.predict(x_test)
    #print('RFC ', f1_score(y_test, rfc_pred))

    ## DTC
    #dtc = DecisionTreeClassifier(random_state=5)
    #dtc.fit(x_train, y_train)
    #dtc_pred = dtc.predict(x_test)
    #print('DTC', f1_score(y_test, dtc_pred))

    model = make_model(x_train)
    Y_in = np.array(Y).reshape(len(Y), 1)
    enc = train_eval(model, X_in, Y_in)

