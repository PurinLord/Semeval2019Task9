import numpy as np
import itertools
from tqdm import tqdm
import pandas
#import spacy
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
# Naive Bayesian
from sklearn.naive_bayes import GaussianNB
# Random Forest
from sklearn.ensemble import RandomForestClassifier
# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
# Suport Vector Machine
from sklearn.svm import SVC

from sklearn.metrics import f1_score 
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_val_score

from nltk import word_tokenize
from nltk.util import ngrams  

from keras.models import Model 
from keras.layers import Dense, Input, Dropout
from keras.callbacks import EarlyStopping

def word_ngram(corpus, n):
    print('corpus_grams')
    corpus_grams = [list(ngrams(word_tokenize(text), n)) for text in corpus]
    print('set')
    index_ref = list({gram for text in corpus_grams for gram in text})
    out = sparse.csr_matrix((len(corpus_grams), len(index_ref)))

    for i, el in tqdm(enumerate(corpus_grams), total=len(corpus_grams)):
        sub = np.zeros(len(index_ref))
        for gram in el:
            sub[index_ref.index(gram)] += 1
        out[i] = sub

    return out, index_ref


def noun_notnoun(phrase):
    doc = nlp(phrase) # create spacy object
    token_not_noun = []
    notnoun_noun_list = []

    for item in doc:
        if item.pos_ != "NOUN": # separate nouns and not nouns
            token_not_noun.append(item.text)
        if item.pos_ == "NOUN":
            noun = item.text

    for notnoun in token_not_noun:
        notnoun_noun_list.append(notnoun + " " + noun)

    return notnoun_noun_list


def make_input(x, models):
    x_pred = (np.hsplit(m.predict_proba(x), 2)[0] for m in models)
    #x_pred = (m.predict(x) for m in models)
    return np.hstack(x_pred)


def train_meta_model(data, models, dropout=0.3):
    x, xt, y_train, y_test = data
    x_all = make_input(x.toarray(), models)
    
    in_shape = (len(models), )
    m_input = Input(shape=in_shape)
    m = Dropout(dropout)(m_input)
    m = Dense(10)(m)
    #m = Dropout(0.2)(m)
    m = Dense(2)(m)
    #m = Dropout(0.2)(m)
    m = Dense(1)(m)
    model = Model(inputs=m_input, outputs=m)
    
    model.compile('sgd', loss='mse', metrics=['acc'])
    
    model.fit(x_all, y_train,
            validation_split=0.2,
            epochs=100,
            callbacks=[EarlyStopping()])
    
    all_pred = make_input(xt.toarray(), models)
    
    final_pred = model.predict(all_pred)
    one_hot = np.zeros(len(final_pred))
    for i, p in enumerate(final_pred):
        if p > 0.5:
            one_hot[i] = 1
    
    print('Final ', f1_score(one_hot, y_test))
    
    return model, final_pred

min_len = 500
max_len = 10000

o = pandas.read_csv('./Subtask-A/Training_Full_V1.3 _predictions.csv', names=['id','x','y'], encoding = "ISO-8859-1")
oo = pandas.read_csv('./Subtask-A/SubtaskA_Trial_Test_Labeled.csv', names=['id','x','y'], encoding = "ISO-8859-1")

X = o['x']
Y = o['y']

#bigrams = word_ngram(X, 2) 

#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=420)
#x_train, y_train = X, Y
#x_test = oo['x']
#y_test = [0]*len(x_test)

#b_train, b_test = train_test_split(bigrams, test_size=0.33, random_state=42)

x_train = o['x'][1:]
y_train = o['y'][1:].astype(int)
x_test = oo['x'][1:]
y_test = oo['y'][1:].astype(int)

tfidf_vect = TfidfVectorizer()
tfidf_vect.fit(x_train)
x = tfidf_vect.transform(x_train)
xt = tfidf_vect.transform(x_test)

# SVM
svm = SVC(C=10000, probability=True)
svm.fit(x, y_train)
svm_pred = svm.predict(xt)
print('SVM ', f1_score(y_test, svm_pred))

# Gauss
gnv = GaussianNB()
gnv.fit(x.toarray(), y_train)
g_pred = gnv.predict(xt.toarray())
print('GNB ', f1_score(y_test, g_pred))

# Random F
rfc = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
rfc.fit(x, y_train)
#print(rfc.feature_importances_)
rfc_pred = rfc.predict(xt)
print('RFC ', f1_score(y_test, rfc_pred))

# DTC
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(x, y_train)
dtc_pred = dtc.predict(xt)
print('DTC', f1_score(y_test, dtc_pred))

# Logistic REgretion

model, predic = train_meta_model((x, xt, y_train, y_test), [svm, dtc, rfc, gnv], dropout=0.3)
