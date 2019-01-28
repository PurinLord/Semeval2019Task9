import numpy as np
import pandas
from fastai.text import * 
from sklearn.metrics import f1_score 

path = 'lm_data_full'

data_lm = TextLMDataBunch.load(path)
data_clas = TextClasDataBunch.load(path, bs=32)

model = text_classifier_learner(data_clas, drop_mult=0.5)
model.load('clas_full')

model.load_encoder('ft_enc_full')

oo = pandas.read_csv('./Subtask-A/SubtaskA_Trial_Test_Labeled.csv', names=['id','text','label'], encoding = "ISO-8859-1")

d = oo.drop('id', axis=1)
df = pandas.DataFrame({'label': d['label'], 'text': d['text']})

#data_lm = TextLMDataBunch.from_df(path, df, df)
#data_clas = TextClasDataBunch.from_df(path, df, df, vocab=data_lm.train_ds.vocab, bs=32)

pred = [model.predict(t)[0].data for t in df.text]

score = f1_score(oo.label[1:].astype(int), pred[1:])

print(score)

