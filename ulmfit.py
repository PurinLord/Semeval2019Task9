import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from fastai.text import * 
from fastai.metrics import fbeta


class fbeta_binary(Callback):
    "Computes the f_beta between preds and targets for binary text classification"

    def __init__(self, beta2 = 1, eps=1e-9,sigmoid = True):      
        self.beta2=beta2**2
        self.eps = eps
        self.sigmoid = sigmoid
    
    def on_epoch_begin(self, **kwargs):
        self.TP = 0
        self.total_y_pred = 0   
        self.total_y_true = 0
    
    def on_batch_end(self, last_output, last_target, **kwargs):
        y_pred = last_output
        y_pred = y_pred.softmax(dim = 1)        
        y_pred = y_pred.argmax(dim=1)
        y_true = last_target.float()
        
        self.TP += ((y_pred==1) * (y_true==1)).float().sum()
        self.total_y_pred += (y_pred==1).float().sum()
        self.total_y_true += (y_true==1).float().sum()
    
    def on_epoch_end(self, **kwargs):
        prec = self.TP/(self.total_y_pred+self.eps)
        rec = self.TP/(self.total_y_true+self.eps)
        res = (prec*rec)/(prec*self.beta2+rec+self.eps)*(1+self.beta2)        
        #self.metric = res.mean()
        self.metric = res     

path = 'lm_data'

o = pandas.read_csv(
        './Subtask-A/V1.4_Training.csv',
        names=['id', 'text', 'label'],
        encoding="ISO-8859-1",
        dtype=np.str)

oo = pandas.read_csv(
        './Subtask-A/SubtaskA_Trial_Test_Labeled.csv',
        names=['id', 'text', 'label'],
        encoding="ISO-8859-1",
        dtype=np.str)

x_train = pandas.DataFrame({'label': o['label'][1:], 'text': o['text'][1:]})
x_test = pandas.DataFrame({'label': oo['label'][1:], 'text': oo['text'][1:]})

#x_train, x_test = train_test_split(
#        df,
#        test_size=0.20,
#        random_state=42,
#        stratify=d['label'])

data_lm = TextLMDataBunch.from_df(path, x_train, x_test)
data_clas = TextClasDataBunch.from_df( path, x_train, x_test, vocab=data_lm.train_ds.vocab, bs=200)

data_lm.save()
data_clas.save()

data_lm = TextLMDataBunch.load(path)
data_clas = TextClasDataBunch.load(path, bs=200)

learn = language_model_learner(
        data_lm, pretrained_model=URLs.WT103,
        drop_mult=0.5,
        bptt=70,
        emb_sz=400,
        nh=1150,
        nl=3,
        pad_token=1,
        tie_weights=True,
        bias=True,
        qrnn=False,
        pretrained_fnames=None)

learn.fit_one_cycle(1, 1e-2)

learn.unfreeze()
learn.fit_one_cycle(1, 1e-3)

learn.predict("I like the", n_words=10)

learn.save_encoder('ft_enc')

learn = text_classifier_learner(
        data_clas,
        drop_mult=0.5,
        bptt=70,
        emb_sz=400,
        nh=1150,
        nl=3,
        pad_token=1,
        qrnn=False,
        max_len=1400,
        lin_ftrs=None,
        ps=None)
fbeta_binary = fbeta_binary()  # default is F1
learn.metrics.append(fbeta_binary) 

learn.load_encoder('ft_enc')

data_clas.show_batch()

learn.fit_one_cycle(1, 1e-2)

learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))

learn.unfreeze()
learn.fit_one_cycle(1, slice(2e-3/100, 2e-3))

learn.predict("My screen does not load")

learn.save('clas')

