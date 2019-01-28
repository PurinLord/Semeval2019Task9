from hyperopt import fmin, hp, tpe, Trials, STATUS_OK

import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from fastai.text import * 

def objective(parms):
    data_clas = TextClasDataBunch.from_df( path, x_train, x_test, vocab=data_lm.train_ds.vocab, bs=int(parms['batch_size']))

    learn = language_model_learner(
            data_lm, pretrained_model=URLs.WT103,
            drop_mult=parms['drop_mult'],
            bptt=int(parms['bptt']),
            emb_sz=int(parms['emb_sz']),
            nh=int(parms['nh']),
            nl=int(parms['nl']),
            pad_token=int(parms['pad_token']),
            tie_weights=parms['tie_weights'],
            bias=parms['bias'],
            qrnn=parms['qrnn'],
            pretrained_fnames=None)

    learn.fit_one_cycle(1, 1e-2)

    learn.unfreeze()
    learn.fit_one_cycle(1, 1e-3)

    learn.save_encoder('ft_enc')

    learn = text_classifier_learner(
            data_clas,
            drop_mult=parms['c_drop_mult'],
            bptt=int(parms['c_bptt']),
            emb_sz=int(parms['c_emb_sz']),
            nh=int(parms['c_nh']),
            nl=int(parms['c_nl']),
            pad_token=int(parms['c_pad_token']),
            qrnn=parms['c_qrnn'],
            max_len=int(parms['max_len']),
            lin_ftrs=None,
            ps=None,
            pretrained_model=None)

    learn.load_encoder('ft_enc')

    learn.fit_one_cycle(1, 1e-2)

    learn.freeze_to(-2)
    learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))

    learn.unfreeze()
    learn.fit_one_cycle(1, slice(2e-3/100, 2e-3))

    return {'loss': acc*(-1), 'status': STATUS_OK}


space = {
        'batch_size': hp.quniform(4, 32, q=1),
        'drop_mult': hp.uniform(0.1, 0.9),
        'bptt': hp.quniform(30, 110, q=1),
        'emb_sz': hp.quniform(10, 1000, q=10),
        'nh': hp.quniform(100, 2500, q=10),
        'nl': hp.quniform(1, 20, q=1),
        'pad_token': hp.quniform(1, 5, q=1),
        'tie_weights': hp.choice([True, False]),
        'bias': hp.choice([True, False]),
        'qrnn': hp.choice([True, False]),

        'c_batch_size': hp.quniform(4, 32, q=1),
        'c_drop_mult': hp.uniform(0.1, 0.9),
        'c_bptt': hp.quniform(30, 110, q=1),
        'c_emb_sz': hp.quniform(10, 1000, q=10),
        'c_nh': hp.quniform(100, 2500, q=10),
        'c_nl': hp.quniform(1, 20, q=1),
        'c_pad_token': hp.quniform(1, 5, q=1),
        'c_qrnn': hp.choice([True, False]),
        'max_len': hp.quniform(100, 2000, q=10)
        }


path = 'lm_data'

o = pandas.read_csv(
        './Subtask-A/Training_Full_V1.3 .csv',
        names=['id', 'text', 'label'],
        encoding="ISO-8859-1",
        dtype=np.str)

oo = pandas.read_csv(
        './Subtask-A/SubtaskA_Trial_Test_Labeled.csv',
        names=['id', 'text', 'label'], encoding="ISO-8859-1")

x_train = pandas.DataFrame({'label': o['label'], 'text': o['text']})
x_test = pandas.DataFrame({'label': oo['label'], 'text': oo['text']})

#x_train, x_test = train_test_split(
#        df,
#        test_size=0.20,
#        random_state=42,
#        stratify=d['label'])

data_lm = TextLMDataBunch.from_df(path, x_train, x_test)

trials = Trials()

best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=100)

