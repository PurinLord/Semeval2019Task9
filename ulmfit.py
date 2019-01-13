import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from fastai.text import * 

path = 'lm_data'

o = pandas.read_csv(
        './Subtask-A/Training_Full_V1.3 .csv',
        names=['id', 'text', 'label'],
        encoding="ISO-8859-1",
        dtype=np.str)

d = o.drop('id', axis=1)
df = pandas.DataFrame({'label': d['label'], 'text': d['text']})

x_train, x_test = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=d['label'])

data_lm = TextLMDataBunch.from_df(path, x_train, x_test)
data_clas = TextClasDataBunch.from_df( path, x_train, x_test, vocab=data_lm.train_ds.vocab, bs=32)

#data_lm.save()
#data_clas.save()
#
#data_lm = TextLMDataBunch.load(path)
#data_clas = TextClasDataBunch.load(path, bs=32)

learn = language_model_learner(data_lm, pretrained_model=URLs.WT103, drop_mult=0.5)
learn.fit_one_cycle(1, 1e-2)

learn.unfreeze()
learn.fit_one_cycle(1, 1e-3)

learn.predict("I like the", n_words=10)

#learn.save_encoder('ft_enc')

learn = text_classifier_learner(data_clas, drop_mult=0.5)
learn.load_encoder('ft_enc')

data_clas.show_batch()

learn.fit_one_cycle(1, 1e-2)

learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))

learn.unfreeze()
learn.fit_one_cycle(1, slice(2e-3/100, 2e-3))

learn.predict("My screen does not load")

