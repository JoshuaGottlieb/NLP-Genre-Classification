import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from gensim.models import nmf, ldamodel
import gensim.models.phrases
from gensim.matutils import corpus2dense
from gensim.corpora.dictionary import Dictionary
from gensim.models.phrases import Phrases
from gensim.models import Nmf
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from sklearn.model_selection import train_test_split
import pyLDAvis
import pyLDAvis.gensim_models
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
from joblib import parallel_backend

def obtain_gensim_bag_of_words(X_train, X_test, no_below = 5, no_above = 0.95):
    with parallel_backend('loky', n_jobs = -1):
        X_train_corpus = list(X_train.lyrics.apply(lambda x: list(x)).values)
        bigram_model = Phrases(X_train_corpus)
        X_train['bigrams'] = X_train.apply(lambda x: bigram_model[x])
        X_train_bigram_corpus = list(X_train.bigrams.apply(lambda x: list(x)).values)
        dct = Dictionary(X_train_bigram_corpus)
        dct.filter_extremes(no_below = no_below, no_above = no_above)
        X_train_bigram_corpus_tokenized = [dct.doc2bow(x) for x in X_train_bigram_corpus]
        X_train_bow = pd.DataFrame(corpus2dense(X_train_bigram_corpus_tokenized,
                                                num_terms = len(dct.token2id)).T,
                                   columns = list(dct.values()), index = X_train.index)
        
        X_test['bigrams'] = X_test.apply(lambda x: bigram_model[x])
        X_test_bigram_corpus = list(X_test.bigrams.apply(lambda x: list(x)).values)
        X_test_bigram_corpus_tokenized = [dct.doc2bow(x) for x in X_test_bigram_corpus]
        X_test_bow = pd.DataFrame(corpus2dense(X_test_bigram_corpus_tokenized,
                                               num_terms = len(dct.token2id)).T,
                                  columns = list(dct.values()), index = X_test.index)
        
    return X_train_bow, X_test_bow, X_train_bigram_corpus_tokenized, X_train_bigram_corpus, dct

def obtain_sklearn_bag_of_words(X_train, X_test, vectorizer, vectorizer_args = None):
    vtr = vectorizer
    if vectorizer_args is not None:
        vtr.set_params(**vectorizer_args)
    
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()
    
    X_train_copy.lyrics = X_train_copy.lyrics.str.join(' ')
    X_test_copy.lyrics = X_test_copy.lyrics.str.join(' ')
    
    with parallel_backend('threading', n_jobs = -1):
        X_train_copy = pd.DataFrame(vtr.fit_transform(X_train_copy.lyrics).todense(),
                               columns = vtr.get_feature_names_out().tolist(),
                               index = X_train_copy.index)
        X_test_copy = pd.DataFrame(vtr.transform(X_test_copy.lyrics).todense(),
                               columns = vtr.get_feature_names_out().tolist(),
                               index = X_test_copy.index)
        
    return X_train_copy, X_test_copy

def create_pyLDAvis(model, corpus, dictionary, plot = False, save = False, save_dest = None):
    with parallel_backend('loky', n_jobs = -1):
        p = pyLDAvis.gensim_models.prepare(model, corpus, dictionary)

    if plot:
        pyLDAvis.enable_notebook()
        pyLDAvis.display(p)
    
    if save:
        if save_dest is None:
            print('No save destination specified, pyLDAvis object not saved.')
        else:
            pyLDAvis.save_html(p, save_dest)
            
    return p

def perform_tsne_analysis(X_train, X_test, y_train, y_test, random_state = 42, learning_rate = 100,
                          pickle_ = False, pickle_dest = None, plot = False):
    
    tsne = TSNE(random_state = random_state, learning_rate = learning_rate)
    
    with parallel_backend('loky', n_jobs = -1):
        tsne_trans = tsne.fit_transform(pd.concat([X_train, X_test]))
    
    tsne_trans = pd.DataFrame(tsne_trans, columns = ['TSNE1', 'TSNE2'])
    tsne_trans.index = X_train.index.values.tolist() + X_test.index.values.tolist()
    tsne_trans['genre'] = y_train.values.tolist() + y_test.values.tolist()
    
    if plot:
        plt.figure(figsize=(10,10))
        sns.scatterplot(x = 'TSNE1', y = 'TSNE2', hue = 'genre', data = tsne_trans, palette = 'tab10')
        plt.title('Visualization of Genres')
        plt.show()
        
    if pickle_:
            if pickle_dest is None:
                print('No pickle destination given, pickling skipped.')
            else:
                with open(pickle_dest, 'wb') as f:
                    pickle.dump(tsne_trans, f)
                    
    return tsne_trans