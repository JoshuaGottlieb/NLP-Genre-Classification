import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from gensim.models import nmf, ldamodel
import gensim.models.phrases
from gensim.matutils import corpus2dense, corpus2csc, Sparse2Corpus
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
import scipy
from scipy.sparse import csr_matrix
from joblib import parallel_backend

def sparse_gensim_to_dataframe(sparse_bow, dictionary, index, documents_columns = True):
    frame_bow = pd.DataFrame(corpus2dense(Sparse2Corpus(sparse_bow, documents_columns = documents_columns),
                                          num_terms = len(dictionary.token2id)).T,
                             columns = list(dictionary.values()), index = index)
    
    return frame_bow

def create_gensim_dictionaries(X_train, X_test, no_below = 5, no_above = 0.95, n_gram_depth = 2):
    texts = []
    phrasers = []
    corpora = []
    dictionaries = []
    X_train_bows_sparse = []
    X_test_bows_sparse = []
    
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()
    
    for i in range(2, n_gram_depth + 1):
        with parallel_backend('loky', n_jobs = -1):
            text = list(X_train_copy.lyrics.apply(lambda x: list(x)).values)
            phraser = Phrases(text)
            phrasers.append(phraser)
            X_train_copy.lyrics = X_train_copy.lyrics.apply(lambda x: phraser[x])
            text = list(X_train_copy.lyrics.apply(lambda x: list(x)).values)
            texts.append(text)
            
            dictionary = Dictionary(text)
            dictionary.filter_extremes(no_below = no_below, no_above = no_above)
            dictionaries.append(dictionary)
            
            corpus = [dictionary.doc2bow(x) for x in text]
            corpora.append(corpus)

    return texts, phrasers, corpora, dictionaries

def obtain_gensim_sparse(X_train, X_test, text, phraser, corpus, dictionary):
    X_train_copy_ngrams = X_train.copy().lyrics.apply(lambda x: phraser[x])
    X_train_copy_text = list(X_train_copy_ngrams.apply(lambda x: list(x)).values)
    X_train_copy_corpus = [dictionary.doc2bow(x) for x in X_train_copy_text]
    X_train_sparse = corpus2csc(X_train_copy_corpus, num_terms = len(dictionary.token2id))
    
    X_test_copy_ngrams = X_test.copy().lyrics.apply(lambda x: phraser[x])
    X_test_copy_text = list(X_test_copy_ngrams.apply(lambda x: list(x)).values)
    X_test_copy_corpus = [dictionary.doc2bow(x) for x in X_test_copy_text]
    X_test_sparse = corpus2csc(X_test_copy_corpus, num_terms = len(dictionary.token2id))
    
    
    return X_train_sparse, X_test_sparse

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