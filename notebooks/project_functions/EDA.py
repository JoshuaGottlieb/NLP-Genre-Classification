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
    '''
    Takes in a sparse bag-of-words matrix, a trained dictionary, and an index.
    Returns a Pandas dataframe of the underlying dense matrix with appropriate token labels and indices.
    
    sparse_bow: Sparse matrix representing a gensim corpus2csc object.
    dictionary: Trained gensim dictionary used to create the sparse_bow.
    index: Pandas index object or list of numbers to apply to the generated dataframe,
            in order to match train-test split data, for example.
    document_columns: Optional bool, True denotes that the documents lie in the columns of the matrix,
                        False indicates that the documents lie in the rows of the matrix. Default True.
    '''
    
    # Converts the sparse matrix to a gensim corpus, then converts the gensim corpus to a dense matrix
    # Creates a dataframe using dense matrix representation of the gensim corpus
    frame_bow = pd.DataFrame(corpus2dense(Sparse2Corpus(sparse_bow, documents_columns = documents_columns),
                                          num_terms = len(dictionary.token2id)).T,
                             columns = list(dictionary.values()), index = index)
    
    return frame_bow

def create_gensim_dictionaries(X_train, no_below = 5, no_above = 0.95, n_gram_depth = 2):
    '''
    Creates the necessary gensim objects to encode data in an n-gram fashion.
    Returns a list of texts (lists of lists of strings), a list of trained phrasers (gensim Phrase objects),
    a list of corpora (lists of doc2bow transformed texts), and a list of trained dictionaries (gensim Dictionary objects).
    
    X_train: Training data to use to train gensim objects.
    no_below: Optional integer denoting the minimum number of documents a token must appear in be used. Default 5.
    no_above: Optional float denoting the maximum percentage of documents a token can appear in and be used. Default 0.95.
    n_gram_depth: Integer greater than or equal to 2 denoting what level of n-gram encoding to be computed. Default 2.
    '''
    
    # Initialize lists
    texts = []
    phrasers = []
    corpora = []
    dictionaries = []
    
    X_train_copy = X_train.copy()
    
    # Loop through to build up to desired n-gram encoding level
    for i in range(2, n_gram_depth + 1):
        with parallel_backend('loky', n_jobs = -1):
            # Draw current n-gram depth text level to train Phrases object
            text = list(X_train_copy.lyrics.apply(lambda x: list(x)).values)
            # Train Phrases object to achieve i-gram encoding
            phraser = Phrases(text)
            phrasers.append(phraser)
            # Update n-gram depth text level
            X_train_copy.lyrics = X_train_copy.lyrics.apply(lambda x: phraser[x])
            text = list(X_train_copy.lyrics.apply(lambda x: list(x)).values)
            texts.append(text)
            # Train dictionary on current i-gram encoding
            dictionary = Dictionary(text)
            dictionary.filter_extremes(no_below = no_below, no_above = no_above)
            dictionaries.append(dictionary)
            # Create a bag-of-words representation for the corpus at i-gram encoding
            corpus = [dictionary.doc2bow(x) for x in text]
            corpora.append(corpus)

    return texts, phrasers, corpora, dictionaries

def obtain_gensim_sparse(X_train, X_test, phraser, dictionary):
    '''
    Takes in a train set, test set, gensim Phrase, and gensim Dictionary to encode data as a sparse bag-of-words matrix.
    Returns sparse X_train and sparse X_test matrices.
    
    X_train: Train set data that was used to train the phraser, corpus and dictionary.
    X_test: Test set data to be transformed.
    phraser: Trained gensim Phrase object, trained on X_train to create n-gram encoding.
    dictionary: Trained gensim Dictionary object, trained on X_train to create bag-of-words representations.
    '''
    
    # Apply phraser to lyrics column to obtain n-gram representation
    # Convert to list of list of strings, then use the dictionary to create a bag-of-words representation
    # Use corpus2csc to convert the bag-of-words representation into a sparse compressed column matrix
    X_train_copy_ngrams = X_train.copy().lyrics.apply(lambda x: phraser[x])
    X_train_copy_text = list(X_train_copy_ngrams.apply(lambda x: list(x)).values)
    X_train_copy_corpus = [dictionary.doc2bow(x) for x in X_train_copy_text]
    X_train_sparse = corpus2csc(X_train_copy_corpus, num_terms = len(dictionary.token2id))
    
    X_test_copy_ngrams = X_test.copy().lyrics.apply(lambda x: phraser[x])
    X_test_copy_text = list(X_test_copy_ngrams.apply(lambda x: list(x)).values)
    X_test_copy_corpus = [dictionary.doc2bow(x) for x in X_test_copy_text]
    X_test_sparse = corpus2csc(X_test_copy_corpus, num_terms = len(dictionary.token2id))
    
    
    return X_train_sparse, X_test_sparse

def obtain_sklearn_bag_of_words(X_train, X_test, vectorizer, vectorizer_args = None, to_frame = True):
    '''
    Takes in a train set, test set, and scikit-learn vectorizer and returns vectorized versions of the train and test sets.
    
    X_train: Train set data.
    X_test: Test set data.
    vectorizer: scikit-learn vectorizer, currently supported are CountVectorizer and TfidfVectorizer.
    vectorizer_args: Optional dictionary of hyperparameters to pass into the vectorizer object.
    to_frame: True means the returned objects will be converted to dense matrices and passed into Pandas dataframes.
              False means the returned objects will be sparse matrices. Default True.
    '''
    
    # Initialize the vectorizer and set hyperparameters if necessary
    vtr = vectorizer
    if vectorizer_args is not None:
        vtr.set_params(**vectorizer_args)
    
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()
    
    # Convert lyrics from lists to strings
    X_train_copy.lyrics = X_train_copy.lyrics.str.join(' ')
    X_test_copy.lyrics = X_test_copy.lyrics.str.join(' ')
    
    # Fit vectorizer on train set, transform on train and test set. Reconvert to dense matrix and wrap in Pandas dataframe.
    with parallel_backend('threading', n_jobs = -1):
        X_train_copy_sparse = vtr.fit_transform(X_train_copy.lyrics)
        X_test_copy_sparse = vtr.transform(X_test_copy.lyrics)
        
        if to_frame:
            X_train_copy = pd.DataFrame(X_train_copy_sparse.todense(),
                                   columns = vtr.get_feature_names_out().tolist(),
                                   index = X_train_copy.index)
            X_test_copy = pd.DataFrame(X_test_copy_sparse.todense(),
                                   columns = vtr.get_feature_names_out().tolist(),
                                   index = X_test_copy.index)
            
            return X_train_copy, X_test_copy
        
    return X_train_copy_sparse, X_test_copy_sparse

def create_pyLDAvis(model, corpus, dictionary, plot = False, save = False, save_dest = None):
    '''
    Takes in a trained model, a corpus, and a trained gensim Dictionary. Returns a prepared pyLDAvis object.
    
    model: Fitted model to pass into pyLDAvis.
    corpus: Corpus to pass into pyLDAvis.
    dictionary: Trained gensim Dictionary to pass into pyLDAvis.
    plot: True means the notebook cell will be enabled to display pyLDAvis objects. Default False.
    save: True means the pyLDAvis will be saved as an html in the save_dest file. Default False.
    save_dest: String denoting save destination. Required to save the pyLDAvis html object. Default None.
    '''
    
    with parallel_backend('loky', n_jobs = -1):
        # Create the pyLDAvis object
        p = pyLDAvis.gensim_models.prepare(model, corpus, dictionary)

    # Plot the object
    if plot:
        pyLDAvis.enable_notebook()
        pyLDAvis.display(p)
    
    # Save the object
    if save:
        if save_dest is None:
            print('No save destination specified, pyLDAvis object not saved.')
        else:
            pyLDAvis.save_html(p, save_dest)
            
    return p

def perform_tsne_analysis(X_train, X_test, y_train, y_test, random_state = 42, learning_rate = 100,
                          pickle_ = False, pickle_dest = None, plot = False):
    '''
    Performs a TSNE transformation of input data, returns a dataframe containing TSNE information.
    
    X_train: Train data to be stitched to test data and passed into scikit-learn TSNE fit_transform method.
    X_test: Test data to be stitched to train data and passed into scikit-learn TSNE fit_transform method.
    y_train: Train labels to be stitched to test labels to be used to create hue on visualizations.
    y_test: Test labels to be stitched to train labels to be used to create hue on visualizations.
    random_state: Integer denoting random state to be used in instantiation of TSNE object. Default 42.
    learning_rate: Float denoting learning rate to be used in instantiation of TSNE object. Default 100.
    pickle_: True means the object will be pickled in the pickle_dest. Default False.
    pickle_dest: String denoting pickle destination. Required to pickle file. Default None.
    plot: True means the TSNE dataframe will be plotted in the cell using seaborn. Default False.
    '''
    
    # Instantiate TSNE object
    tsne = TSNE(random_state = random_state, learning_rate = learning_rate)
    # Fit the TSNE object
    with parallel_backend('loky', n_jobs = -1):
        tsne_trans = tsne.fit_transform(pd.concat([X_train, X_test]))
    
    # Create a dataframe to hold the results
    tsne_trans = pd.DataFrame(tsne_trans, columns = ['TSNE1', 'TSNE2'])
    tsne_trans.index = X_train.index.values.tolist() + X_test.index.values.tolist()
    tsne_trans['genre'] = y_train.values.tolist() + y_test.values.tolist()
    
    # Plot the results
    if plot:
        plt.figure(figsize=(10,10))
        sns.scatterplot(x = 'TSNE1', y = 'TSNE2', hue = 'genre', data = tsne_trans, palette = 'tab10')
        plt.title('Visualization of Genres')
        plt.show()
        
    # Pickle the results
    if pickle_:
            if pickle_dest is None:
                print('No pickle destination given, pickling skipped.')
            else:
                with open(pickle_dest, 'wb') as f:
                    pickle.dump(tsne_trans, f)
                    
    return tsne_trans