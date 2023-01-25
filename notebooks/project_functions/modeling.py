import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
from joblib import parallel_backend

def vectorize_data(X_train, X_test, y_train, y_test,
                   vectorizer, vectorizer_args = None, additional_features = False,
                   min_samples = 0, test_size = 0.25, random_state = 42):
    genres_to_keep = df.genre_name.value_counts()[df.genre_name.value_counts() > min_samples].index.values.tolist()
    vtr = vectorizer
    if vectorizer_args is not None:
        vtr.set_params(**vectorizer_args)
    df = df.loc[df.genre_name.isin(genres_to_keep)].copy()
        
    if additional_features:
        X_train['average_token_length'] = X_train.lyrics.apply(lambda x: np.array([len(i) for i in x]).sum() / len(x))
        X_test['average_token_length'] = X_test.lyrics.apply(lambda x: np.array([len(i) for i in x]).sum() / len(x))
        X_train['num_tokens'] = X_train.lyrics.apply(lambda x: len(x))
        X_test['num_tokens'] = X_test.lyrics.apply(lambda x: len(x))
        
    X_train.lyrics = X_train.lyrics.str.join(' ')
    X_test.lyrics = X_test.lyrics.str.join(' ')
    
    with parallel_backend('loky', n_jobs = -1):
        X_train = pd.concat([pd.DataFrame(vtr.fit_transform(X_train.lyrics).todense(), index = X_train.index),
                             X_train.num_tokens, X_train.average_token_length], axis = 1)
        X_test = pd.concat([pd.DataFrame(vtr.transform(X_test.lyrics).todense(), index = X_test.index),
                            X_test.num_tokens, X_test.average_token_length], axis = 1)
    
    return X_train, X_test, y_train, y_test, vtr    

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