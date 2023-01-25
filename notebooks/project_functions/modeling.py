import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
from joblib import parallel_backend

<<<<<<< HEAD
=======
def vectorize_data(df, vectorizer, vectorizer_args = None, additional_features = False,
                    min_samples = 0, test_size = 0.25, random_state = 42):
    genres_to_keep = df.genre_name.value_counts()[df.genre_name.value_counts() > min_samples].index.values.tolist()
    vtr = vectorizer
    if vectorizer_args is not None:
        vtr.set_params(**vectorizer_args)
    df = df.loc[df.genre_name.isin(genres_to_keep)].copy()
    X_columns = ['lyrics']
    
    if additional_features:
        df['average_token_length'] = df.lyrics.apply(lambda x: np.array([len(i) for i in x]).sum() / len(x))
        df['num_tokens'] = df.lyrics.apply(lambda x: len(x))
        X_columns = X_columns + ['num_tokens', 'average_token_length']
        
    df.lyrics = df.lyrics.str.join(' ')
    
    X = df[X_columns]
    y = df.genre_name
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    
    with parallel_backend('loky', n_jobs = -1):
        X_train = pd.concat([pd.DataFrame(vtr.fit_transform(X_train.lyrics).todense(), index = X_train.index),
                             X_train.num_tokens, X_train.average_token_length], axis = 1)
        X_test = pd.concat([pd.DataFrame(vtr.transform(X_test.lyrics).todense(), index = X_test.index),
                            X_test.num_tokens, X_test.average_token_length], axis = 1)
    
    return X_train, X_test, y_train, y_test, vtr    

def perform_tsne_analysis(X_train, X_test, y_train, y_test, random_state = 42, learning_rate = 100,
                          pickle = False, pickle_dest = None, plot = False):
    
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
        
    if pickle:
            if pickle_dest is None:
                print('No pickle destination given, pickling skipped.')
            else:
                with open(pickle_dest, 'wb') as f:
                    pickle.dump(tsne_trans, f)
                    
    return tsne_trans
>>>>>>> parent of 983d7c8... Started EDA and gensim bigram construction.
