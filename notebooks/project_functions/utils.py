import pickle
import pandas as pd
from scipy.sparse import csr_matrix

def picklify(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)
        
    return

def unpickle(file):
    with open(file, 'rb') as f:
        obj = pickle.load(f)
        
    return obj

def dataframe_to_sparse(df):
    sparse_matrix = csr_matrix(df.astype(pd.SparseDtype("float64",0)).sparse.to_coo())
    
    return sparse_matrix