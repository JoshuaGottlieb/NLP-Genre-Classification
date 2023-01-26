import pickle
import pandas as pd
from scipy.sparse import csr_matrix

def picklify(obj, file):
    '''
    Pickles an object.
    
    obj: Object, to be pickled.
    file: String, destination of pickling.
    '''
    
    with open(file, 'wb') as f:
        pickle.dump(obj, f)
        
    return

def unpickle(file):
    '''
    Unpickles an object. Returns the unpickled object.
    
    file: String, location of pickled object.
    '''
    
    with open(file, 'rb') as f:
        obj = pickle.load(f)
        
    return obj

def dataframe_to_sparse(df):
    '''
    Converts a pandas DataFrame to a sparse compressed-row matrix.
    
    df: DataFrame, to convert.
    
    '''
    sparse_matrix = csr_matrix(df.astype(pd.SparseDtype("float64",0)).sparse.to_coo())
    
    return sparse_matrix