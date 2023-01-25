import pickle

def picklify(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)
        
    return

def unpickle(file):
    with open(file, 'rb') as f:
        obj = pickle.load(f)
        
    return obj