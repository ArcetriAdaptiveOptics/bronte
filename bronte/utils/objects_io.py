import pickle

def save_object(obj, fname):
    
    with open(fname, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_object(fname):
    
    with open(fname, 'rb') as inp:
        obj = pickle.load(inp)
    return obj     