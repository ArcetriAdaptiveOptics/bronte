import numpy as np 

def from_noll_to_radial_order(noll_index_vector):
    n_modes = len(noll_index_vector)
    n_vector = np.zeros(n_modes)
    
    for idx, j in enumerate(noll_index_vector):
        n_vector[idx] = int((-1 + np.sqrt(1 + 8 * (j - 1))) / 2)
    
    return n_vector
    