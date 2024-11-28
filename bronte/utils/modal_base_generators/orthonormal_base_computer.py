import numpy as np 
from functools import cache 


class OrthonormalBaseComputer():
    '''
    Orthonormalises a generic base applying Singular Value Decomposition
    (SVD). It is assumed that the input and output base are 2D array where
    each the column corresponds to a mode. The generic base could be generated with masked 
    array to reproduce pupil obstructions.
    
    '''
    def __init__(self, generic_base):
        
        self._generic_base = generic_base
        
    
    def _compute_orthonormalised_modes(self):
        '''
        generic_base  = U S V^t
        the columns of U contain the orthonormal modes
        '''
        u, s, vt = np.linalg.svd(
            self._generic_base, full_matrices = False)
        self._check_if_is_orthonormal(u)
        self._orthonormal_base = u
        
    @staticmethod
    def _check_if_is_orthonormal(u):
        
        # from othogonality must be U^t U = Id
        matrix = np.dot(u.T, u)
        id_mat = np.eye(matrix.shape[0])
        is_othogonal = np.allclose(matrix, id_mat)
        
        # each columns must have norm = 1 
        norms = np.linalg.norm(u, axis=0)
        is_normalised = np.allclose(norms, 1)
        
        if is_othogonal is False or is_normalised is False:
            error_message = f"U is not orthonormal.\n"\
                            f"- orthogonal: {is_othogonal}\n"\
                            f"- normal: {is_normalised}"
            raise ValueError(error_message)
    
    @cache
    def get_orthonormalised_base(self):
        self._compute_orthonormalised_modes()
        return self._orthonormal_base