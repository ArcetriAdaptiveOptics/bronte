import numpy as np 

class SyntheticReconstructorComputer():
    '''
    this class is meant for computing the reconstruct that
    allows to compute the modal coefficients for the slopes
    '''
    def __init__(self, slope_computer , modal_base):
        
        self._type = 'FromSlopes2ModalCoeff' 
        self._reconstructor = None
        
        self._modal_base = modal_base
        self._interaction_matrix = None
        self._n_of_modes = None
    
    
    
    def _compute_slopes_interaction_matrix(self):
        pass
    
    
    def get_reconstructor(self):
        return self._reconstructor