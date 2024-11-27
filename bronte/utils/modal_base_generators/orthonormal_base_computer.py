import numpy as np 
from bronte.types.slm_pupil_mask_generator import SlmPupilMaskGenerator
from bronte import package_data
from arte.utils.zernike_generator import ZernikeGenerator 

def get_zernike_on_elt_pupil_modal_base(nModes = 10):
    
    fname = 'EELT480pp0.0803m_obs0.283_spider2023'
    elt_pupil_fname = package_data.elt_pupil_folder()/(fname + '.fits')
    
    spmg = SlmPupilMaskGenerator()
    pupil_mask  = spmg.elt_pupil_mask(elt_pupil_fname)
    
    zg = ZernikeGenerator(pupil_mask)
    n_of_points = pupil_mask.shape()[0]*pupil_mask.shape()[1]
    modal_base = np.ma.zeros((n_of_points, nModes-2), fill_value= 0)
    modes_index_list = np.arange(2, nModes)
    
    for idx, j in enumerate(modes_index_list):
        modal_base[:, idx] = zg.getZernike(j).flatten() 
   
    return modal_base

class OrthonormalBaseComputer():
    '''
    Orthonormalises a generic base applying Singular Value Decomposition
    (SVD). It is assumed that the input and output modal base are 2D array where
    each the column is a mode. The generic base could be generated with masked 
    array to reproduce pupil obstructions.
    
    '''
    def __init__(self, generic_base):
        
        self._generic_base = generic_base
        self._compute_orthonormalised_modes()
    
    def _compute_orthonormalised_modes(self):
        '''
        generic_base  = U S V^t
        the colums of U contain the orthonormal modes
        '''
        u, s, vt = np.linalg.svd(
            self._generic_base, full_matrices = False)
        self._check_if_is_orthonormal(u)
        self._orthonormal_modes = u
    
    def _check_if_is_orthonormal(self, u):
        
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
        
    def get_orthonormalised_modes(self):
        
        return self._orthonormal_modes