from bronte.types.slm_pupil_mask_generator import SlmPupilMaskGenerator
from bronte import package_data
from arte.utils.zernike_generator import ZernikeGenerator
import numpy as np 
from bronte.utils.modal_base_generators.orthonormal_base_computer \
    import OrthonormalBaseComputer
from bronte.utils.modal_base_generators.covariance_matrix_computer \
    import CovarianceMatrixComputer
from arte.atmo.von_karman_psd import VonKarmanPsd


def get_zernike_on_elt_pupil_modal_base(lastModeIndex = 10):
    '''
    lastModeIndex is the last zernike mode index (piston is neglected)
    j = 2, ..., lastModeIndex 
    Thus the effective number of modes is nModes = lastModeIndex-1
    '''
    fname = 'EELT480pp0.0803m_obs0.283_spider2023'
    elt_pupil_fname = package_data.elt_pupil_folder()/(fname + '.fits')
    
    spmg = SlmPupilMaskGenerator()
    pupil_mask  = spmg.elt_pupil_mask(elt_pupil_fname)
    
    zg = ZernikeGenerator(pupil_mask)
    n_of_points = pupil_mask.shape()[0]*pupil_mask.shape()[1]
    modal_base = np.ma.zeros((n_of_points, lastModeIndex-1), fill_value= 0)
    modes_index_list = np.arange(2, lastModeIndex+1)
    
    for idx, j in enumerate(modes_index_list):
        modal_base[:, idx] = zg.getZernike(j).flatten() 
   
    return modal_base

def main():
    '''
    the aim is to create a kl modal basis from a generic base
    (like zernikes on elt pupil mask) and compute the covariane matrix
    to take into account the atmospheric psd.
    '''
    # creating a generic base with an elt pupil mask
    lastModeIndex  = 11
    generic_base = get_zernike_on_elt_pupil_modal_base(lastModeIndex)
    
    #orthonormalising the generic base
    obc = OrthonormalBaseComputer(generic_base)
    orthonorm_base = obc.get_orthonormalised_base()
    
    #TODO: compute atmo psd
    L0 = 20
    r0 = 0.3
    psd = VonKarmanPsd(r0,L0)
    
    #TODO: compute covariance matrix
    cmc = CovarianceMatrixComputer(orthonorm_base, (1152, 1920), None)
    
    base_in_freq_dom = cmc._compute_base_fft()
    
    #cov_mat = cmc.get_covariance_matrix()
    
    #TODO: compute KL modes
    
    return base_in_freq_dom