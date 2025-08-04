import numpy as np 
from bronte.startup import set_data_dir
from bronte.calibration.utils.control_matrix_editor import ControlMatrixEditor
from astropy.io import fits

def main250804_163000():
    
    
    fname = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\bronte\\other_data\\250804_151400.fits"
    hduList = fits.open(fname)
    rejection_ratio = hduList[0].data
    index_of_noisy_modes = np.where(rejection_ratio < 2)[0]
    
    calib_tag = '250616_103300'
    set_data_dir()
    cme = ControlMatrixEditor(calib_tag)
    cme.filter_modes_from_intmat(index_list = index_of_noisy_modes, remove_cols=True)
    cme.pseudo_invert_intmat(cond_factor = 0)
    cme.save_control_matrices(ftag = '250804_163000')
    
    return cme
    
    
def main250804_164700():
    
    
    fname = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\bronte\\other_data\\250804_151400.fits"
    hduList = fits.open(fname)
    rejection_ratio = hduList[0].data
    index_of_noisy_modes = np.where(rejection_ratio < 2)[0]
    
    calib_tag = '250616_103300'
    set_data_dir()
    cme = ControlMatrixEditor(calib_tag)
    cme.filter_modes_from_intmat(index_list = index_of_noisy_modes, remove_cols=False)
    cme.pseudo_invert_intmat(cond_factor = 0)
    cme.save_control_matrices(ftag = '250804_164700')
    
    return cme