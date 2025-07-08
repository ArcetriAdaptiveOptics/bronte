import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np

from specula.data_objects.intmat import Intmat
from specula.data_objects.recmat import Recmat
from bronte.package_data import reconstructor_folder
from bronte.calibration.runners.measured_control_matrix_calibration.MeasuredControlMatrixCalibrator import load_calib_config


class ControlMatrixEditor():
    
    def __init__(self, intmat_tag):
        
        self._intmat_tag = intmat_tag
        self._specula_intmat = self._load_specula_intmat(self._intmat_tag)
        
        self._specula_recmat = None
        
    @staticmethod
    def _load_specula_intmat(intmat_tag):
        
        file_name = reconstructor_folder() / (intmat_tag + '_bronte_im.fits')
        int_mat = Intmat.restore(file_name)
        #Npp = 2
        #int_mat._intmat = int_mat._intmat / Npp
        return int_mat
    
    def pseudo_invert_intmat(self, cond_factor):
        
        numpy_intmat = self._specula_intmat._intmat
        numpy_tsvd_recmat = np.linalg.pinv(numpy_intmat, rcond=cond_factor)
        self._specula_recmat = Recmat(numpy_tsvd_recmat)
        self._specula_recmat.im_tag = self._specula_intmat._norm_factor
        
        
    def filter_modes_from_intmat(self, index_list):
        pass
    
    def save_control_matrices(self, ftag):
        
        from astropy.io import  fits
        config_hdr, pp_vector_in_nm = load_calib_config(self._intmat_tag)
        
        config_hdr['REC_TAG'] = ftag 
        
        config_fname = ftag + '_bronte_calib_config'
        config_file_name = reconstructor_folder() / (config_fname + '.fits')
        
        
        fits.writeto(config_file_name, pp_vector_in_nm, config_hdr)
        
        rec_tag = ftag + '_bronte_rec.fits'
        file_name = reconstructor_folder() / rec_tag
        self._specula_recmat.save(str(file_name))
    
    def reset_control_matrices(self):
        self._specula_intmat = self._load_specula_intmat(self._intmat_tag)
        self._specula_recmat = None
    