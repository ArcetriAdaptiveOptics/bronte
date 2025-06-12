from bronte.startup import set_data_dir
from bronte.package_data import subaperture_set_folder, reconstructor_folder, pp_amp_vector_folder
from bronte.calibration.display_slope_maps_from_intmat import DisplaySlopeMapsFromInteractionMatrix
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits 

class ExperimentalPushPullAmplitudeComputer():
    '''
    like for eris. this class is meant to correct
    a pp amplitude vector looking at the slope std
    a parameter that gives us an idea of how much
    the modes are visible
    '''
    
    def __init__(self, subap_tag, intmat_tag, pp_vect_in_nm):
        
        self._subap_tag = subap_tag
        self._intmat_tag = intmat_tag
        self._dsm = DisplaySlopeMapsFromInteractionMatrix(
            intmat_tag, subap_tag, pp_vect_in_nm)
        self._pp_vect_in_nm = pp_vect_in_nm
        self._rescaled_pp_vector  = None
        self._im_std =  self._dsm._intmat._intmat.std(axis=1)*self._pp_vect_in_nm
        self._Nmodes = len(self._im_std)
        
    def display_ifs_std(self):
        
        plt.figure()
        plt.clf()
        plt.plot(self._im_std)
        plt.xlabel('mode index')
        plt.ylabel('Slope std [normalized]')
        
    def compute_rescaled_pp_vector(self, target_val = 0.1):
        
        self._target_val = target_val
        self._rescaled_pp_vector = (self._target_val/self._im_std)*self._pp_vect_in_nm
        
    def display_pp_amplitude_vector(self):
        
        j_noll_vector = np.arange(self._Nmodes) + 2
        plt.figure()
        plt.clf()
        plt.plot(j_noll_vector, self._pp_vect_in_nm, label='original')
        if self._rescaled_pp_vector is not None:
            plt.plot(j_noll_vector, self._rescaled_pp_vector, label = 'rescaled')
        plt.ylabel('Push-Pull [nm] rms wf')
        plt.xlabel('mode index')
        plt.grid('--', alpha=0.3)
        plt.legend(loc='best')
    
    def save_rescaled_pp_vector(self, ftag):
        
        file_name = pp_amp_vector_folder() / (ftag + '.fits') 
        hdr = fits.Header()
        hdr['SUB_TAG'] = self._subap_tag
        hdr['REC_TAG'] = self._intmat_tag
        hdr['TARG_VAL'] = self._target_val
        fits.writeto(file_name, self._rescaled_pp_vector, hdr)
    
    @staticmethod
    def load_pp_vector(ftag):
        
        file_name = pp_amp_vector_folder() / (ftag + '.fits')
        hdr = fits.getheader(file_name)
        hduList = fits.open(file_name)
        pp_amp_vector_in_nm = hduList[0].data
        return pp_amp_vector_in_nm, hdr
        