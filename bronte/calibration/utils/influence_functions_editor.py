import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np
from specula.data_objects.ifunc import IFunc
from bronte.calibration.utils.display_ifs_map import DisplayInfluenceFunctionsMap
from bronte.calibration.utils.kl_modal_base_generator import KarhunenLoeveGenerator
from bronte.startup import set_data_dir
from bronte.package_data import ifs_folder

class InfluenceFucntionEditor():
    
    def __init__(self, ifs_tag):
        
        self._ifs_tag = ifs_tag
        self._ifunc = KarhunenLoeveGenerator.load_modal_ifs(self._ifs_tag)
        self._dispIFs = DisplayInfluenceFunctionsMap(None, self._ifunc)
        self._pupil_mask_idl = self._ifunc.mask_inf_func
        
        self._edited_pupil_mask_idl = None
        self._edited_np_ifs = None
        
    def load_sigular_values(self):
        
        self._s1, self._s2 = KarhunenLoeveGenerator.load_singular_values(self._ifs_tag)
    
    def display_singular_values(self):
        import matplotlib.pyplot as plt
        
        plt.semilogy(self._s1, '.-', label='IF Covariance')
        plt.semilogy(self._s2, '.-', label='Turbulence Covariance')
        plt.xlabel('Mode number')
        plt.ylabel('Singular value')
        plt.title('Singular values of covariance matrices')
        plt.legend(loc='best')
        plt.grid('--', alpha=0.3)
        
    
    def remove_modes(self, Nmodes = 10):

        self._Nmodes = Nmodes
        full_np_ifs = self._ifunc.influence_function.T.copy()
        self._filtered_np_ifs = full_np_ifs[:Nmodes,:]
        
    def rescale_ifs(self, new_frame_size = 545*2):
        pass
    
    def save_filtered_ifs(self, ftag):
        # np_ifs has dim (Nmodes, NPupValidPoints)
        # ifunc_obj has dim (NPupValidPoints,Nmodes)
        # and to be initialized need np_ifs
        pupil_mask_idl = self._edited_pupil_mask_idl
        np_ifs = self._edited_np_ifs
        
        if self._edited_pupil_mask_idl is None:
            pupil_mask_idl = self._pupil_mask_idl
        if self._edited_np_ifs is None:
            np_ifs = self._filtered_np_ifs
        
        set_data_dir()
        edited_pupil_mask_idl = pupil_mask_idl 
        ifunc_obj = IFunc(
            ifunc = np_ifs,
            mask = edited_pupil_mask_idl)
        fname  = ifs_folder() / (ftag + '.fits')
        ifunc_obj.save(fname)
        