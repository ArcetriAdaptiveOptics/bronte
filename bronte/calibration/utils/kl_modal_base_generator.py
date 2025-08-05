import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np
from specula.lib.modal_base_generator import make_modal_base_from_ifs_fft
from specula.data_objects.ifunc import IFunc
#from specula.data_objects.m2c import M2C
from specula import cpuArray
from bronte.startup import set_data_dir
from bronte.package_data import ifs_folder
from bronte.calibration.utils.zonal_influence_function_computer import ZonalInfluenceFunctionComputer

class KarhunenLoeveGenerator():
    
    def __init__(self, ifs_tag):
        
        self._ifs_tag = ifs_tag
        self._ifunc = ZonalInfluenceFunctionComputer.load_ifs(self._ifs_tag)
        self._pupil_diameter_in_pixels = self._ifunc.mask_inf_func.shape[0]
        self._pupil_mask_idl = self._ifunc.mask_inf_func
        
        self._dtype = specula.xp.float32
        
        self._telescope_diameter_in_m = None
        self._r0 = None
        self._L0 = None
    
    
    def set_atmo_parameters(self, Dtel_in_m = 8.2, r0_in_m = 0.15, L0_in_m = 25):
        
        self._telescope_diameter_in_m = Dtel_in_m
        self._r0 = r0_in_m
        self._L0 = L0_in_m
        
    def get_actuator_if_2Dmap(self, act_index):
        
        pup_size = self._pupil_diameter_in_pixels
        pup_mask_idl = self._ifunc.mask_inf_func
        actuator_if = np.zeros((pup_size, pup_size))
        actuator_if[self._ifunc.idx_inf_func] = self._ifunc.influence_function[:, act_index]
        ma_actuator_if = np.ma.array(data = actuator_if, mask = 1 - pup_mask_idl)
        
        return ma_actuator_if
    
    def display_actuator_if(self, act_index):
        
        import matplotlib.pyplot as plt
        if_map = self.get_actuator_if_2Dmap(act_index)
        plt.figure()
        plt.clf()
        plt.title("IF#%d"%act_index)
        plt.imshow(if_map)
        plt.colorbar(label='Normalized')
        
    
    def compute_modal_basis(self, zern_modes = 5, oversampling = 1, if_max_condition_number = None):
        
        #zern_modes is the number of zernike modes to be included on the modal basis 
        ifs = self._ifunc.influence_function.T
        
        self._kl_basis, self._m2c, self._singular_values = make_modal_base_from_ifs_fft(
            pupil_mask = self._pupil_mask_idl,
            diameter = self._telescope_diameter_in_m,
            influence_functions = ifs,
            r0 = self._r0,
            L0 = self._L0,
            zern_modes = zern_modes,
            oversampling = oversampling,
            if_max_condition_number = if_max_condition_number,
            xp = specula.xp,
            dtype = self._dtype)
        
    def save_kl_modes_as_modal_ifs(self, ftag):
        set_data_dir()
        ifunc_obj = IFunc(
            ifunc = self._kl_basis,
            mask = self._pupil_mask_idl)
        fname  = ifs_folder() / (ftag + '.fits')
        ifunc_obj.save(fname)
    
    @staticmethod
    def load_modal_ifs(ftag):
        set_data_dir()
        fname = ifs_folder() / (ftag + '.fits')
        return IFunc.restore(fname)