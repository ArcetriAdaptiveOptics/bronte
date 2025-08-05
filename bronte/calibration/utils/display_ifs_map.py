import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np
from specula.data_objects.ifunc import IFunc
from bronte.startup import set_data_dir
from bronte.package_data import ifs_folder
import matplotlib.pyplot as plt

class DisplayInfluenceFunctionsMap():
    
    def __init__(self, ifs_tag = None, ifunc = None):
        
        if ifs_tag is not None:
            self._ifs_tag = ifs_tag
            self._ifunc  = self.load_ifs(self._ifs_tag)
        if ifunc is not None:
            self._ifunc = ifunc
            
        self._pupil_diameter_in_pixels = self._ifunc.mask_inf_func.shape[0]
        self._pupil_mask_idl = self._ifunc.mask_inf_func
        
    @staticmethod
    def load_ifs(ftag):
        set_data_dir()
        fname = ifs_folder() / (ftag + '.fits')
        return IFunc.restore(fname)
    
    def get_if_2Dmap(self, if_index):
        
        pup_size = self._pupil_diameter_in_pixels
        pup_mask_idl = self._ifunc.mask_inf_func
        if2Dmap = np.zeros((pup_size, pup_size))
        if2Dmap[self._ifunc.idx_inf_func] = self._ifunc.influence_function[:, if_index]
        ma_if2Dmap = np.ma.array(data = if2Dmap, mask = 1 - pup_mask_idl)
        
        return ma_if2Dmap
    
    def display_actuator_if(self, if_index):
        
        if_map = self.get_actuator_if_2Dmap(if_index)
        plt.figure()
        plt.clf()
        plt.title("IF#%d"%if_index)
        plt.imshow(if_map)
        plt.colorbar(label='Normalized')