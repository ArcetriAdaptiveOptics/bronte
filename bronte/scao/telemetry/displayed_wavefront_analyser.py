import numpy as np 
from bronte.scao.specula_scao_runner import SpeculaScaoRunner
import matplotlib.pyplot as plt

class DisplayedWavefrontAnalyser():
    
    def __init__(self, ftag):
        
        self._wf_cube_in_nm, self._hdr = SpeculaScaoRunner.load_displayed_wf(ftag)
        self._Nwf = self._wf_cube_in_nm.shape[0]
    
    def set_slm_pupil_mask(self, slm_pupil_mask):
        
        self._slm_pupil_mask = slm_pupil_mask
        
    
    def _get_recentred_wf_on_slm_pupil_frame(self, wf):
        new_size = 2 * self._slm_pupil_mask.radius()
        wf_on_slm_pupil_frame = np.zeros(self._slm_pupil_mask.shape())
        top_left = self._slm_pupil_mask.center()[0] - self._slm_pupil_mask.radius()
        bottom_left = self._slm_pupil_mask.center()[1] - self._slm_pupil_mask.radius()
        wf_on_slm_pupil_frame[top_left:top_left + new_size,
                           bottom_left: bottom_left + new_size] = wf
        wf_mask = self._slm_pupil_mask.mask()
        
        return np.ma.array(wf_on_slm_pupil_frame, mask = wf_mask)
    
    def apply_slm_pupil_mask_on_displayed_wf(self):
        
        wf_on_slm_pup_list = []
        for idx in range(self._Nwf):
            wf = self._wf_cube_in_nm[idx]
            wf_on_slm_pup = self._get_recentred_wf_on_slm_pupil_frame(wf)
            wf_on_slm_pup_list.append(wf_on_slm_pup)  
        self._wf_cube_on_slm = np.ma.array(wf_on_slm_pup_list)
        
    def set_bias_wf(self, bias_wf_in_nm):
        self._bias_wf = bias_wf_in_nm
        
    
    def display_applied_wf(self, idx, addBias = False, subBias = False):
        
        disp_wf = self._wf_cube_on_slm[idx]
        
        if subBias is True:
            disp_wf -= self._bias_wf 
        
        if addBias is True:
            disp_wf += self._bias_wf 
        
        plt.figure()
        plt.clf()
        plt.imshow(disp_wf)
        plt.colorbar(label='nm')
        plt.title('Applied WF on SLM at step %d'%idx)