import numpy as np
import logging
from arte.utils.decorator import logEnterAndExit
from functools import cache
from bronte.types.slm_pupil_mask_generator import SlmPupilMaskGenerator
from bronte.calibration.utils.display_ifs_map import DisplayInfluenceFunctionsMap

class KLSlmRasterizer():
    
    def __init__(self, slm_pupil_mask, kl_modal_base_ftag):
        
        self._logger = logging.getLogger("KL-SlmRasterizer")
        self.slm_pupil_mask = slm_pupil_mask
        self._kl_modal_base_ftag = kl_modal_base_ftag
        self._load_kl_modal_base()
        self._update_slm_pupil_mask_with_ifs_mask(self._ifs_mask_idl)
        
    def _load_kl_modal_base(self):
        
        self._klg = DisplayInfluenceFunctionsMap(self._kl_modal_base_ftag)
        self._ifs_mask_idl = self._klg._ifunc.mask_inf_func
        self.Nmodes = np.min(self._klg._ifunc.influence_function.shape)
    
    def _update_slm_pupil_mask_with_ifs_mask(self, ifs_mask_idl):
        
        spg = SlmPupilMaskGenerator(
            pupil_radius = self.slm_pupil_mask.radius(), 
            pupil_center = self.slm_pupil_mask.center(),
            frameShape =  self.slm_pupil_mask._shape)
        
        mask = (1 - ifs_mask_idl).astype(bool)
        rescaled_mask = spg._get_rescaled_mask_to_slm_frame(mask)
        self.slm_pupil_mask._mask = rescaled_mask
        
    @logEnterAndExit("Converting KL coefficients to slm map",
                     "KL coefficients converted to slm map", level='debug')        
    def kl_coefficients_to_raster(self, kl_coeff):
        
        wfkl = 0
    
        for idx, coeff in np.ndenumerate(kl_coeff):
            if coeff == 0.:
                pass
            else:
                wfkl += self._klg.get_if_2Dmap(idx)*coeff
    
        wfkl4slm = self.get_recentered_phase_screen_on_slm_pupil_frame(wfkl)
        return wfkl4slm
    
    
    @logEnterAndExit("Recentering wavefront on slm pupil",
                     "Wavefront recentered on the slm pupil", level='debug')     
    def get_recentered_phase_screen_on_slm_pupil_frame(self, phase_screen):
        # TODO: rise error if the shift exceed the slm frame size
        new_size = 2 * self.slm_pupil_mask.radius()
        phase_on_slm_pupil_frame = np.zeros(self.slm_pupil_mask.shape())
        top_left = self.slm_pupil_mask.center()[0] - self.slm_pupil_mask.radius()
        bottom_left = self.slm_pupil_mask.center()[1] - self.slm_pupil_mask.radius()
        phase_on_slm_pupil_frame[top_left:top_left + new_size,
                           bottom_left: bottom_left + new_size] = phase_screen
        phase_screen_mask = self.slm_pupil_mask.mask()
        
        return np.ma.array(phase_on_slm_pupil_frame, mask = phase_screen_mask)

    @cache
    def _get_tilt_over_slm_full_frame(self, c2_m_rms = 60e-6):
        c2 = c2_m_rms * 0.5 * self.slm_pupil_mask.shape()[1]/self.slm_pupil_mask.radius()
        tilt_profile = np.linspace(-2*c2, 2*c2, self.slm_pupil_mask.shape()[1])
        tilt_over_the_full_frame = self.reshape_vector2map(np.tile(tilt_profile, self.slm_pupil_mask.shape()[0]))
        return np.ma.array(data = tilt_over_the_full_frame, mask = self.slm_pupil_mask)
    
    @logEnterAndExit("Loading a huge Tilt under Wavefront Mask",
                     "Huge Tilt loaded on Wavefront Mask", level='debug') 
    def load_a_tilt_under_pupil_mask(self, wf, c2_m_rms = 60e-6):
        '''
        it returns a masked array where on the masked points is set a tilt
        c2_m_rms over cmask, then is extended to the full screen with the same slope
        '''
        tilt_over_the_full_frame = self._get_tilt_over_slm_full_frame(c2_m_rms)
        wf2raster = wf.copy()
        wf2raster[wf.mask == True] = tilt_over_the_full_frame[wf.mask == True]
        return wf2raster

    @logEnterAndExit("Reshaping Wavefront map to vector",
                     "Wavefront map reshaped to vector", level='debug') 
    def reshape_map2vector(self, array2d, length=2211840, method='C'):
        return np.reshape(array2d, (length,), method)
    
    @logEnterAndExit("Reshaping Wavefront vector to map",
                     "Wavefront vector reshaped to map", level='debug')
    def reshape_vector2map(self, vector, shape=[1152, 1920], method='C'):
        return np.reshape(vector, (shape[0], shape[1]), method)
    
    @logEnterAndExit("Converting KL coefficients to slm command",
                     "KL coefficients converted to slm command", level='debug') 
    def m2c(self, kl_coeffs, applyTiltUnderMask = False):
        wf2raster = self.zkl_coefficients_to_raster(kl_coeffs)
        if applyTiltUnderMask is True:
            wf2raster = self.load_a_tilt_under_pupil_mask(wf2raster)
        command_vector = self.reshape_map2vector(wf2raster)
        return command_vector
