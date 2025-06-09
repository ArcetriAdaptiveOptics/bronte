import numpy as np
from arte.utils.zernike_decomposer import ZernikeModalDecomposer
import logging
from arte.utils.decorator import logEnterAndExit
from arte.types.zernike_coefficients import ZernikeCoefficients
from functools import cached_property



class SlmRasterizer():
    '''
    Input: modal coefficients vector, pupil
    Output: wavefront to be applied on the SLM
    
    Assumes that the modal basis is a Zernike base
    '''
    
    def __init__(self, slm_pupil_mask, Nmodes = 200, modal_coefficients = None):
        
        self._logger = logging.getLogger("SlmRasterizer")
        self._n_modes = Nmodes
        self._zernike_modal_decomposer = ZernikeModalDecomposer(n_modes = Nmodes)
        self.slm_pupil_mask = slm_pupil_mask
        self.modal_coefficients = modal_coefficients
        
    @logEnterAndExit("Converting zernike coefficients to slm map",
                     "Zernike coefficients converted to slm map", level='debug')
    def zernike_coefficients_to_raster(self, zernike_coeffs):
        '''
        Convert a ZernikeCoefficients object or an array of coefficients (in m rms)
        to a wavefront raster in wf meter units.
        '''
        if isinstance(zernike_coeffs, np.ndarray):
            zernike_coeffs = self.get_zernike_coefficients_from_numpy_array(zernike_coeffs)
        
        wfz = self._zernike_modal_decomposer.recomposeWavefrontFromModalCoefficients(
            zernike_coeffs, self.slm_pupil_mask)
        return wfz
 
    def get_wavefront_on_pupil(self, wf):
        
        wf_mask = self.slm_pupil_mask.mask()
        wf_on_pupil = np.ma.array(wf, mask = wf_mask)
        wf_on_pupil.fill_value = 0
        return wf_on_pupil
    
    #@cached_property
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
    
    def get_zernike_coefficients_from_numpy_array(self, coef_array):
        return ZernikeCoefficients.fromNumpyArray(coef_array)
    
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
    
    @logEnterAndExit("Converting zernike coefficients to slm command",
                     "Zernike coefficients converted to slm command", level='debug') 
    def m2c(self, zernike_coeffs, applyTiltUnderMask = False):
        wf2raster = self.zernike_coefficients_to_raster(zernike_coeffs).toNumpyArray()
        if applyTiltUnderMask is True:
            wf2raster = self.load_a_tilt_under_pupil_mask(wf2raster)
        command_vector = self.reshape_map2vector(wf2raster)
        return command_vector

