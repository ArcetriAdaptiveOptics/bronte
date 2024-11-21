import numpy as np
from arte.utils.zernike_decomposer import ZernikeModalDecomposer
import logging
from arte.utils.decorator import logEnterAndExit
from arte.types.zernike_coefficients import ZernikeCoefficients



class SlmRasterizer():
    '''
    Input: modal coefficients vector, pupil
    Output: wavefront to be applied on the SLM
    
    Assumes that the modal basis is a Zernike base
    '''
    
    def __init__(self, slm_pupil_mask, modal_coefficients = None):
        
        self._logger = logging.getLogger("SlmRasterizer")
        self._zernike_modal_decomposer = ZernikeModalDecomposer(n_modes=200)
        self.slm_pupil_mask = slm_pupil_mask
        self.modal_coefficients = modal_coefficients
        
    @logEnterAndExit("Converting zernike coefficients to slm map",
                     "Zernike coefficients converted to slm map", level='debug')
    def zernike_coefficients_to_raster(self, zernike_coefficients):
        '''
        Convert a ZernikeCoefficients object to a wavefront raster
        in wf meter units.
        '''
        wfz = self._zernike_modal_decomposer.recomposeWavefrontFromModalCoefficients(
            zernike_coefficients, self.slm_pupil_mask)
        return wfz
 
    def get_wavefront_on_pupil(self, wf):
        
        wf_mask = self.slm_pupil_mask.mask()
        wf_on_pupil = np.ma.array(wf, mask = wf_mask)
        wf_on_pupil.fill_value = 0
        return wf_on_pupil
    
    def load_a_tilt_under_masked_pupil(self, wf, c2_m_rms = 30e-6):
        '''
        it returns a masked array where on the masked points is set a tilt
        c2_m_rms over cmask, then is extended to the full screen with the same slope
        '''
        c2 = c2_m_rms * 0.5 * self.slm_pupil_mask.shape()[1]/self.slm_pupil_mask.radius()
        tilt_profile = np.linspace(-2*c2, 2*c2, self.slm_pupil_mask.shape()[1])
        tilt_over_the_full_frame = self.reshape_vector2map(np.tile(tilt_profile, self.slm_pupil_mask.shape()[0]))
        wf2raster = wf.copy()
        wf2raster[wf.mask == True] = tilt_over_the_full_frame[wf.mask == True]
        return wf2raster
    
    def reshape_map2vector(self, array2d, length=2211840, method='C'):
        return np.reshape(array2d, (length,), method)

    def reshape_vector2map(self, vector, shape=[1152, 1920], method='C'):
        return np.reshape(vector, (shape[0], shape[1]), method)
    
    def get_zernike_coefficients_from_numpy_array(self, coef_array):
        return ZernikeCoefficients.fromNumpyArray(coef_array)
        

