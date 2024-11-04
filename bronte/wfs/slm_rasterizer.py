from functools import cached_property
import numpy as np
from arte.types.mask import CircularMask
from arte.utils.zernike_decomposer import ZernikeModalDecomposer
import logging
from arte.utils.decorator import logEnterAndExit
from arte.types.zernike_coefficients import ZernikeCoefficients


class SlmRasterizer():
    
    #SLM PUPIL MASK PARAMETERS IN PIXEL
    PUPIL_CENTER = (581, 875)#YX
    PUPIL_RADIUS = 571
    FRAME_SHAPE = (1152, 1920)
    
    #ELT PUPIL PARAMETERS SCALED TO SLM 
    SPIDER_DIM = 6
    SPIDER_ANGLE = 60*np.pi/180
    OBSTRUCTION_RADIUS = 161
    
    def __init__(self):
        self._logger = logging.getLogger("SlmRasterizer")
        self._zernike_modal_decomposer = ZernikeModalDecomposer(n_modes=10)

    @cached_property
    def slm_pupil_mask(self):
        
        cmask = CircularMask(
            frameShape = self.FRAME_SHAPE,
            maskRadius = self.PUPIL_RADIUS,
            maskCenter = self.PUPIL_CENTER)
        return cmask

    @logEnterAndExit("Converting zernike coefficients to slm map",
                     "Zernike coefficients converted to slm map", level='debug')
    def zernike_coefficients_to_raster(self, zernike_coefficients):
        '''
        Convert a ZernikeCoefficients object to a wavefront raster
        in wf meter units.
        '''
        wfz = self._zernike_modal_decomposer.recomposeWavefrontFromModalCoefficients(
            zernike_coefficients, self.slm_pupil_mask)
                
        wfz.toNumpyArray().fill_value = 0
        return wfz
    
      
    def _get_circular_obstruction_on_pupil_mask(self):
        
        pupil_cmask = self.slm_pupil_mask
        pupil_mask = pupil_cmask.mask().copy()
        
        obstraction_CircularMask = CircularMask(
            frameShape = self.FRAME_SHAPE,
            maskRadius = self.OBSTRUCTION_RADIUS,
            maskCenter = self.PUPIL_CENTER)
        
        obstraction_mask = obstraction_CircularMask.mask()
        pupil_mask[obstraction_mask == False] = True
        
        return pupil_mask
        
    def _add_spider_on_pupil_mask(self, pupil_mask_with_obstruction):
        
        pupil_mask = pupil_mask_with_obstruction.copy()
        
        ds = int(np.round(0.5 * self.SPIDER_DIM/np.cos(self.SPIDER_ANGLE)))
        y0, x0 = self.PUPIL_CENTER
        x = np.arange(x0 - self.PUPIL_RADIUS, x0 + self.PUPIL_RADIUS + 1)
        
        #horizontal spider
        pupil_mask[y0-ds : y0 + ds + 1, :] = True
        #diagonal spiders
        for x_idx in x:
            
            y1_idx = int(np.round((x_idx - x0)*np.tan(self.SPIDER_ANGLE) + y0))
            y2_idx = int(np.round(-1*(x_idx - x0)*np.tan(self.SPIDER_ANGLE) + y0))
            
            pupil_mask[y1_idx - ds : y1_idx + ds + 1, x_idx] = True
            pupil_mask[y2_idx - ds : y2_idx + ds + 1, x_idx] = True
        
        return pupil_mask
    
    def _get_elt_pupil_mask(self):
        
        pupil_mask_with_obstruction = self._get_circular_obstruction_on_pupil_mask()
        elt_pupil_mask = self._add_spider_on_pupil_mask(pupil_mask_with_obstruction)
        return elt_pupil_mask       
    
    def get_wavefront_on_circular_obstructed_pupil(self, wf):
        
        wf_mask = self._get_circular_obstruction_on_pupil_mask()
        wf_on_pupil = np.ma.array(wf, mask = wf_mask)
        wf_on_pupil.fill_value = 0
        return wf_on_pupil
    
    def get_wavefront_on_elt_pupil(self, wf):
        
        wf_mask = self._get_elt_pupil_mask()
        wf_on_pupil = np.ma.array(wf, mask = wf_mask)
        wf_on_pupil.fill_value = 0
        return wf_on_pupil
    
    def get_slm_raster_from_wf(self, wf, c2_m_rms = 30e-6):
        '''
        it sets a tilt where is masked
        c2_m_rms over cmask, then is extended to the full screen with the same slope
        '''
        c2 = c2_m_rms * 0.5 * self.FRAME_SHAPE[1]/self.PUPIL_RADIUS
        tilt_profile = np.linspace(-2*c2, 2*c2, self.FRAME_SHAPE[1])
        tilt_over_the_full_frame = self.reshape_vector2map(np.tile(tilt_profile, self.FRAME_SHAPE[0]))
        wf2raster = wf.copy()
        wf2raster[wf.mask == True] = tilt_over_the_full_frame[wf.mask == True]
        return wf2raster
    
    def reshape_map2vector(self, array2d, length=2211840, method='C'):
        return np.reshape(array2d, (length,), method)

    def reshape_vector2map(self, vector, shape=[1152, 1920], method='C'):
        return np.reshape(vector, (shape[0], shape[1]), method)
    
    def get_zernike_coefficients_from_numpy_array(self, coef_array):
        return ZernikeCoefficients.fromNumpyArray(coef_array)
        
    
