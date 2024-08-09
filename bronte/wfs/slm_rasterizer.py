from functools import cached_property
import numpy as np
from arte.types.mask import CircularMask
from arte.utils.zernike_decomposer import ZernikeModalDecomposer
import logging
from arte.utils.decorator import logEnterAndExit
from arte.types.zernike_coefficients import ZernikeCoefficients


class SlmRasterizer:

    def __init__(self):
        self._logger = logging.getLogger("SlmRasterizer")
        self._zernike_modal_decomposer = ZernikeModalDecomposer(n_modes=10)

    @cached_property
    def slm_pupil_mask(self):
        centerYX = (581, 875)
        RadiusInPixel = 571
        frameshape = (1152, 1920)
        cmask = CircularMask(
            frameShape=frameshape,
            maskRadius=RadiusInPixel,
            maskCenter=centerYX)
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
    
    def get_raster_on_circular_obstruction(self, array2d, RadiusInPixel):
        '''
        array2d is the command to apply
        map2display is ONLY for plotting
        '''
        centerYX = (581, 875)
        frameshape = (1152, 1920)
        obstruction_cmask = CircularMask(
            frameShape=frameshape,
            maskRadius=RadiusInPixel,
            maskCenter=centerYX)
        tilt_coeff = ZernikeCoefficients.fromNumpyArray(np.array([30e-6,0]))
        obstruction = self._zernike_modal_decomposer.recomposeWavefrontFromModalCoefficients(
            tilt_coeff, obstruction_cmask)
        
        idx_list_y, idx_list_x = np.where(obstruction.toNumpyArray().mask==False)
        array2d[idx_list_y, idx_list_x] = obstruction.toNumpyArray()[idx_list_y, idx_list_x]
        
        if np.ma.is_masked(array2d):
            map2display = array2d.copy()
            map2display.mask[idx_list_y,idx_list_x] = True
        else:
            data2display = array2d.copy()
            map2display = np.ma.array(data=data2display, mask = np.zeros(frameshape))
            map2display.mask[idx_list_y,idx_list_x] = True 
        
        return array2d, map2display
        
    def get_raster_on_elt_pupil(self, wf):
        pass
    
    def reshape_map2vector(self, array2d, length=2211840, method='C'):
        return np.reshape(array2d, (length,), method)

    def reshape_vector2map(self, vector, shape=[1152, 1920], method='C'):
        return np.reshape(vector, (shape[0], shape[1]), method)
    
