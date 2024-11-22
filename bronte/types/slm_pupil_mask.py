import numpy as np 
from arte.types.mask import CircularMask, AnnularMask
import logging
from functools import cached_property
#from arte.utils.decorator import logEnterAndExit

class SlmPupilMask():
    '''
    This class is meant to generate a pupil mask
    for the slm
    '''
    #TODO: extend the class to a generic pupil in order to
    # load an user defined pupil (or a elt like one) 
    #SLM PUPIL MASK PARAMETERS IN PIXEL
    PUPIL_CENTER = (581, 875)#YX
    PUPIL_RADIUS = 571
    FRAME_SHAPE = (1152, 1920)
    
    #CUSTOM ELT LIKE PUPIL PARAMETERS SCALED TO SLM 
    SPIDER_DIM = 6
    SPIDER_ANGLE = 60*np.pi/180
    OBSTRUCTION_RADIUS = 161
    
    def __init__(self):
        self._logger = logging.getLogger("PupilMask")
    
    #@cached_property
    def circular_pupil_mask(self):
        
        cmask = CircularMask(
            frameShape = self.FRAME_SHAPE,
            maskRadius = self.PUPIL_RADIUS,
            maskCenter = self.PUPIL_CENTER)
        return cmask
    
    #@cached_property
    def annular_pupil_mask(self):
        
        amask = AnnularMask(
            frameShape= self.FRAME_SHAPE,
            maskRadius= self.PUPIL_RADIUS,
            maskCenter= self.PUPIL_CENTER,
            inRadius= self.OBSTRUCTION_RADIUS)
        return amask
    
    #@cached_property
    def custom_elt_like_pupil_mask(self):
               
        emask = self.annular_pupil_mask()
        pupil_mask = emask.mask()
        
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
        
        return emask
    
    #@cached_property
    def elt_pupil_mask(self, fname):
        print('here1')
        original_elt_mask = self._get_elt_pupil_from_idl_file_data(fname)
        pupil_mask_on_slm_frame = self._get_rescaled_mask_to_slm_frame(original_elt_mask)
        emask = CircularMask(
            frameShape = self.FRAME_SHAPE,
            maskRadius = self.PUPIL_RADIUS,
            maskCenter = self.PUPIL_CENTER)
        temp_mask = emask.mask() 
        # WHY do I need to specify [:,:] to assign the content of 
        # pupil_mask_on_slm_frame to emask.mask() ???
        temp_mask[:,:] = pupil_mask_on_slm_frame.astype(bool)
        return emask
    
    def _get_elt_pupil_from_idl_file_data(self, fname):
        from astropy.io import fits
        #TODO: could be useful to save the pixel pitch of the original file
        # and rescaled for the new slm pupil mask
        #header = fits.getheader(fname)
        print('here2')
        hduList = fits.open(fname)
        elt_idl_mask = hduList[1].data.astype(bool)
        elt_mask = elt_idl_mask.copy()    
        elt_mask[elt_idl_mask == True] = False
        elt_mask[elt_idl_mask == False] = True
        return elt_mask
    
    def _get_rescaled_mask_to_slm_frame(self, original_pupil_mask):
        from scipy.interpolate import RegularGridInterpolator
        
        new_size = 2 * self.PUPIL_RADIUS
        x_original = np.linspace(0, 1, original_pupil_mask.shape[0])
        y_original = np.linspace(0, 1, original_pupil_mask.shape[1])
        original_grid = (x_original, y_original)
        
        x_new = np.linspace(0, 1, new_size)
        y_new = np.linspace(0, 1, new_size)
        x_new_grid, y_new_grid = np.meshgrid(x_new, y_new)
        new_points = np.column_stack([y_new_grid.ravel(), x_new_grid.ravel()])
        
        interpolator = RegularGridInterpolator(original_grid, original_pupil_mask, method='linear')
        
        interpolated_pupil = interpolator(new_points).reshape(new_size, new_size)
        
        rescaled_pupil = (interpolated_pupil > 0.5).astype(int)
        
        pupil_on_slm_frame = np.ones(self.FRAME_SHAPE)
        #TODO: rise error if the shift exceed the slm frame size
        top_left = self.PUPIL_CENTER[0] - self.PUPIL_RADIUS
        bottom_left = self.PUPIL_CENTER[1] - self.PUPIL_RADIUS
        print(pupil_on_slm_frame.shape)
        print(rescaled_pupil.shape)
        pupil_on_slm_frame[top_left:top_left + new_size,
                            bottom_left : bottom_left + new_size ] = rescaled_pupil
        return pupil_on_slm_frame
        