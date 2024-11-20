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
    
    #ELT LIKE PUPIL PARAMETERS SCALED TO SLM 
    SPIDER_DIM = 6
    SPIDER_ANGLE = 60*np.pi/180
    OBSTRUCTION_RADIUS = 161
    
    def __init__(self):
        self._logger = logging.getLogger("PupilMask")
    
    @cached_property
    def circular_pupil_mask(self):
        
        cmask = CircularMask(
            frameShape = self.FRAME_SHAPE,
            maskRadius = self.PUPIL_RADIUS,
            maskCenter = self.PUPIL_CENTER)
        return cmask
    
    @cached_property
    def annular_pupil_mask(self):
        
        amask = AnnularMask(
            frameShape= self.FRAME_SHAPE,
            maskRadius= self.PUPIL_RADIUS,
            maskCenter= self.PUPIL_CENTER,
            inRadius= self.OBSTRUCTION_RADIUS)
        return amask
    
    @cached_property
    def elt_like_pupil_mask(self):
               
        emask = self.annular_pupil_mask
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
        