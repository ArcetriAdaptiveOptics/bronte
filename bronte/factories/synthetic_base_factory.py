from functools import cached_property
from bronte import package_data
from bronte.wfs.slm_rasterizer import SlmRasterizer
from bronte.types.slm_pupil_mask_generator import SlmPupilMaskGenerator

class SyntheticBaseFactory():
    
    ELT_PUPIL_TAG = None    #'EELT480pp0.0803m_obs0.283_spider2023'
    #SH_FRAMES2AVERAGE  = 1
    
    def __init__(self):
        self._target_device_idx= -1
        self._set_up_basic_logging()
           
    def _set_up_basic_logging(self):
        import importlib
        import logging
        importlib.reload(logging)
        FORMAT = '%(asctime)s:%(levelname)s:%(name)s  %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=FORMAT)   

    def _create_slm_pupil_mask(self):
        spm = SlmPupilMaskGenerator()
        if self.ELT_PUPIL_TAG is not None:
            return spm.elt_pupil_mask(
                package_data.elt_pupil_folder()/(self.ELT_PUPIL_TAG + '.fits'))
        else:
            return spm.circular_pupil_mask()

    @cached_property
    def slm_pupil_mask(self):
        return self._create_slm_pupil_mask()
    
    @cached_property
    def slm_rasterizer(self):
        return SlmRasterizer(self.slm_pupil_mask)