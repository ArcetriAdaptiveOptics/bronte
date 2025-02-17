from pysilico import camera
from plico_dm import deformableMirror
from functools import cached_property

from bronte import package_data
from bronte.wfs.slm_rasterizer import SlmRasterizer
from bronte.types.slm_pupil_mask_generator import SlmPupilMaskGenerator
from bronte.utils.camera_master_bkg import CameraMasterMeasurer

class BaseFactory():
    
    ELT_PUPIL_TAG = None    #'EELT480pp0.0803m_obs0.283_spider2023'
    SHWFS_BKG_TAG = '250211_135800'
    SH_FRAMES2AVERAGE  = 1
    
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
    def sh_camera(self):
        return camera('193.206.155.69', 7110)

    @cached_property
    def psf_camera(self):
        return camera('193.206.155.69', 7100)

    @cached_property
    def deformable_mirror(self):
        return deformableMirror('193.206.155.69', 7010)
    
    @cached_property
    def slm_pupil_mask(self):
        return self._create_slm_pupil_mask()
    
    @cached_property
    def slm_rasterizer(self):
        return SlmRasterizer(self.slm_pupil_mask)
    
    @cached_property
    def sh_camera_master_bkg(self):
        master_bkg = None
        if self.SHWFS_BKG_TAG is not None:
            master_bkg, texp = CameraMasterMeasurer.load_master(self.SHWFS_BKG_TAG)
            self.sh_camera.setExposureTime(texp)
        return master_bkg