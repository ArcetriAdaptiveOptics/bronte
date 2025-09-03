from specula import np, cpuArray
from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.pixels import Pixels
from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField
from bronte.utils.data_cube_cleaner import DataCubeCleaner
from bronte.utils.set_basic_logging import get_logger
from arte.utils.decorator import logEnterAndExit


class PsfCameraDeviceManager(BaseProcessingObj):
    
    def __init__(self, factory, target_device_idx = None, precision = None):
        
        super().__init__(target_device_idx, precision)
        self._logger = get_logger("PsfCameraDeviceManager")
        
        self._psf_camera = factory.psf_camera
        self._psf_camera_bkg = factory.psf_camera_master_bkg
        
        self.output_frame = Pixels(*self._psf_camera.shape())
        self.outputs['out_pixels'] = self.output_frame
        self.inputs['ef'] = InputValue(type=ElectricField)
        
        self._Nframes = factory.PSF_FRAMES2AVERAGE
        
    @logEnterAndExit("Triggering PSF Camera Device...",
                  "PSF Camera Device Triggered.", level='debug')
    def trigger_code(self):
        
        #ef = self.local_inputs['ef']

        psf_camera_frame = self._psf_camera.getFutureFrames(self._Nframes, timeoutSec=30).toNumpyArray()
        
        if self._Nframes > 1:
            if self._psf_camera_bkg is not None:        
                psf_camera_frame = psf_camera_frame[:,:,3:]
                psf_camera_frame = DataCubeCleaner.get_master_from_rawCube(psf_camera_frame, self._psf_camera_bkg)
            else:
                psf_camera_frame = psf_camera_frame.mean(axis=-1) 
        else:
            if self._psf_camera_bkg is not None:
                psf_camera_frame = psf_camera_frame.astype(float) - self._psf_camera_bkg.astype(float)
                psf_camera_frame[psf_camera_frame < 0.] = 0.
            else:
                pass
              
        self.output_frame.pixels = psf_camera_frame
        self.output_frame.generation_time = self.current_time

    def run_check(self, time_step):
        return True