from specula import np, cpuArray
from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.pixels import Pixels
from specula.connections import InputValue
from specula.data_objects.ef import ElectricField
import time

class TestbenchDeviceManager(BaseProcessingObj):
    
    SLM_RESPONSE_TIME = 0.005
    
    def __init__(self, factory, target_device_idx=None, precision=None):
        super().__init__(target_device_idx, precision)
        
        self._slm = factory.deformable_mirror
        self._sh_camera = factory.sh_camera
        self._slm_raster = factory.slm_rasterizer
        self.output_frame = Pixels(*self._sh_camera.shape())
        self.outputs['out_pixels'] = self.output_frame
        self.inputs['ef'] = InputValue(type=ElectricField)
    
    def trigger_code(self):
       
        ef = self.local_inputs['ef']
        phase_screen = cpuArray(ef.phaseInNm) * 1e-9
        phase_screen_to_raster = self._slm_raster.get_recentered_phase_screen_on_slm_pupil_frame(phase_screen)
        self._command = self._slm_raster.reshape_map2vector(phase_screen_to_raster) 
        
        self._slm.set_shape(self._command)
        time.sleep(self.SLM_RESPONSE_TIME)
        
        sh_camera_frame = self._sh_camera.getFutureFrames(1).toNumpyArray()
        self.output_frame.pixels = sh_camera_frame
        self.output_frame.generation_time = self.current_time

    def run_check(self, time_step):
        return True
    

