import specula
specula.init(-1, precision=1)
from specula import np, cpuArray
from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.pixels import Pixels
from specula.connections import InputValue
from specula.data_objects.ef import ElectricField
import time
import matplotlib.pyplot as plt

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
        
        self.fig, self.axs = plt.subplots(2)
        self.first = True

    def trigger_code(self):
        ef = self.local_inputs['ef']
        phase_screen = cpuArray(ef.phaseInNm) * 1e-9
        
        phase_screen_to_raster = self._slm_raster.get_recentered_phase_screen_on_slm_pupil_frame(phase_screen)
        command = self._slm_raster.reshape_map2vector(phase_screen_to_raster)
        self._slm.set_shape(command)
        time.sleep(self.SLM_RESPONSE_TIME)
        #TODO: manage the different integration times for the each wfs group
        # how to reproduce faint source? shall we play with the texp of the hardware?
        sh_camera_frame = self._sh_camera.getFutureFrames(1, 1).toNumpyArray()

        if self.first:
            self.img0 = self.axs[0].imshow(sh_camera_frame)
            self.img1 = self.axs[1].imshow(phase_screen_to_raster)
            self.first = False
        else:
            self.img0.set_data(sh_camera_frame)
            self.img1.set_data(phase_screen_to_raster)
#            self.img.set_clim(frame.min(), frame.max())
        self.fig.canvas.draw()
        plt.pause(0.001)       
        
        self.output_frame.pixels = sh_camera_frame
        self.output_frame.generation_time = self.current_time

    def run_check(self, time_step):
        return True