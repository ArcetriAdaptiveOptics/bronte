from bronte import startup
from bronte.mains import main240802_ao_test
from bronte.telemetry_trash.display_telemetry_data import DisplayTelemetryData
from arte.types.zernike_coefficients import ZernikeCoefficients
import numpy as np
     
class OpenLoopRunner():
    
    def __init__(self):
        
        self._factory =  startup.startup()
        flat = np.zeros(1920*1152)
        self._factory.deformable_mirror.set_shape(flat)
        self._factory.sh_camera.setExposureTime(8)
        self._zc_distrub = None
        self._tao = main240802_ao_test.TestAoLoop(self._factory)
        self._tao.reset_wavefront_disturb()
        self._tao._factory.pure_integrator_controller._gain = 0
    
   
    def set_wavefront_disturb_from_numpy_array(self, modal_coefficients):
        
        self._zc_distrub = ZernikeCoefficients.fromNumpyArray(modal_coefficients)
        wf_disturb = self._factory.slm_rasterizer.zernike_coefficients_to_raster(self._zc_disturb)
        self._tao._factory.rtc.set_wavefront_disturb(wf_disturb)
        cmd = self._factory.slm_rasterizer.reshape_map2vector(wf_disturb.toNumpyArray())
        self._factory.deformable_mirror.set_shape(cmd)
    
    def run(self, steps = 30):
        if self._zc_offset is None:
            self.set_modal_offset_from_numpy_array(np.array([0.,0.,0.]))
       
        self._tao.loop(steps)
        
    def save_telemetry(self, ftag):
        
        self._tao._delete_short_exp_psf()
        self._tao.save_telemetry(ftag)
        

def main():

    bf = startup.startup()
    flattening_modal_coefficients, _ = DisplayTelemetryData.load_modal_offset('250203_134800')
    modal_command = flattening_modal_coefficients.copy()
    modal_command[0] += 1e-6
    olr = OpenLoopRunner()
    olr.set_wavefront_disturb_from_numpy_array(modal_command)
    olr.run(steps=30)
    olr.save_telemetry('250203_135500') 