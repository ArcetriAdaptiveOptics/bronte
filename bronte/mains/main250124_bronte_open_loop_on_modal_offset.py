from bronte import startup
from bronte.mains import main240802_ao_test
from arte.types.zernike_coefficients import ZernikeCoefficients
import numpy as np
     
class OpenLoopRunner():
    
    def __init__(self):
        
        self._factory =  startup.startup()
        flat = np.zeros(1920*1152)
        self._factory.deformable_mirror.set_shape(flat)
        self._factory.sh_camera.setExposureTime(8)
        self._zc_offset = None
        self._tao = main240802_ao_test.TestAoLoop(self._factory)
        self._tao.reset_wavefront_disturb()
        self._tao._factory.pure_integrator_controller._gain = 0
    
    def set_modal_offset_as_numpy_array(self, modal_offset):
        self._zc_offset = ZernikeCoefficients.fromNumpyArray(modal_offset)
        self._tao._factory.rtc.set_modal_offset(self._zc_offset)
    
    def run(self, steps = 30):
        if self._zc_offset is None:
            self.set_modal_offset_as_numpy_array(np.array([0.,0.,0.]))
        self._tao.loop(steps)
        
    def save_telemetry(self, ftag):
        
        self._tao.save_telemetry(ftag)