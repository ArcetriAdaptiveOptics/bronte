import numpy as np
from arte.types.mask import BaseMask
from arte.types.slopes import Slopes
from arte.types.zernike_coefficients import ZernikeCoefficients
import logging
from arte.utils.decorator import logEnterAndExit
from arte.utils.circular_buffer import NumpyCircularBuffer

class ScaoRealTimeComputer:

    def __init__(self, wfs_camera, slope_computer, deformable_mirror, modal_decomposer, controller, slm_rasterizer):
        self._wfs_camera = wfs_camera
        self._sc = slope_computer
        self._dm = deformable_mirror
        self._md = modal_decomposer
        self._controller = controller
        self._slm_rasterizer = slm_rasterizer
        self._logger = logging.getLogger("ScaoRealTimeComputer")
        self.reset_modal_offset()
        self.reset_wavefront_disturb()
        self._initialize_telemetry_buffers()

        #self.pupil_radius = self._slm_rasterizer.slm_pupil_mask.radius() * 9.2e-6
        
        self._slope_unit_2_rad = self._get_geometrical_factor_for_slopes_conversion()#6.23e-3

        self._subap_mask, self._zernike_mask = self._sc._compute_masks()

    def _initialize_telemetry_buffers(self):
        how_many = 100
        self._delta_modal_command_buffer = NumpyCircularBuffer(
            how_many, (self._md.nModes,), float)

    def _update_telemetry_buffers(self, zc):
        self._delta_modal_command_buffer.store(zc.toNumpyArray())
    
    def _get_geometrical_factor_for_slopes_conversion(self):
        f_la = 8.31477e-3
        d_la = 144e-6
        # relay_mag = 150e-3/250e-3
        # alpha = relay_mag * 0.5*d_la*d_la/f_la
        alpha = 0.5*d_la*d_la/f_la
        return alpha
    
    @logEnterAndExit("Computing Zernike coefficients...", "Zernike coefficients computed", level='debug')
    def _compute_zernike_coefficients(self):
        # create Slopes object 
        #TODO: check for the correct scale factor for the slopes to get them in rad
        sl = Slopes(self._sc.slopes()[:, 0]*self._slope_unit_2_rad,
                    self._sc.slopes()[:, 1]*self._slope_unit_2_rad,
                    self._subap_mask)

        # use modal decomposer
        zc = self._md.measureZernikeCoefficientsFromSlopes(
            sl, self._zernike_mask, BaseMask(self._subap_mask))
        return zc-self.modal_offset

    def set_modal_offset(self, zernike_modal_offset):
        self._zc_offset = zernike_modal_offset

    def reset_modal_offset(self):
        self._zc_offset = ZernikeCoefficients(np.zeros(1, dtype=float))

    @property
    def modal_offset(self):
        return self._zc_offset

    def set_wavefront_disturb(self, wavefront_disturb_raster):
        self._wf_disturb = wavefront_disturb_raster

    @property
    def wavefront_disturb(self):
        return self._wf_disturb

    def reset_wavefront_disturb(self):
        self._wf_disturb = np.zeros((1152, 1920), dtype=int)

    @logEnterAndExit("Stepping...", "Stepped", level='debug')
    def step(self):
        # Acquire frame
        wfs_frame = self._wfs_camera.getFutureFrames(1, 1).toNumpyArray()
        # Use frame
        # TODO set background_frame for subtraction
        self._sc.upload_raw_frame(wfs_frame)
        # reconstruct Zernike coefficients
        zc = self._compute_zernike_coefficients()
        # temporal_filter
        zc_filtered = self._controller.process_delta_command(zc.toNumpyArray())
        # convert modal amplitudes in SLM shape
        slm_raster = self._slm_rasterizer.zernike_coefficients_to_raster(
            ZernikeCoefficients.fromNumpyArray(zc_filtered))# + self.modal_offset)
        # apply on slm
        self._dm.set_shape(
            self._slm_rasterizer.reshape_map2vector(slm_raster.toNumpyArray()+self.wavefront_disturb))
        # update telemetry buffers
        self._update_telemetry_buffers(zc)
