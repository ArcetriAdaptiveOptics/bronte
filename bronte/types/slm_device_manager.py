from specula import np, cpuArray
from specula.base_processing_obj import BaseProcessingObj
# from specula.data_objects.pixels import Pixels
from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField
import time
import matplotlib.pyplot as plt
from bronte.utils.set_basic_logging import get_logger
from arte.utils.decorator import logEnterAndExit, override 

class SlmDeviceManager(BaseProcessingObj):
    
    SLM_RESPONSE_TIME = 0.005
    
    def __init__(self, factory,
                 setup_cmd = None,
                 load_huge_tilt_under_mask = False,
                 target_device_idx = None,
                 precision = None,
                 do_plots = True):
        
        super().__init__(target_device_idx, precision)
        self._logger = get_logger("SlmDeviceManager")
        self._setup_cmd = setup_cmd
        self._slm = factory.deformable_mirror
        self._slm_raster = factory.slm_rasterizer
        self.inputs['ef'] = InputValue(type=ElectricField)
        self._do_plots = do_plots
        self._load_tilt_under_mask = load_huge_tilt_under_mask
        self.first = True
        if factory.modal_offset is None:
            self._offset_cmd = 0.
        else:
            self._offset_cmd = self._get_offset_command(factory.modal_offset)
        
        if self._do_plots:
            self.fig, self.axs = plt.subplots(1, figsize=(8, 4))
            
    @logEnterAndExit("Triggering SLM...",
                  "SLM Triggered.", level='debug')
    def trigger_code(self):
        
        ef = self.local_inputs['ef']
        phase_screen = cpuArray(ef.phaseInNm) * 1e-9
        
        phase_screen_to_raster = self._slm_raster.get_recentered_phase_screen_on_slm_pupil_frame(phase_screen)
        if self._load_tilt_under_mask is True:
            phase_screen_to_raster = self._slm_raster.load_a_tilt_under_pupil_mask()
        
        self._command = self._slm_raster.reshape_map2vector(phase_screen_to_raster) + self._offset_cmd
        
        if self._load_tilt_under_mask is True:
            self._slm.set_shape(self._command.data) 
        else:
            self._slm.set_shape(self._command)
        time.sleep(self.SLM_RESPONSE_TIME)
        
        if self._do_plots:
            self._plot(phase_screen_to_raster)

    def run_check(self, time_step):
        return True
    
    # @override
    # @logEnterAndExit("SLM ProcObj  Setup...",
    #               "SLM ProcObj Setup accomplished.", level='debug')
    # def setup(self, loop_dt, loop_niters):
    #
    #     if self._setup_cmd is None:
    #         self._setup_cmd = self._slm.get_shape()
    #     else:
    #         self._slm.set_shape(self._setup_cmd)
    #         time.sleep(self.SLM_RESPONSE_TIME)
    #
    #     self._loop_dt = loop_dt
    #     self._loop_niters = loop_niters
    #     if self.target_device_idx >= 0:
    #         self._target_device.use()
    #     for name, input in self.inputs.items():
    #         if input.get(self.target_device_idx) is None and not input.optional:
    #             raise ValueError(f'Input {name} for object {self} has not been set')
    #     return True
    
    #TODO: adjust plot
    def _plot(self, phase_screen_to_raster):
        if self.first:
            # Prima chiamata: crea le immagini e le colorbar
            self.img0 = self.axs.imshow(phase_screen_to_raster, aspect='auto')
            self.colorbar0 = self.fig.colorbar(self.img0, ax=self.axs, label = 'm')
            
            self.first = False
        else:
            # Aggiorna i dati e i limiti delle immagini
            self.img0.set_data(phase_screen_to_raster)
            self.img0.set_clim(vmin=phase_screen_to_raster.min(), vmax=phase_screen_to_raster.max())
            
            # Aggiorna le colorbar
            self.colorbar0.update_normal(self.img0)
            
        # Ridisegna la figura
        self.fig.canvas.draw()
        plt.pause(0.001)
    
    def _get_offset_command(self, modal_offset):
        wfz_offset = self._slm_raster.zernike_coefficients_to_raster(modal_offset)
        wf_offset = wfz_offset.toNumpyArray()
        cmd_offset = self._slm_raster.reshape_map2vector(wf_offset)
        return cmd_offset