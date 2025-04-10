from specula import np, cpuArray
from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.pixels import Pixels
from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField
import time
from bronte.utils.data_cube_cleaner import DataCubeCleaner
import matplotlib.pyplot as plt
from bronte.utils.set_basic_logging import get_logger
from arte.utils.decorator import logEnterAndExit 

class TestbenchDeviceManager(BaseProcessingObj):
    
    SLM_RESPONSE_TIME = 0.005
    
    def __init__(self, factory, target_device_idx=None, precision=None, do_plots=True):
        
        super().__init__(target_device_idx, precision)
        self._logger = get_logger("TestBenchDeviceManager")
        self._slm = factory.deformable_mirror
        self._sh_camera = factory.sh_camera
        self._sh_camera_bkg = factory.sh_camera_master_bkg
        self._slm_raster = factory.slm_rasterizer
        self.output_frame = Pixels(*self._sh_camera.shape())
        self.outputs['out_pixels'] = self.output_frame
        self.inputs['ef'] = InputValue(type=ElectricField)
        self._Nframes = factory.SH_FRAMES2AVERAGE
        self._do_plots = do_plots
        self.first = True
        if factory.modal_offset is None:
            self._offset_cmd = 0.
        else:
            self._offset_cmd = self._get_offset_command(factory.modal_offset)
        
        if self._do_plots:
            self.fig, self.axs = plt.subplots(2, figsize=(10, 10))
    
    @logEnterAndExit("Triggering Test Bench Devices...",
                  "Test Bench Devices Triggered.", level='debug')
    def trigger_code(self):
        ef = self.local_inputs['ef']
        phase_screen = cpuArray(ef.phaseInNm) * 1e-9
        
        phase_screen_to_raster = self._slm_raster.get_recentered_phase_screen_on_slm_pupil_frame(phase_screen)
        self._command = self._slm_raster.reshape_map2vector(phase_screen_to_raster) + self._offset_cmd
        self._slm.set_shape(self._command)
        time.sleep(self.SLM_RESPONSE_TIME)
        
        #TODO: manage the different integration times for the each wfs group
        # how to reproduce faint source? shall we play with the texp of the hardware?
        #TODO: load dark and bkg for sh frame reduction
        
        sh_camera_frame = self._sh_camera.getFutureFrames(self._Nframes).toNumpyArray()
        
        if self._Nframes > 1:
            if self._sh_camera_bkg is not None:
                sh_camera_frame = DataCubeCleaner.get_master_from_rawCube(
                    sh_camera_frame, self._sh_camera_bkg)
            else:
                sh_camera_frame = sh_camera_frame.mean(axis=-1) 
        else:
            if self._sh_camera_bkg is not None:
                sh_camera_frame = sh_camera_frame.astype(float) - self._sh_camera_bkg.astype(float)
                sh_camera_frame[sh_camera_frame < 0.] = 0.
            else:
                pass
              
        self.output_frame.pixels = sh_camera_frame
        self.output_frame.generation_time = self.current_time

        if self._do_plots:
            self._plot(sh_camera_frame, phase_screen_to_raster)

    def run_check(self, time_step):
        return True
    
    def setup(self, time_step):
        return True

    def _plot(self, sh_camera_frame, phase_screen_to_raster):
        if self.first:
            # Prima chiamata: crea le immagini e le colorbar
            self.img0 = self.axs[0].imshow(sh_camera_frame, aspect='auto')
            self.colorbar0 = self.fig.colorbar(self.img0, ax=self.axs[0])
            
            self.img1 = self.axs[1].imshow(phase_screen_to_raster, aspect='auto')
            self.colorbar1 = self.fig.colorbar(self.img1, ax=self.axs[1])
            
            self.first = False
        else:
            # Aggiorna i dati e i limiti delle immagini
            self.img0.set_data(sh_camera_frame)
            self.img0.set_clim(vmin=sh_camera_frame.min(), vmax=sh_camera_frame.max())
            
            self.img1.set_data(phase_screen_to_raster)
            self.img1.set_clim(vmin=phase_screen_to_raster.min(), vmax=phase_screen_to_raster.max())
            
            # Aggiorna le colorbar
            self.colorbar0.update_normal(self.img0)
            self.colorbar1.update_normal(self.img1)
            
        # Ridisegna la figura
        self.fig.canvas.draw()
        plt.pause(0.001)
    
    def _get_offset_command(self, modal_offset):
        wfz_offset = self._slm_raster.zernike_coefficients_to_raster(modal_offset)
        wf_offset = wfz_offset.toNumpyArray()
        cmd_offset = self._slm_raster.reshape_map2vector(wf_offset)
        return cmd_offset