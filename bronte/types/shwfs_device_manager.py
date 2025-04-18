from specula import np, cpuArray
from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.pixels import Pixels
import time
from bronte.utils.data_cube_cleaner import DataCubeCleaner
import matplotlib.pyplot as plt
from bronte.utils.set_basic_logging import get_logger
from arte.utils.decorator import logEnterAndExit, override 

class ShwfsDeviceManager(BaseProcessingObj):
     
    SLM_RESPONSE_TIME = 0.005
    
    def __init__(self, factory, target_device_idx=None, precision=None, do_plots=True):
        
        super().__init__(target_device_idx, precision)
        self._logger = get_logger("ShwfsDeviceManager")
        #self._factory = factory
        self._sh_camera = factory.sh_camera
        self._sh_camera_bkg = factory.sh_camera_master_bkg
        self.output_frame = Pixels(*self._sh_camera.shape())
        self.outputs['out_pixels'] = self.output_frame
        self._Nframes = factory.SH_FRAMES2AVERAGE
        self._do_plots = do_plots
        self.first = True
        
        if self._do_plots:
            self.fig, self.axs = plt.subplots(1, figsize=(12, 12))

    @logEnterAndExit("Triggering SHWFS...",
                  "SHWFS Triggered.", level='debug')
    def trigger_code(self):
        
        #TODO: manage the different integration times for the each wfs group
        # how to reproduce faint source? shall we play with the texp of the hardware?
        #TODO: load dark and bkg for sh frame reduction
        #print("acquiring frame ")
        sh_camera_frame = self._sh_camera.getFutureFrames(self._Nframes).toNumpyArray()
        #sh_camera_frame= np.zeros((2048,2048))
        # print("frame acquired")
        # print(sh_camera_frame.shape)
        # print(sh_camera_frame.dtype)
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
            self._plot(sh_camera_frame)

    def run_check(self, time_step):
        return True
    
    # @override
    # @logEnterAndExit("SHWFS ProcObj Setup...",
    #           "SHWFS ProcObj Setup accomplished.", level='debug')
    # def setup(self, loop_dt, loop_niters):
    #
    #     self._factory._load_sh_camera_master_bkg()
    #     self._sh_camera_bkg = self._factory.sh_camera_master_bkg
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
    def _plot(self, sh_camera_frame):
        if self.first:
            # Prima chiamata: crea le immagini e le colorbar
            self.img0 = self.axs.imshow(sh_camera_frame, aspect='auto')
            self.colorbar0 = self.fig.colorbar(self.img0, ax=self.axs, label = 'ADU')
            
            self.first = False
        else:
            # Aggiorna i dati e i limiti delle immagini
            self.img0.set_data(sh_camera_frame)
            self.img0.set_clim(vmin=sh_camera_frame.min(), vmax=sh_camera_frame.max())
            
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