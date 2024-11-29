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
        
        self.fig, self.axs = plt.subplots(2, figsize=(10, 10))
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
        self.output_frame.pixels = sh_camera_frame
        self.output_frame.generation_time = self.current_time

        self._plot(sh_camera_frame, phase_screen_to_raster)

    def run_check(self, time_step):
        return True
    
    def _plotOld(self, sh_camera_frame, phase_screen_to_raster):
        if self.first:
            self.img0 = self.axs[0].imshow(sh_camera_frame)
            self.img1 = self.axs[1].imshow(phase_screen_to_raster)
            self.first = False
        else:
            self.img0.set_data(sh_camera_frame)
            self.img0.autoscale()
            self.img1.set_data(phase_screen_to_raster)
            self.img1.autoscale()
#            self.img.set_clim(frame.min(), frame.max())
        self.fig.canvas.draw()
        plt.pause(0.001)       
        
        
    def _plotNo(self, sh_camera_frame, phase_screen_to_raster):
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
            self.colorbar0.draw_all()
            self.colorbar1.draw_all()
        
        # Ridisegna la figura
        self.fig.canvas.draw()
        plt.pause(0.001)

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
