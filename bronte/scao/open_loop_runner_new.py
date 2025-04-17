import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np
from bronte.startup import set_data_dir
from bronte.types.shwfs_device_manager import ShwfsDeviceManager
from specula.display.modes_display import ModesDisplay
from specula.display.slopec_display import SlopecDisplay
from bronte.package_data import telemetry_folder
from astropy.io import fits
from bronte.utils.retry_on_zmqrpc_timeout_error import retry_on_timeout
from bronte.utils.set_basic_logging import get_logger
from arte.utils.decorator import logEnterAndExit 

class OpenLoopRunner():
    
    LOOP_TYPE = 'OPEN'
    
    def __init__(self, open_loop_factory):
        
        self._logger = get_logger("OpenLoopRunner")
        self._factory = open_loop_factory
        self._target_device_idx = self._factory._target_device_idx
        self._load_slope_computer()
        self._load_reconstructor()
        self._setup_bench_devices()
        self._set_inputs()
        self._define_groups()
        self._initialize_telemetry()
    
    @logEnterAndExit("Loading SlopePC...",
                      "SlopePC loaded.", level='debug')
    def _load_slope_computer(self):
        self._subapdata = self._factory.subapertures_set
        self._slopec = self._factory.slope_computer
        self._nslopes = self._subapdata.n_subaps * 2 
    
    @logEnterAndExit("Loading Reconstructor...",
                      "Reconstructor loaded.", level='debug')
    def _load_reconstructor(self):
        
        self._rec = self._factory.reconstructor
        
    @logEnterAndExit("Setting SHWFS Device...",
                      "SHWFS Device set.", level='debug')
    def _setup_bench_devices(self):
        
        self._shwfs_device = ShwfsDeviceManager(self._factory)
        
    @logEnterAndExit("Setting ProcessingObjects inputs ...",
                      "ProcessingObjects inputs set.", level='debug')
    def _set_inputs(self):
        
        self._slopec.inputs['in_pixels'].set(self._shwfs_device.outputs['out_pixels'])
        self._rec.inputs['in_slopes'].set(self._slopec.outputs['out_slopes'])
        # factor 2 moved on factory while loading recmat 
        #self._rec.modes.value = 2 * self._rec.modes.value.copy()
        
        self._modes_disp = ModesDisplay()
        self._modes_disp.inputs['modes'].set(self._rec.modes)
        self._slopes_disp = SlopecDisplay(disp_factor = 3)
        self._slopes_disp.inputs['slopes'].set(self._slopec.outputs['out_slopes'])
        self._slopes_disp.inputs['subapdata'].set(self._factory.subapertures_set)
        
    def _define_groups(self):
        
        group1 = [self._shwfs_device]
        group2 = [self._slopec]
        group3 = [self._rec]
        group4 = [self._modes_disp, self._slopes_disp]

        self._groups = [group1, group2, group3, group4]
    
    @logEnterAndExit("Setting telemetry buffer...",
                      "Telemetry buffer set.", level='debug')   
    def _initialize_telemetry(self):
        
        self._slopes_vector_list = []
        self._zc_delta_modal_command_list = []
    
    @logEnterAndExit("Updating telemetry buffer...",
                  "Telemetry buffer updated.", level='debug')  
    def _update_telemetry(self):
        
        specula_slopes = self._groups[1][0].outputs['out_slopes']
        self._slopes_vector_list.append(specula_slopes.slopes.copy())        
        specula_delta_commands_in_nm = self._groups[2][0].modes.value
        self._zc_delta_modal_command_list.append(
            specula_delta_commands_in_nm*1e-9)

    @logEnterAndExit("Starting Open Loop...",
              "Open Loop Terminated.", level='debug')
    def run(self, Nsteps = 30):

        self._n_steps = Nsteps
        self.time_step = self._factory.TIME_STEP_IN_SEC
        tf = (self._n_steps-1)*self.time_step
        
        for group in self._groups:
            for obj in group:
                obj.loop_dt = self.time_step * 1e9
                obj.setup(obj.loop_dt, self._n_steps)
    
        for step in range(self._n_steps):
            t = 0 + step * self.time_step
            self._logger.info(
                "\n+ Open Loop @ time: %f/%f s\t & step: %d/%d" % (t, tf, step+1, Nsteps))
            for group in self._groups:
                for obj in group:  
                    obj.check_ready(t*1e9)
                    self._logger.info(f"Triggering {str(obj)}")
                    obj.trigger()
                    self._logger.info(f"PostTriggering {str(obj)}")
                    obj.post_trigger()

            self._update_telemetry()
            
        for group in self._groups:
            for obj in group:
                obj.finalize()

    @logEnterAndExit("Saving data...",
                  "Data saved.", level='debug')
    def save_telemetry(self, fname):
                
        psf_camera_texp = retry_on_timeout(self._factory.psf_camera.exposureTime)
        psf_camera_fps = retry_on_timeout(self._factory.psf_camera.getFrameRate)
        shwfs_texp = retry_on_timeout(self._factory.sh_camera.exposureTime)
        shwfs_fps = retry_on_timeout(self._factory.sh_camera.getFrameRate)
        
        file_name = telemetry_folder() / (fname + '.fits')
        hdr = fits.Header()
        
        hdr['LOOP'] = self.LOOP_TYPE
        
        # FILE TAG DEPENDENCY
        hdr['SUB_TAG'] = self._factory.SUBAPS_TAG
        hdr['REC_TAG'] = self._factory.REC_MAT_TAG
        
        if self._factory.ELT_PUPIL_TAG is not None:
            hdr['ELT_TAG'] = self._factory.ELT_PUPIL_TAG
        else:
            hdr['ELT_TAG'] = 'NA'
        
        if self._factory.MODAL_OFFSET_TAG is not None:
            hdr['OFF_TAG'] = self._factory.MODAL_OFFSET_TAG
        else:
            hdr['OFF_TAG'] = 'NA'
        
        # LOOP PARAMETERS
        hdr['TSTEP_S'] = self.time_step
        hdr['INT_GAIN'] = self._factory.INT_GAIN
        hdr['INT_DEL'] = self._factory.INT_DELAY
        hdr['N_STEPS'] = self._n_steps
        hdr['N_MODES'] = self._factory.N_MODES_TO_CORRECT
        
        #HARDWARE PARAMETERS
        hdr['SLM_RAD'] = self._factory.SLM_PUPIL_RADIUS # in pixels
        hdr['SLM_YX'] = str(self._factory.SLM_PUPIL_CENTER) # YX pixel coordinates
        hdr['SHPX_THR'] = self._factory.SH_PIX_THR # in ADU
        hdr['PC_TEXP'] = psf_camera_texp # in ms
        hdr['PC_FPS'] = psf_camera_fps
        hdr['SH_TEXP'] = shwfs_texp # in ms
        hdr['SH_FPS'] = shwfs_fps
        
        fits.writeto(file_name, np.array(self._slopes_vector_list), hdr)
        fits.append(file_name, np.array(self._zc_delta_modal_command_list))

        
    @staticmethod
    def load_telemetry(fname):
        set_data_dir()
        file_name = telemetry_folder() / (fname + '.fits')
        header = fits.getheader(file_name)
        hduList = fits.open(file_name)

        slopes_vect = hduList[0].data
        zc_delta_modal_commands = hduList[1].data
        
        return  header, slopes_vect, zc_delta_modal_commands

