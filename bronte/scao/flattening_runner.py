import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np
from bronte.startup import set_data_dir
from bronte.types.slm_device_manager import SlmDeviceManager
from bronte.types.shwfs_device_manager import ShwfsDeviceManager
from specula.display.modes_display import ModesDisplay
from specula.display.slopec_display import SlopecDisplay
from bronte.package_data import telemetry_folder
from astropy.io import fits
from plico.rpc.zmq_remote_procedure_call import ZmqRpcTimeoutError
import time
from bronte.utils.set_basic_logging import get_logger
from arte.utils.decorator import logEnterAndExit 

class FlatteningRunner():
    
    LOOP_TYPE = 'FLATTENING'
    
    def __init__(self, flattening_factory, logger_name ="FlatteningRunner", xp=np):
        
        self._logger = get_logger(logger_name)
        self._factory = flattening_factory
        self._target_device_idx = self._factory._target_device_idx
        self._load_slope_computer()
        self._load_reconstructor()
        self._setup_control()
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
    
    @logEnterAndExit("Setting control...",
                      "Control set.", level='debug')
    def _setup_control(self):
        
        self._control = self._factory.integrator_controller
        self._dm = self._factory.virtual_deformable_mirror
    
    @logEnterAndExit("Setting Bench Devices...",
                      "Bench Devices set.", level='debug')
    def _setup_bench_devices(self):
        
        #self._factory.sh_camera.setExposureTime(self._factory._sh_texp)
        self._shwfs_device = ShwfsDeviceManager(self._factory)
        self._slm_device = SlmDeviceManager(self._factory)
    
    @logEnterAndExit("Setting ProcessingObjects inputs ...",
                      "ProcessingObjects inputs set.", level='debug')
    def _set_inputs(self):
        
        self._slopec.inputs['in_pixels'].set(self._shwfs_device.outputs['out_pixels'])
        self._rec.inputs['in_slopes'].set(self._slopec.outputs['out_slopes'])
        self._control.inputs['delta_comm'].set(self._rec.modes)
        self._dm.inputs['in_command'].set(self._control.out_comm)
        self._slm_device.inputs['ef'].set(self._dm.outputs['out_layer'])
        
        self._modes_disp = ModesDisplay()
        self._modes_disp.inputs['modes'].set(self._rec.modes)
        self._slopes_disp = SlopecDisplay(disp_factor = 3)
        self._slopes_disp.inputs['slopes'].set(self._slopec.outputs['out_slopes'])
        self._slopes_disp.inputs['subapdata'].set(self._factory.subapertures_set)
        
    def _define_groups(self):
        
        group1 = [self._shwfs_device]
        group2 = [self._slopec]
        group3 = [self._rec]
        group4 = [self._control, self._modes_disp, self._slopes_disp]
        group5 = [self._dm]
        group6 = [self._slm_device]

        self._groups = [group1, group2, group3, group4, group5, group6]
    
    @logEnterAndExit("Setting telemetry buffer...",
                      "Telemetry buffer set.", level='debug')   
    def _initialize_telemetry(self):
        
        self._slopes_vector_list = []
        self._zc_delta_modal_command_list = []
        self._zc_integrated_modal_command_list = []
    
    @logEnterAndExit("Updating telemetry buffer...",
                  "Telemetry buffer updated.", level='debug')  
    def _update_telemetry(self):
        
        specula_slopes = self._groups[1][0].outputs['out_slopes']
        self._slopes_vector_list.append(specula_slopes.slopes.copy())        
        specula_delta_commands_in_nm = self._groups[2][0].modes.value
        self._zc_delta_modal_command_list.append(
            specula_delta_commands_in_nm*1e-9)
        
        specula_integrated_commands_in_nm = self._groups[3][0].out_comm.value
        self._zc_integrated_modal_command_list.append(
            specula_integrated_commands_in_nm*1e-9)
    
    @logEnterAndExit("Starting Loop...",
              "Loop Terminated.", level='debug')
    def run(self, Nsteps = 30):

        # print(f'{self._dm.if_commands.shape=}')
        # print(f'{self._dm._ifunc.influence_function.shape=}')
        # print(f'{self._dm._ifunc.idx_inf_func[0].shape=}')
        # print(f'{self._dm._ifunc.idx_inf_func[1].shape=}')
        # print(f'{self._dm.layer.phaseInNm.shape=}')
        
        self._n_steps = Nsteps
        # time step of the simulated loop isnt it QM
        self.time_step = self._factory.TIME_STEP_IN_SEC
        tf = (self._n_steps-1)*self.time_step
        
        for group in self._groups:
            for obj in group:
                obj.loop_dt = self.time_step * 1e9
                #self._logger.info(self.time_step * 1e9)
                #obj.run_check(self.time_step)
                obj.setup(self.time_step * 1e9, self._n_steps)
    
        for step in range(self._n_steps):
            t = 0 + step * self.time_step
            self._logger.info(
                "\n+ Loop @ time: %f/%f s\t steps: %d/%d" % (t, tf, step+1, Nsteps))
            for group in self._groups:
                for obj in group:
                    # print('Before ', obj)
                    # print(f'{self._dm.if_commands.shape=}')
                    # print(f'{self._dm._ifunc.influence_function.shape=}')
                    # print(f'{self._dm._ifunc.idx_inf_func[0].shape=}')
                    # print(f'{self._dm._ifunc.idx_inf_func[1].shape=}')
                    # print(f'{self._dm.layer.phaseInNm.shape=}')
                        
                    obj.check_ready(t*1e9)
                    self._logger.info(f"Triggering {str(obj)}")
                    obj.trigger()
                    self._logger.info(f"PostTriggering {str(obj)}")
                    obj.post_trigger()

                    # print('After', obj)
                    # print(f'{self._dm.if_commands.shape=}')
                    # print(f'{self._dm._ifunc.influence_function.shape=}')
                    # print(f'{self._dm._ifunc.idx_inf_func[0].shape=}')
                    # print(f'{self._dm._ifunc.idx_inf_func[1].shape=}')
                    # print(f'{self._dm.layer.phaseInNm.shape=}')

            self._update_telemetry()
            
        for group in self._groups:
            for obj in group:
                obj.finalize()
                
    @logEnterAndExit("Saving data...",
                  "Data saved.", level='debug')
    def save_telemetry(self, fname):
        
        def retry_on_timeout(func, max_retries = 5000, delay = 0.001):
            '''Retries a function call if ZmqRpcTimeoutError occurs.'''
            for attempt in range(max_retries):
                try:
                    return func()
                except ZmqRpcTimeoutError:
                    print(f"Timeout error, retrying {attempt + 1}/{max_retries}...")
                    time.sleep(delay)
            raise ZmqRpcTimeoutError("Max retries reached")
                
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
        fits.append(file_name, np.array(self._zc_integrated_modal_command_list))

    @logEnterAndExit("Loading data...",
           "Data loaded.", level='debug')    
    @staticmethod
    def load_telemetry(fname):
        set_data_dir()
        file_name = telemetry_folder() / (fname + '.fits')
        header = fits.getheader(file_name)
        hduList = fits.open(file_name)

        slopes_vect = hduList[0].data
        zc_delta_modal_commands = hduList[1].data
        zc_integrated_modal_commands  = hduList[2].data
        
        return  header, slopes_vect, zc_delta_modal_commands, zc_integrated_modal_commands


    