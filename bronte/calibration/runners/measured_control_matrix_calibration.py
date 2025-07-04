import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np
from specula.display.slopec_display import SlopecDisplay
from bronte.startup import  set_data_dir
from bronte.package_data import reconstructor_folder
from astropy.io import fits
from plico.rpc.zmq_remote_procedure_call import ZmqRpcTimeoutError
import time
from bronte.utils.set_basic_logging import get_logger
from arte.utils.decorator import logEnterAndExit 


class MeasuredControlMatrixCalibrator():
    
    CALIBRATION_TYPE = 'MEASURED'
    
    def __init__(self, calib_factory, ftag, pp_amp_in_nm = None, xp=np):
        
        self._logger = get_logger("MeasuredControlMatrixCalibrator")
        self._ftag = ftag
        self._factory = calib_factory
        self._Nmodes = self._factory.N_MODES_TO_CORRECT 
        if pp_amp_in_nm is not None:
            self._factory.PP_AMP_IN_NM = pp_amp_in_nm
            
        self._build_processing_objects()        
        self._set_inputs()
        self._define_groups()
    
    @logEnterAndExit("Creating ProcessingObjects...",
                  "ProcessingObjects created.", level='debug')
    def _build_processing_objects(self):
        
        self._prop = self._factory.disturb_propagation
        self._slopec = self._factory.slope_computer
        self._dm = self._factory.virtual_deformable_mirror
        self._bench_devices = self._factory.testbench_devices
        self._pp = self._factory.push_pull
        self._im_calibrator = self._factory.interaction_matrix_calibrator(self._ftag)
        self._empty_layer = self._factory.empty_layer
        self._empty_layer.generation_time = 0
    
    @logEnterAndExit("Setting ProcessingObjects inputs ...",
                      "ProcessingObjects inputs set.", level='debug')
    def _set_inputs(self):
        
        self._im_calibrator.inputs['in_slopes'].set(self._slopec.outputs['out_slopes'])
        self._im_calibrator.inputs['in_commands'].set(self._pp.output)
        self._bench_devices.inputs['ef'].set(self._prop.outputs['out_on_axis_source_ef'])
        self._slopec.inputs['in_pixels'].set(self._bench_devices.outputs['out_pixels'])
        self._dm.inputs['in_command'].set(self._pp.output)
        self._prop.inputs['atmo_layer_list'].set([self._empty_layer])
        self._prop.inputs['common_layer_list'].set([self._dm.outputs['out_layer']])
        
        self._slopes_disp = SlopecDisplay()
        self._slopes_disp.inputs['slopes'].set(self._slopec.outputs['out_slopes'])
        self._slopes_disp.inputs['subapdata'].set(self._factory.subapertures_set)
    
    def _define_groups(self):
        
        group1 = [self._pp]
        group2 = [self._dm]
        group3 = [self._prop]
        group4 = [self._bench_devices]
        group5 = [self._slopec]
        group6 = [self._im_calibrator]
        group7 = [self._slopes_disp]
        
        self._groups = [group1, group2, group3, group4, group5, group6, group7]
    
    @logEnterAndExit("Starting Calibration...",
          "Calibration Terminated.", level='debug')
    def run(self):
        
        self.time_step = self._factory.TIME_STEP_IN_SEC
        self._n_steps = 2 * self._Nmodes
        tf = (self._n_steps - 1) * self.time_step
        self.save_calib_config()
        
        for group in self._groups:
            for obj in group:
                obj.loop_dt = self.time_step * 1e9
                obj.setup(self.time_step * 1e9, self._n_steps)
        
        for step in range(self._n_steps):
            t = 0 + step * self.time_step
            self._logger.info(
                "\n+ Push/Pull @ time: %f/%f s\t steps: %d/%d" % (t, tf, step+1, self._n_steps))
            for group in self._groups:
                for obj in group:
                    obj.check_ready(t*1e9)
                    self._logger.info(f"Triggering {str(obj)}")
                    obj.trigger()
                    obj.post_trigger()
        
        for group in self._groups:
            for obj in group:
                obj.finalize()
        
    @logEnterAndExit("Saving data...",
          "Data saved.", level='debug')        
    def save_calib_config(self):
        
        def retry_on_timeout(func, max_retries = 5000, delay = 0.001):
            '''Retries a function call if ZmqRpcTimeoutError occurs.'''
            for attempt in range(max_retries):
                try:
                    return func()
                except ZmqRpcTimeoutError:
                    print(f"Timeout error, retrying {attempt + 1}/{max_retries}...")
                    time.sleep(delay)
            raise ZmqRpcTimeoutError("Max retries reached")
                
        #psf_camera_texp = retry_on_timeout(lambda: self._factory.psf_camera.exposureTime())
        #psf_camera_fps = retry_on_timeout(lambda: self._factory.psf_camera.getFrameRate())
        shwfs_texp = retry_on_timeout(lambda: self._factory.sh_camera.exposureTime())
        shwfs_fps = retry_on_timeout(lambda: self._factory.sh_camera.getFrameRate())
        
        fname = self._ftag + '_bronte_calib_config'
        file_name = reconstructor_folder() / (fname + '.fits')
        hdr = fits.Header()
        
        hdr['CALIB_TY'] = self.CALIBRATION_TYPE
        
        # GENERATED FILE TAG AND DEPENDENCY
        hdr['SUB_TAG'] = self._factory.SUBAPS_TAG
        hdr['REC_TAG'] = self._ftag 
        hdr['IM_TAG'] = self._ftag 
        hdr['SOFF_TAG'] = self._factory.SLOPE_OFFSET_TAG
        
        if self._factory.ELT_PUPIL_TAG is not None:
            hdr['ELT_TAG'] = self._factory.ELT_PUPIL_TAG
        else:
            hdr['ELT_TAG'] = 'NA'
 
        # PROJECTION PARAMETERS
        hdr['GS_POS'] = str(self._factory.SOURCE_COORD)
        hdr['GS_MAG'] = self._factory.SOURCE_MAG
        hdr['GS_WL'] = self._factory.SOURCE_WL_IN_NM
        
        # LOOP PARAMETERS
        hdr['TSTEP_S'] = self.time_step
        hdr['N_MODES'] = self._factory.N_MODES_TO_CORRECT
        hdr['MOD_BASE'] = self._factory.MODAL_BASE_TYPE
        
        #HARDWARE PARAMETERS
        hdr['SLM_RAD'] = self._factory.SLM_PUPIL_RADIUS # in pixels
        hdr['SLM_YX'] = str(self._factory.SLM_PUPIL_CENTER) # YX pixel coordinates
        hdr['SHPX_THR'] = np.max((self._factory.SH_PIX_THR, self._factory.PIX_THR_RATIO)) 
        hdr['PC_TEXP'] = 'NA' # in ms
        hdr['PC_FPS'] = 'NA'
        hdr['SH_TEXP'] = shwfs_texp # in ms
        hdr['SH_FPS'] = shwfs_fps
        
        fits.writeto(file_name, self._factory._pp_ampl_vect, hdr)

    @staticmethod
    def load_calib_config(ftag):
        set_data_dir()
        fname = ftag + '_bronte_calib_config'
        file_name = reconstructor_folder()/ (fname + '.fits')
        header = fits.getheader(file_name)
        hduList = fits.open(file_name)
        pp_in_nm = hduList[0].data
        
        return header, pp_in_nm
        