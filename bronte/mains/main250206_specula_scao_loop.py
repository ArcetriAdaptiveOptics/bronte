import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np

from bronte.startup import specula_startup
from bronte.types.testbench_device_manager import TestbenchDeviceManager
from specula.display.modes_display import ModesDisplay
from specula.display.slopec_display import SlopecDisplay
from bronte.package_data import telemetry_folder
from astropy.io import fits


class SpeculaScaoLoop():
    
    def __init__(self, xp=np):
        
        self._factory = specula_startup()
        self._target_device_idx = self._factory._target_device_idx
        self._setup_atmosphere()
        self._load_slope_computer()
        self._load_reconstructor()
        self._setup_control()
        self._setup_bench_devices()       
        self._set_inputs()
        self._define_groups()
        self._initialize_telemetry()

    def _setup_atmosphere(self):
        
        self._seeing = self._factory.seeing
        self._wind_speed = self._factory.wind_speed 
        self._wind_direction = self._factory.wind_direction
        self._atmo = self._factory.atmo_evolution
        self._prop = self._factory.atmo_propagation
    
    def _load_slope_computer(self):
        self._subapdata = self._factory.subapertures_set
        self._slopec = self._factory.slope_computer
        self._nslopes = self._subapdata.n_subaps * 2 
    
    def _load_reconstructor(self):

        self._rec = self._factory.reconstructor
        
    def _setup_control(self):
        
        self._control = self._factory.integrator_controller
        self._dm = self._factory.virtual_deformable_mirror
        
    def _setup_bench_devices(self):
        
        self._factory.sh_camera.setExposureTime(8)
        self._bench_devices = TestbenchDeviceManager(self._factory, 
                                do_plots=True,
                                target_device_idx= self._target_device_idx)        
        
    def _set_inputs(self):
        
        self._atmo.inputs['seeing'].set(self._seeing.output)
        self._atmo.inputs['wind_direction'].set(self._wind_direction.output)
        self._atmo.inputs['wind_speed'].set(self._wind_speed.output)
        self._prop.inputs['layer_list'].set(self._atmo.layer_list + [self._dm.outputs['out_layer']])
    
        self._bench_devices.inputs['ef'].set(self._prop.outputs['out_on_axis_source_ef'])
        self._slopec.inputs['in_pixels'].set(self._bench_devices.outputs['out_pixels'])
        self._rec.inputs['in_slopes'].set(self._slopec.outputs['out_slopes'])
        self._control.inputs['delta_comm'].set(self._rec.out_modes)
        self._dm.inputs['in_command'].set(self._control.out_comm)
        
        self._modes_disp = ModesDisplay()
        self._modes_disp.inputs['modes'].set(self._rec.out_modes)
        self._slopes_disp = SlopecDisplay()
        self._slopes_disp.inputs['slopes'].set(self._slopec.outputs['out_slopes'])
        self._slopes_disp.inputs['subapdata'].set(self._factory.subapertures_set)
    
    def _define_groups(self):
        
        group1 = [self._seeing, self._wind_speed, self._wind_direction]
        group2 = [self._atmo]
        group3 = [self._prop]
        group4 = [self._bench_devices]
        group5 = [self._slopec]
        group6 = [self._rec]
        group7 = [self._control, self._modes_disp, self._slopes_disp]
        group8 = [self._dm]

        self._groups = [group1, group2, group3, group4, group5, group6, group7, group8]
    
    def _initialize_telemetry(self):
        
        #self._short_exp_psf_list = []
        self._slopes_vector_list = []
        self._zc_delta_modal_command_list = []
        self._zc_integrated_modal_command_list = []
        
    def _update_telemetry(self):
        
        specula_slopes = self._groups[4][0].outputs['out_slopes']
        self._slopes_vector_list.append(specula_slopes.slopes.copy())        
        # self._short_exp_psf_list.append(self._short_exp)
        specula_delta_commands_in_nm = self._groups[5][0].out_modes.value
        self._zc_delta_modal_command_list.append(
            specula_delta_commands_in_nm*1e-9)
        
        specula_integrated_commands_in_nm = self._groups[6][0].out_comm.value
        self._zc_integrated_modal_command_list.append(
            specula_integrated_commands_in_nm*1e-9)
        

    def run(self, Nsteps = 30):
        
        self._n_steps = Nsteps
        # time step of the simulated loop isnt it QM
        self.time_step = self._factory.TIME_STEP_IN_SEC
        
        for group in self._groups:
            for obj in group:
                obj.loop_dt = self.time_step * 1e9
                obj.run_check(self.time_step)
    
        for step in range(self._n_steps):
            t = 0 + step * self.time_step
            print('T=',t)
            for group in self._groups:
                for obj in group:
                    obj.check_ready(t*1e9)
                    print('trigger', obj)
                    obj.trigger()
                    obj.post_trigger()
            self._update_telemetry()
            
        for group in self._groups:
            for obj in group:
                obj.finalize()
    
    def save_telemetry(self, fname):
    
        psf_camera_texp = self._factory.psf_camera.exposureTime()
        psf_camera_fps = self._factory.psf_camera.getFrameRate()
        shwfs_texp = self._factory.sh_camera.exposureTime()
        shwfs_fps = self._factory.sh_camera.getFrameRate()
        
        file_name = telemetry_folder() / (fname + '.fits')
        hdr = fits.Header()

        # #FILE TAG DEPENDENCY
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
        
        # ATMO PARAMETERS
        hdr['SEEING'] = self._factory.SEEING
        hdr['L0_IN_M'] = self._factory.OUTER_SCALE_L0
        hdr['D_IN_M'] = self._factory.TELESCOPE_PUPIL_DIAMETER
        hdr['WIND_SP'] = str(self._factory.WIND_SPEED_LIST)
        hdr['WIND_DIR'] = str(self._factory.WIND_DIR_LIST)
        hdr['CN2_W'] = str(self._factory.Cn2_WEIGHTS_LIST)
        hdr['NGS_POS'] = str(self._factory.ONAXIS_SOURCE_COORD)
        hdr['NGS_MAG'] = self._factory.ONAXIS_SOURCE_MAG
        hdr['NGS_WL'] = self._factory.ONAXIS_SOURCE_WL_IN_NM
        hdr['LGS_POS'] = str(self._factory.LGS_COORD)
        hdr['LGS_MAG'] = self._factory.LGS_MAG
        hdr['LGS_ALT'] = self._factory.LGS_HEIGHT_IN_M
        hdr['LGS_WL'] = self._factory.LGS_WL_IN_NM
        
        # LOOP PARAMETERS
        hdr['TSTEP_S'] = self.time_step
        hdr['INT_GAIN'] = self._factory.INT_GAIN
        hdr['INT_DEL'] = self._factory.INT_DELAY
        hdr['N_STEPS'] = self._n_steps
        hdr['N_MODES'] = self._factory.N_MODES_TO_CORRECT
        hdr['SHPX_THR'] = self._factory.SH_PIX_THR
        hdr['PC_TEXP'] = psf_camera_texp # in ms
        hdr['PC_FPS'] = psf_camera_fps
        hdr['SH_TEXP'] = shwfs_texp # in ms
        hdr['SH_FPS'] = shwfs_fps
        
        fits.writeto(file_name, np.array(self._slopes_vector_list), hdr)
        fits.append(file_name, np.array(self._zc_delta_modal_command_list))
        fits.append(file_name, np.array(self._zc_integrated_modal_command_list))

        
    @staticmethod
    def load_telemetry(fname):

        file_name = telemetry_folder() / (fname + '.fits')
        header = fits.getheader(file_name)
        hduList = fits.open(file_name)

        
        slopes_vect = hduList[0].data
        #MODAL COMMANDS
        zc_delta_modal_commands = hduList[1].data
        zc_integrated_modal_commands  = hduList[2].data
        

        return  header, slopes_vect, zc_delta_modal_commands, zc_integrated_modal_commands

