import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np
from specula.data_objects.source import Source
from specula.processing_objects.atmo_propagation import AtmoPropagation
from specula.processing_objects.atmo_evolution import AtmoEvolution
from specula.processing_objects.func_generator import FuncGenerator
from specula.processing_objects.int_control import IntControl
from specula.processing_objects.sh_slopec import ShSlopec
from specula.processing_objects.dm import DM
from specula.processing_objects.modalrec import Modalrec
from specula.data_objects.subap_data import SubapData
from specula.data_objects.recmat import Recmat
from specula.display.modes_display import ModesDisplay
from bronte.startup import startup
from bronte.types.testbench_device_manager import TestbenchDeviceManager
from bronte.package_data import subaperture_set_folder, reconstructor_folder,\
    phase_screen_folder, telemetry_folder
from astropy.io import fits


class SpeculaScaoLoop():
    
    def __init__(self, target_device_idx=-1, xp=np):
        
        self._target_device_idx = target_device_idx
        self._factory = startup()
        self._telescope_pupil_diameter = 40
        self._pupil_diameter_in_pixel  = 2 * self._factory.slm_pupil_mask.radius()
        self._pupil_pixel_pitch = round(self._telescope_pupil_diameter/self._pupil_diameter_in_pixel, 3)
        self._nModes = self._factory.N_MODES_TO_CORRECT 
        self._load_sources()
        self._setup_atmosphere()
        self._load_slope_computer()
        self._load_reconstructor()
        self._setup_control()
        self._setup_bench_devices()       
        self._set_inputs()
        self._define_groups()
        self._initialize_telemetry()
    
    def _load_sources(self):
        
        self._on_axis_source = Source(polar_coordinate=[0.0, 0.0], magnitude=8, wavelengthInNm=750,)
        self._lgs1_source = Source(polar_coordinate=[45.0, 0.0], height=90000, magnitude=5, wavelengthInNm=589)
    
    def _setup_atmosphere(self):
        
        self._seeing = FuncGenerator(constant=0.3,
                               target_device_idx=self._target_device_idx)
        self._wind_speed = FuncGenerator(constant=[25.5, 5.5],
                                   target_device_idx=self._target_device_idx) 
        self._wind_direction = FuncGenerator(constant=[0, 0],
                                       target_device_idx=self._target_device_idx)
    
        
        self._atmo = AtmoEvolution(pixel_pupil = self._pupil_diameter_in_pixel,              # Linear dimension of pupil phase array
                             pixel_pitch = self._pupil_pixel_pitch,         # Linear dimension of pupil phase array
                             data_dir = phase_screen_folder(),      # Data directory for phasescreens
                             L0=23,                        # [m] Outer scale
                             heights = [300.000,  20500.0], # [m] layer heights at 0 zenith angle
                             Cn2 = [1 - 0.119977, 0.119977], # Cn2 weights (total must be eq 1)
                            source_dict = {'on_axis_source': self._on_axis_source,
                                            'lgs1_source': self._lgs1_source,
                                            },
                            target_device_idx=self._target_device_idx,
                            )
    
        self._prop = AtmoPropagation(pixel_pupil = self._pupil_diameter_in_pixel,              # Linear dimension of pupil phase array
                               pixel_pitch = self._pupil_pixel_pitch,         # Linear dimension of pupil phase array
                               source_dict = {'on_axis_source': self._on_axis_source,
                                            'lgs1_source': self._lgs1_source,
                                            },
                               target_device_idx=self._target_device_idx)
    
    def _load_slope_computer(self):
        
        self._subapdata = SubapData.restore_from_bronte(
            subaperture_set_folder() / (self._factory.SUBAPS_TAG + ".fits"))
        self._slopec = ShSlopec(subapdata= self._subapdata)
        self._nslopes = self._subapdata.n_subaps * 2 
    
    def _load_reconstructor(self):
        
        recmat = Recmat.restore(reconstructor_folder() / "250127_155400_bronte_rec.fits")
        modal_offset= np.zeros(self._nModes)
       
        if self._factory.MODAL_OFFSET_TAG is not None:
            modal_offset = self._factory.modal_offset[:self._nModes]
            
        self._rec = Modalrec(self._nModes, recmat=recmat, modal_offset=modal_offset)
        
    def _setup_control(self):
        
        int_gains = np.ones(self._nModes)*-0.3
        #int_gains = np.zeros(nModes); int_gains[0:3]=-0.5  
        self._control = IntControl(delay=2, int_gain=int_gains)
        self._dm = DM(type_str='zernike',
                pixel_pitch = self._pupil_pixel_pitch,
                nmodes = self._nModes,
                npixels = self._pupil_diameter_in_pixel,                    # linear dimension of DM phase array
                obsratio= 0,                    # obstruction dimension ratio w.r.t. diameter
                height=  0)     # DM height [m]
        
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
        
        self._disp = ModesDisplay()
        self._disp.inputs['modes'].set(self._rec.out_modes)
    
    def _define_groups(self):
        
        group1 = [self._seeing, self._wind_speed, self._wind_direction]
        group2 = [self._atmo]
        group3 = [self._prop]
        group4 = [self._bench_devices]
        group5 = [self._slopec]
        group6 = [self._rec]
        group7 = [self._control, self._disp]
        group8 = [self._dm]

        self._groups = [group1, group2, group3, group4, group5, group6, group7, group8]
    
    def _initialize_telemetry(self):
        
        self._slopes_x_maps_list = []
        self._slopes_y_maps_list = []
        #self._short_exp_psf_list = []
        self._zc_delta_modal_command_list = []
        self._zc_integrated_modal_command_list = []
        
    def _update_telemetry(self):
        
        specula_slopes = self._groups[4][0].outputs['out_slopes']
        slopes_vector = specula_slopes.slopes
        slope_x = slopes_vector[specula_slopes.indicesX]
        slope_y = slopes_vector[specula_slopes.indicesY]
        
        self._slopes_x_maps_list.append(slope_x)
        self._slopes_y_maps_list.append(slope_y)
        
        # self._short_exp_psf_list.append(self._short_exp)
        
        specula_delta_commands_in_nm = self._groups[5][0].out_modes.value
        self._zc_delta_modal_command_list.append(
            specula_delta_commands_in_nm*1e-9)
        
        specula_integrated_commands_in_nm = self._groups[6][0].out_comm.value
        self._zc_integrated_modal_command_list.append(
            specula_integrated_commands_in_nm*1e-9)
        

    def run(self, Nsteps = 30):
        
        self._n_steps = Nsteps
        time_step = 0.01
        
        for group in self._groups:
            for obj in group:
                obj.loop_dt = time_step * 1e9
                obj.run_check(time_step)
    
        for step in range(self._n_steps):
            t = 0 + step * time_step
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
        pass
        # psf_camera_texp = self._factory.psf_camera.exposureTime()
        # psf_camera_fps = self._factory.psf_camera.getFrameRate()
        # shwfs_texp = self._factory.sh_camera.exposureTime()
        # shwfs_fps = self._factory.sh_camera.getFrameRate()
        #
        # file_name = telemetry_folder() / (fname + '.fits')
        # hdr = fits.Header()
        #
        #
        # #FILE TAG DEPENDENCY
        # hdr['SUB_TAG'] = self._factory.SUBAPS_TAG
        # hdr['PS_TAG'] = 'NA'
        # hdr['MD_TAG'] = 'NA'
        #
        # #ATMO PARAMETERS
        # if self._wavefront_disturb is None:
        #     self._wind_speed = 0
        #     self._factory._r0 = 0
        # hdr['R0_IN_M'] = self._factory._r0
        # hdr['WIND_SP'] = self._wind_speed # in phase screen/step
        #
        # #LOOP PARAMETERS
        # #hdr['AO_STAT'] = self._ao_status # 'open' or 'closed'
        # hdr['INT_TYPE'] = self._factory.pure_integrator_controller._integrator_type
        # hdr['INT_GAIN'] = self._factory.pure_integrator_controller._gain
        # hdr['N_STEPS'] = self._t
        #
        # #HARDWARE PARAMETERS
        #
        # hdr['PC_TEXP'] = psf_camera_texp # in ms
        # hdr['PC_FPS'] = psf_camera_fps
        # hdr['SH_TEXP'] = shwfs_texp # in ms
        # hdr['SH_FPS'] = shwfs_fps
        # hdr['SLM_RT'] = self.SLM_RESPONSE_TIME # in sec
        #
        # fits.writeto(file_name, self._long_exp, hdr)
        # fits.append(file_name, np.array(self._short_exp_psf_list))
        # fits.append(file_name, np.array(self._slopes_x_maps_list))
        # fits.append(file_name, np.array(self._slopes_y_maps_list))
        #
        # #CONTROL MATRICES
        # fits.append(file_name, self._factory.modal_decomposer._lastIM)
        # fits.append(file_name, self._factory.modal_decomposer._lastReconstructor)
        #
        # #MODAL COMMANDS
        # fits.append(file_name, np.array(self._zc_delta_modal_command_list))
        # fits.append(file_name, np.array(self._zc_integrated_modal_command_list))
        #
        # #OFFSETS
        # fits.append(file_name, self._factory.rtc.modal_offset.toNumpyArray())
        
    @staticmethod
    def load_telemetry(fname):
        pass
        # file_name = telemetry_folder() / (fname + '.fits')
        # header = fits.getheader(file_name)
        # hduList = fits.open(file_name)
        #
        # #FILE TAG DEPENDENCY
        # SUBAPS_TAG = header['SUB_TAG']
        # PHASE_SCREEN_TAG = header['PS_TAG']
        # MODAL_DEC_TAG = header['MD_TAG']
        # tag_list = [SUBAPS_TAG, PHASE_SCREEN_TAG, MODAL_DEC_TAG]
        #
        # #ATMO PARAMETERS
        # r0 = header['R0_IN_M']
        # wind_speed = header['WIND_SP'] # in phase screen/step
        # atmospheric_param_list = [r0, wind_speed]
        #
        # #LOOP PARAMETERS
        # #ao_status = header['AO_STAT']
        # integrator_type = header['INT_TYPE']
        # int_gain = header['INT_GAIN']
        # loop_steps = header['N_STEPS']
        # loop_param_list = [integrator_type, int_gain, loop_steps]
        #
        # #HARDWARE PARAMETERS
        # psf_camera_texp = header['PC_TEXP']
        # psf_camera_fps = header['PC_FPS']
        # shwfs_texp = header['SH_TEXP']
        # shwfs_fps = header['SH_FPS']
        # slm_response_time = header['SLM_RT']
        # hardware_param_list = [psf_camera_texp, psf_camera_fps,\
        #                         shwfs_texp, shwfs_fps, slm_response_time]
        #
        # #FRAMES
        # long_exp_psf = hduList[0].data
        # short_exp_psfs = hduList[1].data
        # slopes_x_maps = hduList[2].data
        # slopes_y_maps = hduList[3].data
        #
        # #CONTROL MATRICES
        # #TO DO: to fix cached prorerties are not picklable
        # # save the slope2modal_coeff matrix (rec_mat)
        # # and the modal_coeff2command/wf matrix (int_mat)
        # interaction_matrix = hduList[4].data
        # reconstructor = hduList[5].data
        #
        # #MODAL COMMANDS
        # zc_delta_modal_commands = hduList[6].data
        # zc_integrated_modal_commands  = hduList[7].data
        #
        # #OFFSETS
        # zc_modal_offset = hduList[8].data
        #
        # return tag_list,\
        #      atmospheric_param_list,\
        #       loop_param_list,\
        #        hardware_param_list,\
        #         long_exp_psf,\
        #          short_exp_psfs,\
        #           slopes_x_maps, slopes_y_maps,\
        #           interaction_matrix, reconstructor,\
        #             zc_delta_modal_commands, zc_integrated_modal_commands,\
        #                 zc_modal_offset
        #
