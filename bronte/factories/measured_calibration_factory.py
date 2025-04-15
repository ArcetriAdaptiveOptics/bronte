import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np
from bronte.factories.base_factory import BaseFactory
from specula.processing_objects.im_rec_calibrator import ImRecCalibrator
from specula.processing_objects.func_generator import FuncGenerator
from specula.data_objects.source import Source
from specula.processing_objects.atmo_propagation import AtmoPropagation
from specula.data_objects.layer import Layer
from specula.data_objects.subap_data import SubapData
from specula.processing_objects.sh_slopec import ShSlopec
from specula.processing_objects.dm import DM
from bronte.package_data import subaperture_set_folder, reconstructor_folder
from functools import cached_property
from bronte.utils.noll_to_radial_order import from_noll_to_radial_order
from bronte.types.testbench_device_manager import TestbenchDeviceManager

class MeasuredCalibrationFactory(BaseFactory):
    
    SUBAPS_TAG = '250120_122000'
    MODAL_BASE_TYPE = 'Zernike'
    #N_MODES_TO_CORRECT = 200
    TELESCOPE_PUPIL_DIAMETER = 40   # m
    SH_PIX_THR = 200 # in ADU
    PP_AMP_IN_NM = 2000
    TIME_STEP_IN_SEC = None 
    SOURCE_COORD = [0.0, 0.0] # [radius(in_arcsec), angle(in_deg)]
    FOV = 2*SOURCE_COORD[0] # diameter in arcsec
    SOURCE_MAG = 8
    SOURCE_WL_IN_NM = 750 
    
    def __init__(self):
        
        super().__init__()
        self._pupil_diameter_in_pixel  = 2 * self.slm_pupil_mask.radius()
        self._pupil_pixel_pitch = self.TELESCOPE_PUPIL_DIAMETER/self._pupil_diameter_in_pixel
        self._load_sh_camera_master_bkg()
        self.TIME_STEP_IN_SEC = self._sh_texp * 1e-3
        
    @cached_property
    def source_dict(self):
        on_axis_source = Source(
            polar_coordinates = self.SOURCE_COORD,
            magnitude = self.SOURCE_MAG,
            wavelengthInNm = self.SOURCE_WL_IN_NM,)
        source_dict = {'on_axis_source': on_axis_source}
        return source_dict
    
    @cached_property
    def disturb_propagation(self):
        prop = AtmoPropagation(pixel_pupil = self._pupil_diameter_in_pixel,              # Linear dimension of pupil phase array
                               pixel_pitch = self._pupil_pixel_pitch,
                               source_dict = self.source_dict,         # Linear dimension of pupil phase array
                               target_device_idx=self._target_device_idx)
        return prop

    @cached_property
    def subapertures_set(self):
        subapdata = SubapData.restore_from_bronte(
            subaperture_set_folder() / (self.SUBAPS_TAG + ".fits"))
        return subapdata
    
    @cached_property
    def slope_computer(self):
        return ShSlopec(subapdata= self.subapertures_set, thr_value =  self.SH_PIX_THR)
    
    @cached_property
    def virtual_deformable_mirror(self):
        virtual_dm = DM(type_str='zernike',
                pixel_pitch = self._pupil_pixel_pitch,
                nmodes = self.N_MODES_TO_CORRECT,
                npixels = self._pupil_diameter_in_pixel,                    # linear dimension of DM phase array
                obsratio = 0,                    # obstruction dimension ratio w.r.t. diameter
                height =  0)     # DM height [m]
        return virtual_dm
    
    @cached_property
    def testbench_devices(self):
        #self.sh_camera.setExposureTime(self._sh_texp)
        tbd = TestbenchDeviceManager(
            factory = self,
            target_device_idx = self._target_device_idx)
        return tbd
    
    @cached_property
    def push_pull(self):
        
        j_noll_vector = np.arange(self.N_MODES_TO_CORRECT) + 2
        radial_order = from_noll_to_radial_order(j_noll_vector)
        self._pp_ampl_vect = self.PP_AMP_IN_NM /(radial_order) # in nm
        #ampl_vect = np.ones(self.N_MODES_TO_CORRECT)*500
        pp = FuncGenerator(func_type = 'PUSHPULL',
                   nmodes = self.N_MODES_TO_CORRECT,
                   vect_amplitude = self._pp_ampl_vect,#in nm
                   target_device_idx = self._target_device_idx)
        return pp
    
    def interaction_matrix_calibrator(self, ftag):
        im_calibrator = ImRecCalibrator(
                            data_dir = reconstructor_folder(),
                            nmodes=self.N_MODES_TO_CORRECT,
                            rec_tag= ftag + '_bronte_rec',
                            im_tag= ftag + '_bronte_im',
                            target_device_idx=self._target_device_idx)
        return im_calibrator
    
    @cached_property
    def empty_layer(self):
        empty_layer = Layer(
            self._pupil_diameter_in_pixel,
            self._pupil_diameter_in_pixel,
            self._pupil_pixel_pitch,
            height = 0)
        return empty_layer
    
    @cached_property
    def modal_offset(self):
        return None