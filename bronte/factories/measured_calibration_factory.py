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
from specula.data_objects.slopes import Slopes
from bronte.package_data import subaperture_set_folder, reconstructor_folder
from functools import cached_property
from bronte.utils.noll_to_radial_order import from_noll_to_radial_order
from bronte.types.testbench_device_manager import TestbenchDeviceManager
from bronte.utils.slopes_covariance_matrix_analyser import SlopesCovariaceMatrixAnalyser

class MeasuredCalibrationFactory(BaseFactory):
    
    SUBAPS_TAG = '250120_122000'
    SLOPE_OFFSET_TAG = None
    LOAD_HUGE_TILT_UNDER_MASK  = False
    MODAL_BASE_TYPE = 'zernike'#'zernike'
    PP_VCT_LOADED = False
    #N_MODES_TO_CORRECT = 200 in BaseFactory
    #TELESCOPE_PUPIL_DIAMETER = 568*2*9.2e-6   # m
    SH_PIX_THR = 0#200 # in ADU
    PIX_THR_RATIO = 0.2
    PP_AMP_IN_NM = 2000
    TIME_STEP_IN_SEC = None 
    SOURCE_COORD = [0.0, 0.0] # [radius(in_arcsec), angle(in_deg)]
    FOV = 2*SOURCE_COORD[0] # diameter in arcsec
    SOURCE_MAG = 8
    SOURCE_WL_IN_NM = 750 
    
    def __init__(self):
        
        super().__init__()
        self._pupil_diameter_in_pixel  = 2 * self.slm_pupil_mask.radius()
        
        self._pupil_pixel_pitch = 9.2e-6 #self.TELESCOPE_PUPIL_DIAMETER/self._pupil_diameter_in_pixel
        self._load_sh_camera_master_bkg()
        self.TIME_STEP_IN_SEC = self._sh_texp * 1e-3
        self.SH_FRAMES2AVERAGE = 10
        
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
        if self.SLOPE_OFFSET_TAG is not None:
            sn = Slopes(len(self.slope_offset), self.slope_offset)
        else:
            sn = None
        slopec =  ShSlopec(subapdata= self.subapertures_set, thr_value =  self.SH_PIX_THR, sn = sn)
        slopec.thr_ratio_value = self.PIX_THR_RATIO
        return slopec
    
    @cached_property
    def virtual_deformable_mirror(self):
        virtual_dm = DM(type_str=self.MODAL_BASE_TYPE,
                pixel_pitch = self._pupil_pixel_pitch,
                nmodes = self.N_MODES_TO_CORRECT,
                npixels = self._pupil_diameter_in_pixel,                    # linear dimension of DM phase array
                obsratio = 0,                    # obstruction dimension ratio w.r.t. diameter
                height =  0)     # DM height [m]
        return virtual_dm
    
    @cached_property
    def slope_offset(self):
        if self.SLOPE_OFFSET_TAG is None:
            return None
        slope_offset, _, _ = SlopesCovariaceMatrixAnalyser.load_slope_offset(self.SLOPE_OFFSET_TAG)
        return slope_offset
    
    @cached_property
    def testbench_devices(self):
        #self.sh_camera.setExposureTime(self._sh_texp)
        tbd = TestbenchDeviceManager(
            factory = self,
            load_huge_tilt_under_mask = self.LOAD_HUGE_TILT_UNDER_MASK,
            target_device_idx = self._target_device_idx)
        return tbd
    
    @cached_property
    def push_pull(self):
        
        if self.PP_VCT_LOADED is not True:
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
    
    def load_custom_pp_amp_vector(self, pp_vector_in_nm):
        self.PP_VCT_LOADED = True
        self._pp_ampl_vect = pp_vector_in_nm