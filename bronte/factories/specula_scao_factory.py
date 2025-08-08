import specula

specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np
from specula.data_objects.source import Source
from specula.processing_objects.atmo_propagation import AtmoPropagation
from specula.processing_objects.atmo_evolution import AtmoEvolution
from specula.processing_objects.func_generator import FuncGenerator
from specula.processing_objects.integrator import Integrator
#from specula.processing_objects.int_control import IntControl
from specula.processing_objects.sh_slopec import ShSlopec
from specula.data_objects.slopes import Slopes
from specula.processing_objects.dm import DM
from specula.processing_objects.modalrec import Modalrec
from specula.data_objects.subap_data import SubapData
from specula.data_objects.recmat import Recmat

from functools import cached_property
from bronte.factories.base_factory import BaseFactory
from bronte import package_data
from bronte.wfs.slope_computer import PCSlopeComputer
from bronte.wfs.subaperture_set import ShSubapertureSet
from bronte.telemetry_trash.display_telemetry_data import DisplayTelemetryData
from bronte.utils.slopes_covariance_matrix_analyser import SlopesCovariaceMatrixAnalyser
from bronte.types.slm_pupil_mask_generator import SlmPupilMaskGenerator

class SpeculaScaoFactory(BaseFactory):
    
    #FILE TAGS
    SUBAPS_TAG = '250612_143100'#'250610_140500'#'250120_122000'
    MODAL_OFFSET_TAG = None 
    REC_MAT_TAG = '250307_140600'#'250207_120300' #'250127_155400'
    SLOPE_OFFSET_TAG = '250610_150900'
    LOAD_HUGE_TILT_UNDER_MASK  = False
    #ATMO PARAMETERS
    ATMO_SEED = 1
    TELESCOPE_PUPIL_DIAMETER = 40   # m
    OUTER_SCALE_L0 = 23             # m
    SEEING = 0.3                    # arcsec
    WIND_SPEED_LIST = [10.0, 5.5]#[25.5, 5.5]   #in m/s
    WIND_DIR_LIST = [0, 0]
    LAYER_HEIGHTS_LIST = [300.000,  20500.0] # in m
    Cn2_WEIGHTS_LIST = [1 - 0.119977, 0.119977] # sum must be=1
    
    #SOURCE PARAMETERS
    FOV = 90. # diameter in arcsec
    ONAXIS_SOURCE_COORD = [0.0, 0.0] #[radius(in_arcsec), angle(in_deg)]
    ONAXIS_SOURCE_MAG = 8
    ONAXIS_SOURCE_WL_IN_NM = 750 
    LGS_COORD = [45.0, 0.0]
    LGS_HEIGHT_IN_M = 90000
    LGS_MAG = 5
    LGS_WL_IN_NM = 589
    
    #AO PARAMETERS
    MODAL_BASE_TYPE = 'zernike'
    KL_MODAL_IFS_TAG  = None
    #N_MODES_TO_CORRECT = 200 # moved to base factory
    INT_GAIN = -0.3
    INT_DELAY = 2                   # frames or ms
    SH_PIX_THR = 0#200                # threshold in ADU for pixels in subapertures
    PIX_THR_RATIO = 0.2
    TIME_STEP_IN_SEC = 0.01          # time step of the simulated loop in sec
    
    def __init__(self):
        super().__init__()
        self._pupil_diameter_in_pixel  = 2 * self.slm_pupil_mask.radius()
        self._pupil_pixel_pitch = self.TELESCOPE_PUPIL_DIAMETER/self._pupil_diameter_in_pixel
    
    @cached_property
    def subapertures_set(self):
        subapdata = SubapData.restore_from_bronte(
            package_data.subaperture_set_folder() / (self.SUBAPS_TAG + ".fits"))
        return subapdata
    
    @cached_property
    def subaperture_set_grid_map(self):
        subap_set = ShSubapertureSet.restore(
            package_data.subaperture_set_folder() / (self.SUBAPS_TAG + '.fits'))
        sc = PCSlopeComputer(subap_set)
        return sc.subapertures_map()
    
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
    def reconstructor(self):
        recmat = Recmat.restore(package_data.reconstructor_folder() / (self.REC_MAT_TAG + "_bronte_rec.fits"))
        modal_offset = np.zeros(self.N_MODES_TO_CORRECT)
        #added factor 2 missed on IFs normalization
        N_pp = 2
        recmat.recmat = N_pp*recmat.recmat  
        if self.MODAL_OFFSET_TAG is not None:
            modal_offset = self.modal_offset[:self.N_MODES_TO_CORRECT]
            
        return Modalrec(self.N_MODES_TO_CORRECT, recmat=recmat, modal_offset=modal_offset)
    
    @cached_property
    def integrator_controller(self):
        int_gains = np.ones(self.N_MODES_TO_CORRECT)* self.INT_GAIN
        #int_gains = np.zeros(nModes); int_gains[0:3]=-0.5  
        #return IntControl(delay = self.INT_DELAY, int_gain = int_gains)
        return Integrator(delay = self.INT_DELAY, int_gain = int_gains)
    
    @cached_property
    def virtual_deformable_mirror(self):
        
        if self.MODAL_BASE_TYPE == 'zernike':
            virtual_dm = DM(type_str=self.MODAL_BASE_TYPE,
                    pixel_pitch = self._pupil_pixel_pitch,
                    nmodes = self.N_MODES_TO_CORRECT,
                    npixels = self._pupil_diameter_in_pixel,                    # linear dimension of DM phase array
                    obsratio = 0,                    # obstruction dimension ratio w.r.t. diameter
                    height =  0)     # DM height [m]
        
        if self.MODAL_BASE_TYPE == 'kl':
            
            kl_modal_ifs = self.load_kl_modal_ifs()
            ifs_mask = kl_modal_ifs.mask_inf_func
            self._update_slm_pupil_mask_with_ifs_mask(ifs_mask)
            virtual_dm = DM(
                pixel_pitch= self._pupil_pixel_pitch,
                height = 0,
                ifunc = kl_modal_ifs)
        
        return virtual_dm
    
    def _update_slm_pupil_mask_with_ifs_mask(self, ifs_mask_idl):
        
        spg = SlmPupilMaskGenerator(
            pupil_radius = self.SLM_PUPIL_RADIUS, 
            pupil_center= self.SLM_PUPIL_CENTER)
        
        mask = (1 - ifs_mask_idl).astype(bool)
        rescaled_mask = spg._get_rescaled_mask_to_slm_frame(mask)
        self.slm_rasterizer.slm_pupil_mask._mask = rescaled_mask
     
    
    @cached_property
    def modal_offset(self):
        if self.MODAL_OFFSET_TAG is None:
            return None
        modal_offset,_ = DisplayTelemetryData.load_modal_offset(self.MODAL_OFFSET_TAG)
        return modal_offset 
    
    @cached_property
    def slope_offset(self):
        if self.SLOPE_OFFSET_TAG is None:
            return None
        slope_offset, _, _ = SlopesCovariaceMatrixAnalyser.load_slope_offset(self.SLOPE_OFFSET_TAG)
        return slope_offset
    
    @cached_property
    def source_dict(self):
        
        on_axis_source = Source(
            polar_coordinates = self.ONAXIS_SOURCE_COORD,
            magnitude = self.ONAXIS_SOURCE_MAG,
            wavelengthInNm = self.ONAXIS_SOURCE_WL_IN_NM,)
        
        lgs1_source = Source(
            polar_coordinates = self.LGS_COORD,
            height = self.LGS_HEIGHT_IN_M,
            magnitude = self.LGS_MAG,
            wavelengthInNm = self.LGS_WL_IN_NM)
        
        source_dict = {'on_axis_source': on_axis_source,
                       'lgs1_source': lgs1_source}
        
        return source_dict
    
    @cached_property
    def seeing(self):
        seeing = FuncGenerator(constant = self.SEEING,
                               target_device_idx = self._target_device_idx)
        return seeing
    
    @cached_property
    def wind_speed(self):
        
        wind_speed = FuncGenerator(constant = self.WIND_SPEED_LIST,
                                   target_device_idx=self._target_device_idx)
        return wind_speed
    
    @cached_property
    def wind_direction(self):
    
        wind_direction = FuncGenerator(constant = self.WIND_DIR_LIST,
                                       target_device_idx=self._target_device_idx)
        return wind_direction
    
    @cached_property
    def atmo_evolution(self):

        atmo = AtmoEvolution(seed = self.ATMO_SEED,
                             pixel_pupil = self._pupil_diameter_in_pixel,              # Linear dimension of pupil phase array
                             pixel_pitch = self._pupil_pixel_pitch,         # Linear dimension of pupil phase array
                             data_dir = package_data.phase_screen_folder(),      # Data directory for phasescreens
                             L0 = self.OUTER_SCALE_L0,                        # [m] Outer scale
                             heights = self.LAYER_HEIGHTS_LIST, # [m] layer heights at 0 zenith angle
                             Cn2 = self.Cn2_WEIGHTS_LIST, # Cn2 weights (total must be eq 1)
                             fov = self.FOV,
                             target_device_idx=self._target_device_idx,
                            )
        return atmo
    
    @cached_property
    def atmo_propagation(self):
        prop = AtmoPropagation(pixel_pupil = self._pupil_diameter_in_pixel,              # Linear dimension of pupil phase array
                               pixel_pitch = self._pupil_pixel_pitch,         # Linear dimension of pupil phase array
                               source_dict = self.source_dict,
                               target_device_idx=self._target_device_idx)
        return prop

   
        