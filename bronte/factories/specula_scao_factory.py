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


from functools import cached_property
from bronte.factories.base_factory import BaseFactory
from bronte import package_data
from bronte.telemetry.display_telemetry_data import DisplayTelemetryData


class SpeculaScaoFactory(BaseFactory):
    
    #FILE TAGS
    SUBAPS_TAG = '250120_122000'
    MODAL_OFFSET_TAG = None #'250203_134800'
    REC_MAT_TAG = '250307_140600'#'250207_120300' #'250127_155400'
    
    #AO PARAMETERS
    N_MODES_TO_CORRECT = 200 
    TELESCOPE_PUPIL_DIAMETER = 40   # m
    OUTER_SCALE_L0 = 23             # m
    SEEING = 0.3                    # arcsec
    INT_GAIN = -0.3
    INT_DELAY = 2                   # frames or ms
    SH_PIX_THR = 200                # threshold in ADU for pixels in subapertures
    
    def __init__(self):
        super().__init__()
        self._pupil_diameter_in_pixel  = 2 * self.slm_pupil_mask.radius()
        self._pupil_pixel_pitch = round(self.TELESCOPE_PUPIL_DIAMETER/self._pupil_diameter_in_pixel, 3)

    @cached_property
    def subapertures_set(self):
        subapdata = SubapData.restore_from_bronte(
            package_data.subaperture_set_folder() / (self.SUBAPS_TAG + ".fits"))
        return subapdata
    
    @cached_property
    def slope_computer(self):
        return ShSlopec(subapdata= self.subapertures_set, thr_value =  self.SH_PIX_THR)
    
    @cached_property
    def reconstructor(self):
        recmat = Recmat.restore(package_data.reconstructor_folder() / (self.REC_MAT_TAG + "_bronte_rec.fits"))
        modal_offset= np.zeros(self.N_MODES_TO_CORRECT)
       
        if self.MODAL_OFFSET_TAG is not None:
            modal_offset = self.modal_offset[:self.N_MODES_TO_CORRECT]
            
        return Modalrec(self.N_MODES_TO_CORRECT, recmat=recmat, modal_offset=modal_offset)
    
    @cached_property
    def integrator_controller(self):
        int_gains = np.ones(self.N_MODES_TO_CORRECT)* self.INT_GAIN
        #int_gains = np.zeros(nModes); int_gains[0:3]=-0.5  
        return IntControl(delay = self.INT_DELAY, int_gain = int_gains)
    
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
    def modal_offset(self):
        if self.MODAL_OFFSET_TAG is None:
            return None
        modal_offset,_ = DisplayTelemetryData.load_modal_offset(self.MODAL_OFFSET_TAG)
        return modal_offset 
    
    @cached_property
    def source_dict(self):
        
        on_axis_source = Source(polar_coordinate=[0.0, 0.0], magnitude=8, wavelengthInNm=750,)
        lgs1_source = Source(polar_coordinate=[45.0, 0.0], height=90000, magnitude=5, wavelengthInNm=589)
        
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
        wind_speed = FuncGenerator(constant=[25.5, 5.5],
                                   target_device_idx=self._target_device_idx)
        return wind_speed
    
    @cached_property
    def wind_direction(self):
        wind_direction = FuncGenerator(constant=[0, 0],
                                       target_device_idx=self._target_device_idx)
        return wind_direction
    
    @cached_property
    def atmo_evolution(self):
        
        layer_heights_list = [300.000,  20500.0]
        Cn2_weights_list = [1 - 0.119977, 0.119977]
        
        atmo = AtmoEvolution(pixel_pupil = self._pupil_diameter_in_pixel,              # Linear dimension of pupil phase array
                             pixel_pitch = self._pupil_pixel_pitch,         # Linear dimension of pupil phase array
                             data_dir = package_data.phase_screen_folder(),      # Data directory for phasescreens
                             L0 = self.OUTER_SCALE_L0,                        # [m] Outer scale
                             heights = layer_heights_list, # [m] layer heights at 0 zenith angle
                             Cn2 = Cn2_weights_list, # Cn2 weights (total must be eq 1)
                             source_dict = self.source_dict,
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

   
        