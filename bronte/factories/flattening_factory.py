import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np
from specula.processing_objects.integrator import Integrator
from specula.processing_objects.sh_slopec import ShSlopec
from specula.processing_objects.dm import DM
from specula.processing_objects.modalrec import Modalrec
from specula.data_objects.subap_data import SubapData
from specula.data_objects.recmat import Recmat


from functools import cached_property
from bronte.factories.base_factory import BaseFactory
from bronte import package_data
from bronte.telemetry_trash.display_telemetry_data import DisplayTelemetryData

class SpeculaFlatteningFactory(BaseFactory):
    
    #FILE TAGS
    SUBAPS_TAG = '250120_122000'
    MODAL_OFFSET_TAG = None 
    REC_MAT_TAG = '250307_140600'
    
    #AO PARAMETERS
    MODAL_BASE_TYPE = 'Zernike'
    #N_MODES_TO_CORRECT = 200 
    INT_GAIN = -0.3
    INT_DELAY = 2                   # frames or ms
    SH_ABS_PIX_THR = 0               # threshold in ADU for pixels in subapertures
    SH_THR_RATIO = 0.17                # threshold ratio per subap
    TIME_STEP_IN_SEC = 0.008          # time step of the simulated loop in sec
    
    def __init__(self):
        super().__init__()
        self._pupil_diameter_in_pixel  = 2 * self.slm_pupil_mask.radius()
        self._pupil_pixel_pitch = 9.2e-6

    @cached_property
    def subapertures_set(self):
        subapdata = SubapData.restore_from_bronte(
            package_data.subaperture_set_folder() / (self.SUBAPS_TAG + ".fits"))
        return subapdata
    
    @cached_property
    def slope_computer(self):
        slopec = ShSlopec(subapdata= self.subapertures_set, thr_value =  self.SH_ABS_PIX_THR)
        slopec.thr_ratio_value = self.SH_THR_RATIO
        return slopec 
    
    @cached_property
    def reconstructor(self):
        recmat = Recmat.restore(package_data.reconstructor_folder() / (self.REC_MAT_TAG + "_bronte_rec.fits"))
        modal_offset= np.zeros(self.N_MODES_TO_CORRECT)
       
        if self.MODAL_OFFSET_TAG is not None:
            modal_offset = self.modal_offset[:self.N_MODES_TO_CORRECT]
        # added factor 2 missed on IFs normalization
        N_pp = 2
        recmat.recmat = N_pp*recmat.recmat    
        return Modalrec(self.N_MODES_TO_CORRECT, recmat=recmat, modal_offset=modal_offset)
    
    @cached_property
    def integrator_controller(self):
        int_gains = np.ones(self.N_MODES_TO_CORRECT)* self.INT_GAIN
        #int_gains = np.zeros(nModes); int_gains[0:3]=-0.5  
        return Integrator(delay = self.INT_DELAY, int_gain = int_gains)
    
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
    
