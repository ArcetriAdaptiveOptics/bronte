from functools import cached_property
from bronte.utils.objects_io import load_object
from bronte import package_data
from bronte.wfs.rtc import ScaoRealTimeComputer
from bronte.wfs.slope_computer import PCSlopeComputer
from bronte.wfs.subaperture_set import ShSubapertureSet
from arte.utils.modal_decomposer import ModalDecomposer
from arte.atmo.phase_screen_generator import PhaseScreenGenerator
from bronte.wfs.temporal_controller import PureIntegrator
from bronte.telemetry.display_telemetry_data import DisplayTelemetryData
from bronte.factories.base_factory import BaseFactory

class BronteFactory(BaseFactory):
    
    SUBAPS_TAG = '250120_122000' 
    PHASE_SCREEN_TAG = '240806_124700'
    MODAL_DEC_TAG = None 
    #N_ZERNIKE_MODES_TO_CORRECT = 200
    N_MODES_TO_CORRECT = 10
    MODAL_OFFSET_TAG = None #'250203_134800'
    SH_PIX_THR = 200

    def __init__(self):
        super().__init__()

    def _create_phase_screen_generator(self):
        self._r0 = 0.3
        psg = PhaseScreenGenerator.load_normalized_phase_screens(
            package_data.phase_screen_folder() / (self.PHASE_SCREEN_TAG+'.fits'))
        psg.rescale_to(self._r0)
        return psg
    
    @cached_property
    def subapertures_set(self):
        return ShSubapertureSet.restore(
            package_data.subaperture_set_folder() / (self.SUBAPS_TAG+'.fits'))

    @cached_property
    def slope_computer(self):
        return PCSlopeComputer(self.subapertures_set)
    
    @cached_property
    def modal_offset(self):
        if self.MODAL_OFFSET_TAG is None:
            return None
        modal_offset,_ = DisplayTelemetryData.load_modal_offset(self.MODAL_OFFSET_TAG)
        return modal_offset 
        
    @cached_property
    def modal_decomposer(self):
        
        if self.MODAL_DEC_TAG is None:
            return ModalDecomposer(self.N_MODES_TO_CORRECT)
        
        md_fname = package_data.modal_decomposer_folder() / (self.MODAL_DEC_TAG + '.pkl')
        return load_object(md_fname)

    @cached_property
    def pure_integrator_controller(self):
        return PureIntegrator()

    @cached_property
    def rtc(self):
        return ScaoRealTimeComputer(self.sh_camera, self.slope_computer, self.deformable_mirror, self.modal_decomposer, self.pure_integrator_controller, self.slm_rasterizer)

    @cached_property
    def phase_screen_generator(self):
        return self._create_phase_screen_generator()
