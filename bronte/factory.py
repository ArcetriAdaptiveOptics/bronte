from pysilico import camera
from plico_dm import deformableMirror
from functools import cached_property

from bronte.utils.objects_io import load_object
from bronte import package_data
from bronte.wfs.rtc import ScaoRealTimeComputer
from bronte.wfs.slm_rasterizer import SlmRasterizer
from bronte.wfs.slope_computer import PCSlopeComputer
from bronte.wfs.subaperture_set import ShSubapertureSet

from arte.utils.modal_decomposer import ModalDecomposer
from arte.atmo.phase_screen_generator import PhaseScreenGenerator
from bronte.wfs.temporal_controller import PureIntegrator
from bronte.types.slm_pupil_mask_generator import SlmPupilMaskGenerator

class BronteFactory():
    SUBAPS_TAG = '240807_152700'  # '240802_122800'
    PHASE_SCREEN_TAG = '240806_124700'
    MODAL_DEC_TAG = '241105_170400' #None
    ELT_PUPIL_TAG = None #'EELT480pp0.0803m_obs0.283_spider2023'
    N_ZERNIKE_MODES_TO_CORRECT = 200
    N_MODES_TO_CORRECT = 200

    def __init__(self):
        self._set_up_basic_logging()
        
    def _set_up_basic_logging(self):
        import importlib
        import logging
        importlib.reload(logging)
        FORMAT = '%(asctime)s:%(levelname)s:%(name)s  %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    def _create_phase_screen_generator(self):
        self._r0 = 0.3
        psg = PhaseScreenGenerator.load_normalized_phase_screens(
            package_data.phase_screen_folder() / (self.PHASE_SCREEN_TAG+'.fits'))
        psg.rescale_to(self._r0)
        return psg
    
    def _create_slm_pupil_mask(self):
        spm = SlmPupilMaskGenerator()
        if self.ELT_PUPIL_TAG is not None:
            return spm.elt_pupil_mask(
                package_data.elt_pupil_folder()/(self.ELT_PUPIL_TAG + '.fits'))
        else:
            return spm.circular_pupil_mask()
    
    @cached_property
    def sh_camera(self):
        return camera('193.206.155.69', 7110)

    @cached_property
    def psf_camera(self):
        return camera('193.206.155.69', 7100)

    @cached_property
    def deformable_mirror(self):
        return deformableMirror('193.206.155.69', 7010)

    @cached_property
    def subapertures_set(self):
        return ShSubapertureSet.restore(
            package_data.subaperture_set_folder() / (self.SUBAPS_TAG+'.fits'))

    @cached_property
    def slope_computer(self):
        return PCSlopeComputer(self.subapertures_set)
    
    @cached_property
    def slm_pupil_mask(self):
        return self._create_slm_pupil_mask()
    
    @cached_property
    def slm_rasterizer(self):
        return SlmRasterizer(self.slm_pupil_mask)

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
