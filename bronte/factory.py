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


class BronteFactory():
    SUBAPS_TAG = '240807_152700'  # '240802_122800'
    PHASE_SCREEN_TAG = '240806_124700'
    MODAL_DEC_TAG = '241105_170400' #None
    N_ZERNIKE_MODES_TO_CORRECT = 200

    def __init__(self):
        self._set_up_basic_logging()
        self._create_phase_screen_generator()
        self._subaps = ShSubapertureSet.restore(
            package_data.subaperture_set_folder() / (self.SUBAPS_TAG+'.fits'))
        self._sc = PCSlopeComputer(self._subaps)

    def _set_up_basic_logging(self):
        import importlib
        import logging
        importlib.reload(logging)
        FORMAT = '%(asctime)s:%(levelname)s:%(name)s  %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    def _create_phase_screen_generator(self):
        r0 = 0.3
        self._psg = PhaseScreenGenerator.load_normalized_phase_screens(
            package_data.phase_screen_folder() / (self.PHASE_SCREEN_TAG+'.fits'))
        self._psg.rescale_to(r0)

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
        return self._subaps

    @cached_property
    def slope_computer(self):
        return self._sc

    @cached_property
    def slm_rasterizer(self):
        return SlmRasterizer()

    @cached_property
    def modal_decomposer(self):
        
        if self.MODAL_DEC_TAG is None:
            return ModalDecomposer(self.N_ZERNIKE_MODES_TO_CORRECT)
        
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
        return self._psg
