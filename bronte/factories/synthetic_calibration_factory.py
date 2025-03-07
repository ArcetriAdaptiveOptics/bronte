import specula
from _testconsole import read_output
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np
from bronte.factories.synthetic_base_factory import SyntheticBaseFactory
from specula.processing_objects.im_rec_calibrator import ImRecCalibrator
from specula.processing_objects.func_generator import FuncGenerator
from specula.data_objects.source import Source
from specula.processing_objects.atmo_propagation import AtmoPropagation
from specula.data_objects.layer import Layer
from specula.data_objects.subap_data import SubapData
from specula.processing_objects.sh_slopec import ShSlopec
from specula.processing_objects.sh import SH
from specula.processing_objects.ccd import CCD
from specula.processing_objects.dm import DM
from bronte.package_data import subaperture_set_folder, reconstructor_folder
from functools import cached_property
from bronte.utils.noll_to_radial_order import from_noll_to_radial_order

class SyntheticCalibrationFactory(SyntheticBaseFactory):
    
    SUBAPS_TAG = '250120_122000'
    N_MODES_TO_CORRECT = 200 
    TELESCOPE_PUPIL_DIAMETER = 40   # m
    SH_PIX_THR = 0
    PP_AMP_IN_NM = 2000
    TEXP_SH_CAM_IN_S = 8e-3
    WL_IN_NM = 633
    
    def __init__(self):
        
        super().__init__()
        self._pupil_diameter_in_pixel  = 2 * self.slm_pupil_mask.radius()
        self._pupil_pixel_pitch = round(self.TELESCOPE_PUPIL_DIAMETER/self._pupil_diameter_in_pixel, 3)
        self._subap_size_in_px = self.subapertures_set.np_sub
        
        
    @cached_property
    def source_dict(self):
        on_axis_source = Source(polar_coordinate=[0.0, 0.0], magnitude=8, wavelengthInNm = self.WL_IN_NM,)
        source_dict = {'on_axis_source': on_axis_source}
        return source_dict
    
    @cached_property
    def disturb_propagation(self):
        prop = AtmoPropagation(pixel_pupil = self._pupil_diameter_in_pixel,              # Linear dimension of pupil phase array
                               pixel_pitch = self._pupil_pixel_pitch,         # Linear dimension of pupil phase array
                               source_dict = self.source_dict,
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
    def virtual_sh(self):
        
        f_la = 8.31477e-3
        ccd_pixel_size = 5.5e-6 
        RAD2ARCSEC = 180/np.pi*3600
        pixel_scale_in_arcsec = RAD2ARCSEC * ccd_pixel_size/f_la 
        #pixel_scale_in_arcsec = 0.1
        subap_fov = pixel_scale_in_arcsec * self._subap_size_in_px
        eff_subap_on_diameter = 42
        print(subap_fov)
        print(f'{self._subap_size_in_px=}')
        print(f'{pixel_scale_in_arcsec=}')
        sh_lenslet = SH(
            subap_wanted_fov = subap_fov,
            sensor_pxscale = pixel_scale_in_arcsec,
            subap_npx = self._subap_size_in_px,
            subap_on_diameter = eff_subap_on_diameter,
            wavelengthInNm = self.WL_IN_NM
            )
        return sh_lenslet

    @cached_property
    def virtual_ccd(self):
        
        eff_subap_on_diameter = 42
        frame_size = self._subap_size_in_px * eff_subap_on_diameter
        detector = CCD(
            size = [frame_size, frame_size],
            dt = self.TEXP_SH_CAM_IN_S,
            bandw = 300,
            photon_noise = False,
            readout_noise = False,
            quantum_eff = 1)
        
        return detector

    @cached_property
    def push_pull(self):
        
        j_noll_vector = np.arange(self.N_MODES_TO_CORRECT) + 2
        radial_order = from_noll_to_radial_order(j_noll_vector)
        ampl_vect = self.PP_AMP_IN_NM /(radial_order) # in nm
        #ampl_vect = np.ones(self.N_MODES_TO_CORRECT)*500
        pp = FuncGenerator(func_type = 'PUSHPULL',
                   nmodes = self.N_MODES_TO_CORRECT,
                   vect_amplitude = ampl_vect,#in nm
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