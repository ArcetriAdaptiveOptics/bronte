import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np
from specula.lib.compute_zonal_ifunc import compute_zonal_ifunc
from specula.data_objects.ifunc import IFunc
from specula import cpuArray
from bronte.startup import set_data_dir
from bronte.package_data import ifs_folder

class ZonalInfluenceFunctionComputer():
    
    def __init__(self, pupil_diameter_in_pixel, Nact_on_diameter):
        
        self._pupil_diameter_in_pixels = pupil_diameter_in_pixel
        self._Nact_on_diameter = Nact_on_diameter
        #self._pupil_diameter_in_m =  pupil_diameter_in_m
        
        # Pupil geometry
        self._obs_ratio = 0.14              # 14% central obstruction
        self._dia_ratio = 1.0               # Full pupil diameter

        # Actuator geometry - aligned with test_modal_basis.py
        self._circGeom = True              # Circular geometry (better for round pupils)
        self._angleOffset = 0              # No rotation
        
        # Mechanical coupling between actuators
        self._doMechCoupling = False       # Enable realistic coupling
        self._couplingCoeffs = [0.31, 0.05] # Nearest and next-nearest neighbor coupling

        # Actuator slaving (disable edge actuators outside pupil)
        self._doSlaving = True             # Enable slaving (very simple slaving)
        self._slavingThr = 0.1             # Threshold for master actuators
        
        self._dtype = specula.xp.float32
        
        self._custom_pupil_mask = None
    
    def set_pupil_geometry(self, obstraction_ratio = 0.14, diameter_ratio = 1):
        
        self._obs_ratio = obstraction_ratio             
        self._dia_ratio = diameter_ratio
        
    def set_actuators_mechanical_coupling(self, couplingCoeffs = [0.31, 0.05]):
        self._doMechCoupling = True
        self._couplingCoeffs = couplingCoeffs
    
    def set_actuators_slaving(self, slaving_thr = 0.1):
        self._doSlaving = True             
        self._slavingThr = slaving_thr
        
    def load_custom_pupil_mask(self, custom_mask):
        
        self._custom_pupil_mask = custom_mask
    
    def compute_zonal_ifs(self, return_coordinates = False):
        
        self._ifs, self._pupil_mask_idl = compute_zonal_ifunc(
            self._pupil_diameter_in_pixels,
            self._Nact_on_diameter,
            circ_geom = self._circGeom,
            angle_offset = self._angleOffset,
            do_mech_coupling = self._doMechCoupling,
            coupling_coeffs = self._couplingCoeffs,
            do_slaving = self._doSlaving,
            slaving_thr=self._slavingThr,
            obsratio = self._obs_ratio,
            diaratio=self._dia_ratio,
            mask = self._custom_pupil_mask,
            xp = specula.xp,
            dtype = self._dtype,
            return_coordinates = return_coordinates)
    
    def get_zonal_ifs(self):
        
        return self._ifs, self._pupil_mask_idl
    
    def save_ifs(self, ftag):
        
        set_data_dir()
        self._ifunc_obj = IFunc(
            ifunc = self._ifs,
            mask = self._pupil_mask_idl)
        fname  = ifs_folder() / (ftag + '.fits')
        self._ifunc_obj.save(fname)
    
    def get_actuator_if_2Dmap(self, act_index):
        
        ifs_1d = self._ifs[act_index]
        frame_size = self._pupil_diameter_in_pixels
        masked_array_ifs_map = np.ma.array(
            data = np.zeros((frame_size, frame_size)),
            mask = 1 - self._pupil_mask_idl)
        masked_array_ifs_map[masked_array_ifs_map.mask == False] = ifs_1d
        
        return masked_array_ifs_map
    
    @staticmethod
    def load_ifs(ftag):
        set_data_dir()
        fname = ifs_folder() / (ftag + '.fits')
        return IFunc.restore(fname)