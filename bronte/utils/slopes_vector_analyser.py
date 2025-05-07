import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np
from specula.processing_objects.sh_slopec import ShSlopec
from specula.data_objects.subap_data import SubapData
from specula.data_objects.pixels import Pixels
import copy
from bronte.wfs.slope_computer import PCSlopeComputer
from bronte.wfs.subaperture_set import ShSubapertureSet
from bronte.utils.from_slope_vector_to_slope_maps import SlopeVectorTo2DMap
from bronte.startup import set_data_dir
from bronte.package_data import subaperture_set_folder

import matplotlib.pyplot as plt

class SlopesVectorAnalyser():
    
    ABSOLUTE_PIX_THR = 200
    RELATIVE_PIX_THR_RATIO = 0
    
    def __init__(self, subap_tag):
        set_data_dir()
        self._subap_tag = subap_tag
        self._s2map_display = None
        self._subaperture_grid_map = None
        self._subapertures_set = self._load_subaperture_set(subap_tag)
        self._slope_pc = None
        self._load_slope_pc()
        self._frame = None
        
    def _load_subaperture_set(self, subap_tag):
        self._s2map_display = SlopeVectorTo2DMap(self._subap_tag)
        self._load_subapertures_grid_map(subap_tag)
        return SubapData.restore_from_bronte(
            subaperture_set_folder() / (subap_tag + ".fits"))
        
    def _load_subapertures_grid_map(self, subap_tag):
        subap_set = ShSubapertureSet.restore(subaperture_set_folder() / (subap_tag + '.fits'))
        sc = PCSlopeComputer(subap_set)
        self._subaperture_grid_map = sc.subapertures_map()
        
    def _load_slope_pc(self, pix_thr_ratio = None, abs_pix_thr = None):
        if pix_thr_ratio is None:
            pix_thr_ratio = self.RELATIVE_PIX_THR_RATIO
        if abs_pix_thr is None:
            abs_pix_thr = self.ABSOLUTE_PIX_THR
        
        self._slopec = ShSlopec(subapdata= self._subapertures_set, thr_value = abs_pix_thr)
        self._slopec.thr_ratio_value = pix_thr_ratio
    
    def get_slopes_from_frame(self, frame = None, fluxperSub=False):
        if frame is None:
            frame = self._frame
            
        frame_shape = frame.shape
        pix = Pixels(dimx = frame_shape[1], dimy = frame_shape[0])
        pix.pixels = frame
        self._slopec.inputs['in_pixels'].set(pix)
        self._slopec.calc_slopes_nofor()
        s = copy.copy(self._slopec.slopes.slopes)
        if fluxperSub is True:
            flux_per_sub = copy.copy(self._slopec.flux_per_subaperture_vector.value)
            return s, flux_per_sub
        else:
            return s
    
    def reload_slope_pc(self, pix_thr_ratio = None, abs_pix_thr = None):
        self._load_slope_pc(pix_thr_ratio, abs_pix_thr)
    
    def load_shwfs_frame(self, frame):
        self._frame = frame
    
    def get_rms_slopes(self, slope_vector):
        Nsubap = self._subapertures_set.n_subaps
        rms_slope_x = np.sqrt(np.mean(slope_vector[0:Nsubap]**2))
        rms_slope_y = np.sqrt(np.mean(slope_vector[Nsubap:]**2))
        return rms_slope_x, rms_slope_y
    
    def get2Dslope_maps_from_slopes_vector(self, slope_vector):
        return self._s2map_display.get2Dmaps_from_slopes_vector(slope_vector)
    
    def get2Dslope_maps_from_frame(self, frame):
        slope_vector = self.get_slopes_from_frame(frame)
        return self.get2Dslope_maps_from_slopes_vector(slope_vector)
    
    def display2Dslope_maps_from_slope_vector(self, slope_vector):
        self._s2map_display.display_2Dmap_from_slope_vector(slope_vector)
        
    def display2Dslope_maps_from_frame(self, frame):
        slope_vector = self.get_slopes_from_frame(frame)
        self.display2Dslope_maps_from_slope_vector(slope_vector)
        
    def display_shframe(self, frame = None):
        if frame is None:
            frame = self._frame
        
        plt.figure()
        plt.clf()
        plt.imshow(frame + 1000*self._subaperture_grid_map)
        plt.colorbar()
    
    def display_subap_mask(self):
        plt.figure()
        plt.clf()
        plt.imshow(self._subapertures_set._single_mask)
        plt.colorbar()
        
    