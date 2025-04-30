import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np
from specula.processing_objects.sh_slopec import ShSlopec
from specula.data_objects.subap_data import SubapData
from specula.data_objects.pixels import Pixels
from bronte.mains.main250429shwfs_frames_tilt_scan import load
from bronte.startup import set_data_dir
from bronte.package_data import subaperture_set_folder
import copy
from bronte.utils.from_slope_vector_to_slope_maps import SlopeVectorTo2DMap
import matplotlib.pyplot as plt

def main():
    
    ftag = '250430_144800'
    sa = SlopesAnalyser(ftag)
    sa.set_sh_slope_pc()
    
    # index relative to +-5um rms wf
    idx_min = 25
    idx_plus = 35
    amp_in_meters = sa._coef_vector[idx_plus]
    
    # the slopes from spc are normalized to -1 to +1
    s_plus = sa.compute_slopes_from_frame(sa._frame_cube[idx_plus])
    s_min = sa.compute_slopes_from_frame(sa._frame_cube[idx_min])
   
    Nsubap = sa._subapertures_set.n_subaps
    Npixelpersub = 26
    pixel_size_in_meters = 5.5e-6
    NofPushPull = 2
    
    s_if_in_pixel = (0.5*Npixelpersub*pixel_size_in_meters)*(s_plus - s_min)/(NofPushPull*amp_in_meters)
    
    s_median_x = np.median(s_if_in_pixel[:Nsubap])
    print(f"{s_median_x} pixel")
    
    f2 = 250e-3
    f3 = 150e-3
    fla = 8.31477e-3
    D = 10.2e-3
    
    s_exp_in_pixel = (4*amp_in_meters*(f2/f3)*fla/D)/pixel_size_in_meters
    
    plt.figure()
    plt.clf()
    plt.plot(s_if_in_pixel, label='measured')
    plt.grid('--', alpha = 0.3)
    plt.ylabel("Slopes [pixel]")
    plt.title(f"c = {amp_in_meters} m rms wf: Expected slope {s_exp_in_pixel} pixels")
        
class SlopesAnalyser():
    
    #DEFAULT 
    SH_PIX_THR = 200
    SUBAPS_TAG = '250120_122000'
    
    def __init__(self, ftag = '250430_144800'):
        set_data_dir()
        self._hdr, self._frame_cube, self._ref_frame, self._coef_vector = load(ftag)
        self._frame_cube[self._frame_cube < 0] = 0
        self._ref_frame[self._ref_frame < 0] = 0
        self._subapertures_set = self.load_subaperture_set(None)
        self._s2map = SlopeVectorTo2DMap(self.SUBAPS_TAG)
    
    def set_sh_slope_pc(self, pix_thr = None, subap_set = None):
        
        if pix_thr is None:
            pix_thr = self.SH_PIX_THR
        if subap_set is None:
            subap_set = self._subapertures_set
        
        self._slopec = ShSlopec(subapdata= subap_set, thr_value = pix_thr)
        
    def compute_slopes_from_frame(self, frame):
        frame_shape = frame.shape
        pix = Pixels(dimx = frame_shape[1], dimy = frame_shape[0])
        pix.pixels = frame
        
        self._slopec.inputs['in_pixels'].set(pix)
        self._slopec.calc_slopes_nofor()
        s = copy.copy(self._slopec.slopes.slopes)
        return s
    
    def get2D(self, slope_vector):
        return self._s2map.get2Dmaps_from_slopes_vector(slope_vector)
    
    def display2Dslope(self, slope_vector):
        self._s2map.display_2Dmap_from_slope_vector(slope_vector)
    
    def load_subaperture_set(self, ftag_subap = None):
        if ftag_subap is None:
            ftag_subap = self.SUBAPS_TAG
            
        self._s2map = SlopeVectorTo2DMap(ftag_subap)
        return SubapData.restore_from_bronte(
                subaperture_set_folder() / (ftag_subap + ".fits"))