from bronte.mains.main250429shwfs_frames_tilt_scan import load
from bronte.utils.slopes_vector_analyser import SubapertureSetAndSlopesAnalyser
from bronte.startup import set_data_dir
from astropy.io import fits
from bronte.utils.camera_master_bkg import CameraMasterMeasurer
import matplotlib.pyplot as plt
import numpy as np 

def main():
    
    set_data_dir()
    fpath = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\bronte\\shwfs_frames\\"
    fname = fpath + "250505_095200.fits"
    hudList = fits.open(fname)
    frame_raw = hudList[0].data
    
    sh_master_bkg, _ = CameraMasterMeasurer.load_master('250211_135800')
    
    frame = frame_raw - sh_master_bkg
    frame[frame < 0] = 0
    
    subap_tag = '250120_122000'
    ssa = SubapertureSetAndSlopesAnalyser(subap_tag)
    ssa.display_shframe(frame)
    
    slopes = ssa.get_slopes_from_frame(frame)
    ssa.display2Dslope_maps_from_slope_vector(slopes)


def main_scan_tilt():
    
    set_data_dir()
    frames_ftag = '250430_144800' #Z2
    frames_ftag = '250430_145200' #Z3
    hdr, frame_cube, ref_frame, coef_vector = load(frames_ftag)
    
    subap_tag = '250120_122000'
    
    ssa = SubapertureSetAndSlopesAnalyser(subap_tag)
    
    # index relative to +-5um rms wf
    idx_min = 25
    idx_plus = 35
    amp_in_meters = coef_vector[idx_plus]
    ssa.display_shframe(frame_cube[idx_plus] - frame_cube[idx_min])
    
    Nsubap = ssa._subapertures_set.n_subaps
    Npixelpersub = ssa._subapertures_set.np_sub
    pixel_size_in_meters = 5.5e-6
    NofPushPull = 2
    
    ssa.reload_slope_pc(pix_thr_ratio=0.17, abs_pix_thr=0)
    
    # the slopes from spc are normalized to -1 to +1
    s_plus = ssa.get_slopes_from_frame(frame_cube[idx_plus])
    s_min = ssa.get_slopes_from_frame(frame_cube[idx_min])
   
    s_if_in_pixel = (0.5*Npixelpersub*pixel_size_in_meters)*(s_plus - s_min)/(NofPushPull*amp_in_meters)
    
    ssa.display2Dslope_maps_from_slope_vector(s_if_in_pixel)
    print(ssa.get_rms_slopes(s_if_in_pixel))
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
    
def main_scan_tilt_vs_thr():
    
    set_data_dir()
    #frames_ftag = '250430_144800' #Z2
    frames_ftag = '250430_145200' #Z3
    hdr, frame_cube, ref_frame, coef_vector = load(frames_ftag)
    
    subap_tag = '250120_122000'
    
    ssa = SubapertureSetAndSlopesAnalyser(subap_tag)
    
    # index relative to +-5um rms wf
    idx_min = 25
    idx_plus = 35
    amp_in_meters = coef_vector[idx_plus]
    ssa.display_shframe(frame_cube[idx_plus] - frame_cube[idx_min])
    
    Nsubap = ssa._subapertures_set.n_subaps
    Npixelpersub = ssa._subapertures_set.np_sub
    pixel_size_in_meters = 5.5e-6
    NofPushPull = 2
    
    thr_ratio_vector = np.array([0.0, 0.10, 0.15, 0.20, 0.30, 0.5])#, 0.99])
    
    plt.figure()
    plt.clf()
    
    for thr in thr_ratio_vector:
        ssa.reload_slope_pc(pix_thr_ratio = thr, abs_pix_thr=0)
    
        # the slopes from spc are normalized to -1 to +1
        s_plus = ssa.get_slopes_from_frame(frame_cube[idx_plus])
        s_min = ssa.get_slopes_from_frame(frame_cube[idx_min])
   
        s_if_in_pixel = (0.5*Npixelpersub*pixel_size_in_meters)*(s_plus - s_min)/(NofPushPull*amp_in_meters)
        plt.plot(s_if_in_pixel, label=f"thr_ratio={thr}")
 
    f2 = 250e-3
    f3 = 150e-3
    fla = 8.31477e-3
    D = 10.2e-3
    
    s_exp_in_pixel = (4*amp_in_meters*(f2/f3)*fla/D)/pixel_size_in_meters
    
    plt.legend(loc = 'best')
    plt.grid('--', alpha = 0.3)
    plt.ylabel("Slopes [pixel]")
    plt.title(f"c = {amp_in_meters} m rms wf: Expected slope {s_exp_in_pixel} pixels")