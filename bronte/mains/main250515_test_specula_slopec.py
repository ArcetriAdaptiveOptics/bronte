from astropy.io import fits
from bronte.startup import set_data_dir
from bronte.package_data import shframes_folder, modal_offsets_folder
import numpy as np 
from bronte.utils.slopes_vector_analyser import SlopesVectorAnalyser
import matplotlib.pyplot as plt


def main_test_specula_slopec():
    
    subap_tag = '250120_122000'
    sva = SlopesVectorAnalyser(subap_tag)
    NpixperSub = sva._subapertures_set.np_sub
    fake_sh_frame = get_fake_sh_frame(NpixperSub, shift_x = 12, shift_y = 0)
    
    frame = fake_sh_frame + 0.5*sva._subaperture_grid_map
    
    plt.figure()
    plt.clf()
    plt.imshow(fake_sh_frame + 0.5*sva._subaperture_grid_map, vmin=0, vmax=1)
    plt.colorbar()
    
    roi = frame[791:791+3*NpixperSub+8,970:970+3*NpixperSub+8]
    plt.figure()
    plt.clf()
    plt.imshow(roi, vmin=0, vmax=1)
    plt.colorbar()
    
    sva.reload_slope_pc(pix_thr_ratio = 0.18, abs_pix_thr = 0)
    slopes = sva.get_slopes_from_frame(fake_sh_frame)
    
    plt.figure()
    plt.clf()
    plt.plot(slopes)
    plt.xlabel('2Nsubap')
    plt.ylabel('Slopes [normalized]')
    plt.grid('--', alpha = 0.3)
    sva.display2Dslope_maps_from_slope_vector(slopes)
    
    return slopes, sva

def test_slopes_vs_thr(shift_x = 0, shift_y = 0):
    '''
    this main is meant to test the output slopes as a function
    of the threshold
    '''
    subap_tag = '250120_122000'
    sva = SlopesVectorAnalyser(subap_tag)
    NpixperSub = sva._subapertures_set.np_sub
    fake_sh_frame = get_fake_sh_frame(NpixperSub, shift_x, shift_y, addNoise=False)
    #fake_sh_frame[fake_sh_frame==0]=0.2
    thr_vector = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5])
    
    frame = fake_sh_frame + 0.5*sva._subaperture_grid_map
    plt.figure()
    plt.clf()
    plt.imshow(frame, vmin=0, vmax=1)
    plt.colorbar()
    
    roi = frame[791:791+3*NpixperSub+8,970:970+3*NpixperSub+8]
    plt.figure()
    plt.clf()
    plt.imshow(roi, vmin=0, vmax=1)
    plt.colorbar()
    
    
    plt.figure()
    plt.clf()
    
    for thr in thr_vector:
        sva.reload_slope_pc(pix_thr_ratio = thr, abs_pix_thr=0)
        s = sva.get_slopes_from_frame(fake_sh_frame)
        plt.plot(s, label = f"thr_ratio = {thr}")
    
    for thr in thr_vector:
        sva.reload_slope_pc(pix_thr_ratio = thr, abs_pix_thr=0)
        s = sva.get_slopes_from_frame(fake_sh_frame)
        plt.plot(s,'--' ,label = f"thr_ABS = {thr}")
    
    plt.grid('--', alpha = 0.3)
    plt.legend(loc = 'best')
    plt.xlabel('2Nsubap')
    plt.ylabel('Normalized Slopes')
    

def test_slopes_vs_shift():
    
    subap_tag = '250120_122000'
    sva = SlopesVectorAnalyser(subap_tag)
    NpixperSub = sva._subapertures_set.np_sub
    Nsubap = sva._subapertures_set.n_subaps
    
    shifts_vector = np.arange(-13,14)
    Nshifts = len(shifts_vector)
    sx_mean = np.zeros(Nshifts)
    sy_mean = np.zeros(Nshifts)
    err_sx = np.zeros(Nshifts)
    err_sy = np.zeros(Nshifts)
    
    for idx, shift in enumerate(shifts_vector):
        fake_sh_frame = get_fake_sh_frame(NpixperSub, shift_x = np.int32(shift), shift_y = 0, addNoise=False)
        sva.reload_slope_pc(pix_thr_ratio = 0.18, abs_pix_thr = 0)
        s = sva.get_slopes_from_frame(fake_sh_frame)
        sx_mean[idx] = s[:Nsubap].mean()
        err_sx[idx] = s[:Nsubap].std()
        sy_mean[idx] = s[Nsubap:].mean()
        err_sy[idx] = s[Nsubap:].std()
    
    
    sh_vect = shifts_vector.copy()
    #sh_vect[sh_vect>0] =sh_vect[sh_vect>0] + 0.5
    #sh_vect[sh_vect<0] =sh_vect[sh_vect<0] - 0.5
    s_exp = sh_vect/12.5

    plt.figure()
    plt.clf()
    plt.plot(sh_vect, s_exp, 'r.-', label = '$S_{exp}$')
    plt.errorbar(shifts_vector, sx_mean, err_sx, fmt='.-', label='$S_x$')
    plt.errorbar(shifts_vector, sy_mean, err_sy, fmt='.-', label='$S_y$')
    plt.xlabel('Shift-x in pixels')
    plt.ylabel('Normalized Slopes')
    plt.legend(loc = 'best')
    plt.grid('--', alpha = 0.3)
    
def get_fake_sh_frame(NpixperSub = 26, shift_x = 0, shift_y = 0, addNoise=False):
    '''
    returns a simulated sh frames with 2x2 uniform spots displaced of
    shift_x and shift_y pixels wrt the center of the subaperture
    '''
    #NpixperSub = sva._subapertures_set.np_sub
    frame_size = 2048
    Nsubap_on_diameter_along_x = 44
    Nsubap_on_diameter_along_y = 46
    
    sim_spot_size = 2
    
    y_ref = (822 - NpixperSub*0.5) -(NpixperSub*Nsubap_on_diameter_along_y*0.5)
    x_ref = (453-26) + NpixperSub*0.5
    
    #lenslet_row = np.zeros((NpixperSub, Nsubap_on_diameter*NpixperSub))
    fake_sh_frame = np.zeros((frame_size, frame_size))
    if addNoise is True:
        fake_sh_frame = np.random.random((frame_size,frame_size))*0.2
    
    for col_idx in range(Nsubap_on_diameter_along_y):
        yc = np.int32(y_ref + col_idx*NpixperSub) + shift_y
        #print(f"yc = {yc}")
        for row_idx in range(Nsubap_on_diameter_along_x):
            
            xc = np.int32(x_ref + row_idx*NpixperSub) + shift_x
            #print(f"xc = {xc}")
            fake_sh_frame[yc-1:yc+1, xc-1:xc+1] = np.ones((sim_spot_size, sim_spot_size))
    
    return fake_sh_frame