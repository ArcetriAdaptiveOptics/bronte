from astropy.io import fits
from bronte.startup import set_data_dir
from bronte.package_data import shframes_folder, modal_offsets_folder
import numpy as np 
from bronte.utils.slopes_vector_analyser import SlopesVectorAnalyser
import matplotlib.pyplot as plt

def check_frame_after_thr():
    '''
    this main is meant to check the who the frame looks like
    after thresholding
    '''
    set_data_dir()
    # loading TT scan data
    ftag ='250512_102100'# Z2 data
    file_name = shframes_folder() / (ftag + '.fits')
    hdr = fits.getheader(file_name)
    #jnoll_index = hdr['NOLL_J']
    hdl = fits.open(file_name)
    frame_cube = hdl[0].data 
    #Nframes = frame_cube.shape[0]
    ref_frame = hdl[1].data
    ref_frame[ref_frame<0] = 0
    c_vector = hdl[2].data
    
    subap_tag = '250120_122000'
    sva = SlopesVectorAnalyser(subap_tag)
    sva.reload_slope_pc(pix_thr_ratio = 0.18, abs_pix_thr = 0)
    #Nsubap = sva._subapertures_set.n_subaps
    #NpixperSub = sva._subapertures_set.np_sub
    
    
    plt.figure()
    plt.clf()
    plt.title('Reference frame before thr')
    fr2disp = ref_frame + 1000*sva._subaperture_grid_map
    plt.imshow(fr2disp[840:955, 940:1054])
    plt.colorbar()
    
    thr_abs = 200
    sva.reload_slope_pc(pix_thr_ratio = 0, abs_pix_thr = thr_abs)
    s = sva.get_slopes_from_frame(ref_frame)
    fr_abs_thr = sva.get_frame_after_thresholding()
    plt.figure()
    plt.clf()
    plt.title(f'Reference frame after abs thr = {thr_abs}')
    fr2disp = fr_abs_thr + 1000*sva._subaperture_grid_map
    plt.imshow(fr2disp[840:955, 940:1054])
    plt.colorbar()
    
    thr_ratio = 0.18
    sva.reload_slope_pc(pix_thr_ratio = thr_ratio, abs_pix_thr = 0)
    s = sva.get_slopes_from_frame(ref_frame)
    fr_thr_ratio = sva.get_frame_after_thresholding()
    plt.figure()
    plt.clf()
    plt.title(f'Reference frame after thr_ratio = {thr_ratio}')
    fr2disp = fr_thr_ratio + 1000*sva._subaperture_grid_map
    plt.imshow(fr2disp[840:955, 940:1054])
    plt.colorbar()
    
    return sva
    

def check_spot_displacement():
    '''
    this main is meant to compute the displacement
    of the spot in a fixed subaperture with a simple 
    algorithm. then this is compared to the expected
    slope and the one computed by specula slopec  
    '''
    set_data_dir()
    # loading TT scan data
    ftag ='250512_102100'# Z2 data
    file_name = shframes_folder() / (ftag + '.fits')
    hdr = fits.getheader(file_name)
    #jnoll_index = hdr['NOLL_J']
    hdl = fits.open(file_name)
    frame_cube = hdl[0].data 
    #Nframes = frame_cube.shape[0]
    ref_frame = hdl[1].data
    ref_frame[ref_frame<0] = 0
    c_vector = hdl[2].data
    
    subap_tag = '250120_122000'
    sva = SlopesVectorAnalyser(subap_tag)
    sva.reload_slope_pc(pix_thr_ratio = 0.18, abs_pix_thr = 0)
    NpixPerSub = sva._subapertures_set.np_sub
    Nsub = sva._subapertures_set.n_subaps
    f2 = 250e-3
    f3 = 150e-3
    fla = 8.31477e-3
    D = 568*2*9.2e-6
    pp = 5.5e-6
    idx = 35
    c = c_vector[idx]
    dx_in_pixels = (f2/f3)*(4*c/D)*fla/pp
     
    plt.figure()
    plt.clf()
    plt.title('Reference frame')
    fr2disp = ref_frame + 1000*sva._subaperture_grid_map
    plt.imshow(fr2disp[840:955, 940:1054])
    plt.colorbar()
    
    plt.figure()
    plt.clf()
    plt.title(f'tilt command frame')
    fr2disp = frame_cube[idx] + 1000*sva._subaperture_grid_map
    plt.imshow(fr2disp[840:955, 940:1054])
    plt.colorbar()
    
    s_ref = sva.get_slopes_from_frame(ref_frame)
    
    s_tilt = sva.get_slopes_from_frame(frame_cube[idx])
    
    s_in_pixels = (s_tilt-s_ref)*0.5*NpixPerSub
    
    plt.figure()
    plt.clf()
    plt.plot(s_in_pixels,label ='computed w specula')
    plt.hlines(dx_in_pixels, 0, Nsub, color = 'k', ls='--', label = 'expected')
    plt.ylabel('Slopes [pixels]')
    plt.grid('--', alpha = 0.3)
    plt.legend(loc = 'best')
    sx_mean = s_in_pixels[:Nsub].mean()
    err_sx = s_in_pixels[:Nsub].std()
    print(f"s_exp = {dx_in_pixels} pixel")
    print(f"<sx> = {sx_mean} +/- {err_sx} pixel")
    
    
    
    ref_subap_roi = ref_frame[840:955, 940:1054] 
    ref_single_subap_roi = ref_subap_roi[60:86, 32:59]
    Iref = ref_single_subap_roi.max()
    y_ref = np.where(ref_single_subap_roi == Iref)[0][0]
    x_ref = np.where(ref_single_subap_roi == Iref)[1][0]
    
    y_ref,x_ref = get_centroid(ref_single_subap_roi)
    
    coeff_vect = c_vector[30:36]
    
    dx_in_pixels_vector = np.zeros(len(coeff_vect))
    dy_in_pixels_vector = np.zeros(len(coeff_vect))
    exp_slope_in_pixels = np.zeros(len(coeff_vect))
    
    specula_slopes = np.zeros(len(coeff_vect))
    err_specula_slopes = np.zeros(len(coeff_vect))
    
    plt.figure()
    plt.clf()
    plt.imshow(ref_single_subap_roi)
    plt.colorbar()
    plt.title('ref')
    
    for idx, coef in enumerate(coeff_vect):
        
        frame = frame_cube[30+idx]
        subaps_roi = frame[840:955, 940:1054] 
        single_subap_roi = subaps_roi[60:86, 32:59]
        Imax = single_subap_roi.max()
        yc = np.where(single_subap_roi == Imax)[0][0]
        xc = np.where(single_subap_roi == Imax)[1][0]
        
        yc, xc = get_centroid(single_subap_roi)
        
        dx_in_pixels_vector[idx] = xc - x_ref
        dy_in_pixels_vector[idx] = yc - y_ref
        exp_slope_in_pixels[idx] = (f2/f3)*(4*coef/D)*fla/pp
        
        s = sva.get_slopes_from_frame(frame) - s_ref
        sx_in_pixels = s[:Nsub]*0.5*NpixPerSub
        specula_slopes[idx] = sx_in_pixels.mean()
        err_specula_slopes[idx] = sx_in_pixels.std() 
        
        roi_thr = single_subap_roi.copy()
        thr = 180
        roi_thr-=thr
        roi_thr[roi_thr<0] = 0
        
        plt.figure()
        plt.clf()
        plt.imshow(roi_thr)
        plt.colorbar()
        plt.title(f'c={coef}')
        
    
    tc = coeff_vect/1e-6
    plt.figure()
    plt.clf()
    plt.plot(tc, dx_in_pixels_vector, '.-', label='simple centroid')
    plt.plot(tc, exp_slope_in_pixels, '-', label = 'Expected')
    plt.errorbar(tc , specula_slopes, err_specula_slopes, label='$<s_x>_{slopec}$')
    plt.legend(loc='best')
    plt.grid('--', alpha=0.3)
    plt.ylabel('Slopes [Pixel]')
    plt.xlabel('Tilt Coefficient um rms wf')
    
    return frame_cube, c_vector, sva

def get_centroid(roi, hs = 3):
        
    #single_subap_roi = subaps_roi[60:86, 32:59]

    roi_thr = roi.copy()
    thr = 180
    roi_thr-=thr
    roi_thr[roi_thr<0] = 0
    Imax = roi_thr.max()
    ym = np.where(roi_thr == Imax)[0][0]
    xm = np.where(roi_thr == Imax)[1][0]
    roi_window = roi_thr[ym-hs:ym + hs +1, xm - hs: xm + hs + 1]
    Itot_window = roi_window.sum()
    Npt = hs*2 +1
    
    xv = np.arange(1,Npt+1,1)
    yv = np.arange(1,Npt+1,1)
    
    x = 0
    y = 0
    
    for idx_x in range(Npt):
        for idx_y in range(Npt):
            
            x += xv[idx_x]*roi_window[idx_y,idx_x]/Itot_window
            y += yv[idx_y]*roi_window[idx_y,idx_x]/Itot_window
    
    xc = xm + (x -1 - Npt*0.5)
    yc = ym + (y -1 - Npt*0.5)
    
    return yc, xc
    