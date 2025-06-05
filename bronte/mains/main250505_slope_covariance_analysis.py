from bronte.startup import set_data_dir
from bronte.package_data import shframes_folder
from bronte.utils.slopes_covariance_matrix_analyser import SlopesCovariaceMatrixAnalyser
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

def main():
    
    set_data_dir()
    #loading frame cubes
    ftag_flat = '250507_142000'#'250505_151700'
    ftag_tip = '250507_142300'#'250505_152300'
    ftag_offset = '250514_172200'
    ftag = ftag_flat #ftag_offset
    fname = shframes_folder() / (ftag + '.fits')
    hdl = fits.open(fname)
    frame_cube = hdl[0].data
    
    subap_tag = '250120_122000'
    scma = SlopesCovariaceMatrixAnalyser(subap_tag)
    scma.set_slopes_from_frame_cube(frame_cube, pix_thr_ratio=0.18, abs_pix_thr=0)
    
    scma.display_rms_slopes()
    scma.display_slope_covariance_matrix()
    
    plt.figure()
    plt.clf()
    plt.imshow(scma._slope_covariance_matrix, vmin = -2e-4,vmax=1e-5);
    plt.colorbar()
    
    Nsub = scma.Nsubap
    NpixperSub = scma.NpixperSub
    sloeps_variance_in_pixel2 = scma._slope_covariance_matrix.diagonal()*(0.5*NpixperSub)**2
    slopex_var = sloeps_variance_in_pixel2[:Nsub]
    slopey_var = sloeps_variance_in_pixel2[Nsub:]
    #slopexy_var_in_pixel = slopex_var+ slopey_var
    
    count_in_adu = scma._flux_per_sub_cube.mean(axis=0)
    
    # plt.figure()
    # plt.clf()
    # plt.plot(count_in_adu)
    # plt.xlabel('Nsubap')
    # plt.ylabel('Mean counts [ADU]')
    
    sh_gain = 2.34 #e-/ADU
    QE = 0.62
    Nph = count_in_adu*sh_gain/QE
    theta_in_pixel = 36.55/5.5
    sigma_ron = (6.5 * sh_gain/QE)
    expected_slop_var = (theta_in_pixel**2)/Nph + (82/np.sqrt(theta_in_pixel))**4*(sigma_ron**2/Nph**2)
    
    plt.figure()
    plt.clf()
    plt.plot(expected_slop_var, 'k--', label= r"$\theta^2 /N_{ph}$")
    plt.plot(slopex_var + slopey_var, '-', label = "$\sigma^2_{slope}$")
    #plt.plot(slopex_var, '-', label = "$\sigma^2_{slope-x}$")
    #plt.plot(slopey_var, '-', label = "$\sigma^2_{slope-y}$")
    plt.xlabel('Nsubap')
    plt.ylabel('Slope Variance [pixel^2]')
    plt.grid('--', alpha = 0.3)
    plt.legend(loc = 'best')

    return scma
    
def main2():
    '''
    this main is meat to check who the slopes variance scales with the
    exposure time
    '''
    set_data_dir()
    #ftag_flat12ms = '250507_142000' 
    ftag4ms = '250515_145500'
    ftag8ms = '250515_150200'#'250505_151700' 
    ftag16ms = '250515_145900'
   
    fname4 = shframes_folder() / (ftag4ms + '.fits')
    hdl4 = fits.open(fname4)
    frame_cube4 = hdl4[0].data
   
    fname8 = shframes_folder() / (ftag8ms + '.fits')
    hdl8 = fits.open(fname8)
    frame_cube8 = hdl8[0].data
    
    fname16 = shframes_folder() / (ftag16ms + '.fits')
    hdl16 = fits.open(fname16)
    frame_cube16 = hdl16[0].data
    
    subap_tag = '250120_122000'
    scma = SlopesCovariaceMatrixAnalyser(subap_tag)
    
    Npixpersub = scma.NpixperSub
    Nsubaps =  scma.Nsubap
    
    scma.set_slopes_from_frame_cube(frame_cube4, pix_thr_ratio = 0.18, abs_pix_thr = 0)
    slopes_var_in_pixels4 = (scma._slopes_cube.std(axis = 0)*0.5*Npixpersub)**2
    count_in_adu4 = scma._flux_per_sub_cube.mean(axis=0)
    
    scma.set_slopes_from_frame_cube(frame_cube8, pix_thr_ratio = 0.18, abs_pix_thr = 0)
    slopes_var_in_pixels8 = (scma._slopes_cube.std(axis = 0)*0.5*Npixpersub)**2
    count_in_adu8 = scma._flux_per_sub_cube.mean(axis=0)
    
    scma.set_slopes_from_frame_cube(frame_cube16, pix_thr_ratio = 0.18, abs_pix_thr = 0)
    slopes_var_in_pixels16 = (scma._slopes_cube.std(axis = 0)*0.5*Npixpersub)**2
    count_in_adu16 = scma._flux_per_sub_cube.mean(axis=0)
    
    # count_in_adu8 = scma._flux_per_sub_cube.mean(axis=0)
    # sh_gain = 2.34 #e-/ADU
    # QE = 0.62
    # Nph = count_in_adu*sh_gain*QE
    # theta_in_pixel = 36.55/5.5
    
    plt.figure()
    plt.clf()
    plt.plot(slopes_var_in_pixels4, '-', label = '$\sigma^2_s @4ms$')
    plt.plot(slopes_var_in_pixels8, '-', label = '$\sigma^2_s @8ms$')
    plt.plot(slopes_var_in_pixels16, '-', label = '$\sigma^2_s @16ms$')
    plt.xlabel('2Nsubap')
    plt.ylabel('Slope Variance [pixel^2]')
    plt.grid('--', alpha = 0.3)
    plt.legend(loc = 'best')
    
    var_ratiox2 =  slopes_var_in_pixels8/slopes_var_in_pixels16
    var_ratiox4 = slopes_var_in_pixels4/slopes_var_in_pixels16
    meanx2 = var_ratiox2.mean()
    errx2 = var_ratiox2.std()
    meanx4 = var_ratiox4.mean()
    errx4 = var_ratiox4.std()
    plt.figure()
    plt.clf()
   
    plt.plot(var_ratiox2 ,'b-', label='texp x2')
    plt.plot(var_ratiox4, 'r-', label='texp x4')
    plt.hlines(meanx2, 0, len(var_ratiox2), ls='--', color='k', label = 'mean x2')
    plt.hlines(meanx4, 0, len(var_ratiox4), ls='--', color='g', label = 'mean x4')
    plt.legend(loc='best')
    plt.grid('--', alpha = 0.3)
    plt.ylabel('Slope Variance ratio')
    plt.xlabel('2Nsubap')
    
    print(f'mean x 2 : {meanx2} +/- {errx2}')
    print(f'mean x 4 : {meanx4} +/- {errx4}')
    
def main3():
    '''
    checking slopes variance as a function of the pixel thr
    '''
    set_data_dir()
    ftag_flat = '250505_151700'#'250507_142000' #
    #ftag_tip = '250507_142300'
    ftag = ftag_flat
    fname = shframes_folder() / (ftag + '.fits')
    hdl = fits.open(fname)
    frame_cube = hdl[0].data
    
    subap_tag = '250120_122000'
    scma = SlopesCovariaceMatrixAnalyser(subap_tag)
    
    Npixpersub = scma.NpixperSub
    Nsubaps =  scma.Nsubap
    
    thr_ratio_vector = np.array([0, 0.15, 0.20, 0.25, 0.3, 0.4, 0.5])
    thr_abs_vector = np.array([0, 100, 200, 300, 400])
    slopes_var_list = []
    for thr_ratio in thr_ratio_vector:
        scma.set_slopes_from_frame_cube(frame_cube, pix_thr_ratio = thr_ratio, abs_pix_thr = 0)
        slopes_var_in_pixels = (scma._slopes_cube.std(axis = 0)*0.5*Npixpersub)**2
        slopes_var_list.append(slopes_var_in_pixels)
    
    plt.figure()
    plt.clf()
    for idx, thr_ratio in enumerate(thr_ratio_vector):
        plt.plot(slopes_var_list[idx], '-', label=f'thr_ratio = {thr_ratio}')
        print(f"f'thr_ratio = {thr_ratio}: median {np.median(slopes_var_list[idx])}")
    plt.legend(loc='best')
    plt.grid('--', alpha = 0.3)
    plt.xlabel('2Nsubap')
    plt.ylabel('Slopes variance [pix^2]')
    
    
    slopes_var_list = []
    for thr_abs in thr_abs_vector:
        scma.set_slopes_from_frame_cube(frame_cube, pix_thr_ratio = 0, abs_pix_thr = thr_abs)
        slopes_var_in_pixels = (scma._slopes_cube.std(axis = 0)*0.5*Npixpersub)**2
        slopes_var_list.append(slopes_var_in_pixels)
    
    plt.figure()
    plt.clf()
    for idx, thr_abs in enumerate(thr_abs_vector):
        plt.plot(slopes_var_list[idx], '-', label=f'thr_abs = {thr_abs}ADU')
        print(f"f'thr_ABS = {thr_abs}: median {np.median(slopes_var_list[idx])}")
    plt.legend(loc='best')
    plt.grid('--', alpha = 0.3)
    plt.xlabel('2Nsubap')
    plt.ylabel('Slopes variance [pix^2]')