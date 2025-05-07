from bronte.startup import set_data_dir
from bronte.package_data import shframes_folder
from bronte.utils.slopes_covariance_matrix_analyser import SlopesCovariaceMatrixAnalyser
from astropy.io import fits
import matplotlib.pyplot as plt

def main():
    
    set_data_dir()
    #loading frame cubes
    ftag_flat = '250507_142000'#'250505_151700'
    ftag_tip = '250507_142300'#'250505_152300'
    ftag = ftag_tip
    fname = shframes_folder() / (ftag + '.fits')
    hdl = fits.open(fname)
    frame_cube = hdl[0].data
    
    subap_tag = '250120_122000'
    scma = SlopesCovariaceMatrixAnalyser(subap_tag)
    scma.set_slopes_from_frame_cube(frame_cube, pix_thr_ratio=0.2, abs_pix_thr=0)
    
    scma.display_rms_slopes()
    scma.display_slope_covariance_matrix()
    return scma
    
    
    

def main2():
    
    set_data_dir()
    ftag_flat = '250507_142000' #'250505_151700'
    #ftag_tip = '250507_142300'
    ftag = ftag_flat
    fname = shframes_folder() / (ftag + '.fits')
    hdl = fits.open(fname)
    frame_cube = hdl[0].data
    
    subap_tag = '250120_122000'
    scma = SlopesCovariaceMatrixAnalyser(subap_tag)
    
    Npixpersub = scma.NpixperSub
    Nsubaps =  scma.Nsubap
    
    scma.set_slopes_from_frame_cube(frame_cube, pix_thr_ratio = 0.35, abs_pix_thr=0)
    
    slopes_var_in_pixels = (scma._slopes_cube.std(axis = 0)*0.5*Npixpersub)**2
    
    
    count_in_adu = scma._flux_per_sub_cube.mean(axis=0)
    sh_gain = 2.34 #e-/ADU
    QE = 0.62
    Nph = count_in_adu*sh_gain/QE
    theta_in_pixel = 36.55/5.5
    
    expected_slop_var = theta_in_pixel**2/Nph
    measured_slope_varX = slopes_var_in_pixels[:Nsubaps]
    measured_slope_varY = slopes_var_in_pixels[Nsubaps:]
    
    plt.figure()
    plt.clf()
    plt.plot(expected_slop_var, 'k--', label= r"$\theta^2 /N_{ph}$")
    plt.plot(measured_slope_varX+measured_slope_varY, '-', label = "$\sigma^2_{slope}$")
    #plt.plot(measured_slope_varY, '-', label = "$\sigma^2_{slope_Y}$")
    plt.xlabel('Nsubap')
    plt.ylabel('Slope Variance [pixel^2]')
    plt.grid('--', alpha = 0.3)
    plt.legend(loc = 'best')
    