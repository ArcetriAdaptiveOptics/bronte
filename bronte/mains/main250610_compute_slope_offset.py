from bronte.startup import set_data_dir
from bronte.package_data import shframes_folder, slope_offset_folder
from bronte.utils.slopes_covariance_matrix_analyser import SlopesCovariaceMatrixAnalyser
from bronte.utils.slopes_vector_analyser import SlopesVectorAnalyser
from astropy.io import fits
import matplotlib.pyplot as plt

def main():
    
    subap_tag = '250612_143100'#'250610_140500'
    #load file
    set_data_dir()
    fname = shframes_folder() / ('250610_143100.fits')
    hduList = fits.open(fname)
    frame_cube = hduList[0].data
    
    #select thr ratio looking at average sh frame
    average_frame = frame_cube.mean(axis = 0)
    
    plt.figure()
    plt.clf()
    plt.imshow(average_frame)
    plt.colorbar()
    
    sva = SlopesVectorAnalyser(subap_tag)
    thr_ratio = 0.18
    sva.reload_slope_pc(thr_ratio, 0)
    s = sva.get_slopes_from_frame(average_frame)
    frame_after_thr = sva.get_frame_after_thresholding()
    
    plt.figure()
    plt.clf()
    plt.imshow(frame_after_thr + sva._subaperture_grid_map*1000)
    plt.colorbar()
    
    #once the thr is selected compute the slope and save the as offset
    scma = SlopesCovariaceMatrixAnalyser(subap_tag)
    scma.set_slopes_from_frame_cube(frame_cube, thr_ratio, 0)
    scma._compute_average_slopes()
    s_average = scma.get_average_slopes()
    
    plt.figure()
    plt.clf()
    plt.plot(s_average)
    plt.grid('--', alpha=0.3)
    plt.xlabel('2Nsub')
    plt.ylabel('Slopes [normalized]')
    
    scma._sva.display2Dslope_maps_from_slope_vector(s_average)
    
    #scma.save_average_slopes_as_slope_offset('250610_150900')
    scma.save_average_slopes_as_slope_offset('250613_140600')