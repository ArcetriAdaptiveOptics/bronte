from bronte.startup import set_data_dir
from bronte.package_data import reconstructor_folder
from bronte.calibration.experimental_push_pull_amplitude_computer import ExperimentalPushPullAmplitudeComputer
from astropy.io import fits
from bronte.calibration.mains.main250609_on_axis_calibratrion import eris_like_calib
import matplotlib.pyplot as plt
import numpy as np 

def main():
    set_data_dir()
    
    subap_tag = '250612_143100'#'250610_140500'#
    intmat_tag = '250613_111400'#'250613_094300'#'250612_150500'#'250611_155700'#
    calib_tag = '_bronte_calib_config'
    
    file_name = reconstructor_folder() / (intmat_tag + calib_tag + '.fits')
    config_data = fits.open(file_name)
    pp_vect_in_nm = config_data[0].data
    #pp_vect_in_nm = eris_like_calib()
    epp = ExperimentalPushPullAmplitudeComputer(subap_tag, intmat_tag, pp_vect_in_nm)
    
    epp.display_ifs_std()
    epp.compute_rescaled_pp_vector(target_val = 0.10)
    epp.display_pp_amplitude_vector()
    
    epp._dsm.display_all_slope_maps(size = 45, ncols=5, nrows=4)
    
    return epp
    #epp.save_rescaled_pp_vector(ftag='250612_161500')

def _get_pp_vector_in_nm(intmat_tag):
    calib_tag = '_bronte_calib_config'
    file_name = reconstructor_folder() / (intmat_tag + calib_tag + '.fits')
    config_data = fits.open(file_name)
    pp_vect_in_nm = config_data[0].data
    return pp_vect_in_nm

def compare_calib_with_subapset_250612_143100():
    
    set_data_dir()
    subap_tag = '250612_143100'
   
    
    intmat_tag0 = '250613_094300' 
    intmat_tag1 = '250613_102700'
    intmat_tag2 = '250613_111400'
    
    pp_vect_in_nm0 = _get_pp_vector_in_nm(intmat_tag0)
    pp_vect_in_nm1 = _get_pp_vector_in_nm(intmat_tag1)
    pp_vect_in_nm2 = _get_pp_vector_in_nm(intmat_tag2)
    
    epp0 = ExperimentalPushPullAmplitudeComputer(subap_tag, intmat_tag0, pp_vect_in_nm0)
    epp1 = ExperimentalPushPullAmplitudeComputer(subap_tag, intmat_tag1, pp_vect_in_nm1)
    epp2 = ExperimentalPushPullAmplitudeComputer(subap_tag, intmat_tag2, pp_vect_in_nm2)
    
    j_noll_vector = np.arange(200) + 2
    
    plt.figure()
    plt.clf()
    plt.plot(j_noll_vector, epp0._im_std, label=intmat_tag0)
    plt.plot(j_noll_vector, epp1._im_std, label=intmat_tag1)
    plt.plot(j_noll_vector, epp2._im_std, label=intmat_tag2)
    plt.grid('--', alpha=0.3)
    plt.legend(loc='best')
    plt.ylabel('Slopes (IFS) std [normalized]')
    plt.xlabel('Mode index')
    
    plt.figure()
    plt.clf()
    plt.plot(j_noll_vector, pp_vect_in_nm0, label=intmat_tag0)
    plt.plot(j_noll_vector, pp_vect_in_nm1, label=intmat_tag1)
    plt.plot(j_noll_vector, pp_vect_in_nm2, label=intmat_tag2)
    plt.ylabel('Push-Pull [nm] rms wf')
    plt.xlabel('j index')
    plt.grid('--', alpha=0.3)
    plt.legend(loc='best')
    
    epp0._dsm.display_all_slope_maps(size = 45, ncols=5, nrows=40, title=intmat_tag0)
    epp1._dsm.display_all_slope_maps(size = 45, ncols=5, nrows=40, title=intmat_tag1)
    epp2._dsm.display_all_slope_maps(size = 45, ncols=5, nrows=40, title=intmat_tag2)
    