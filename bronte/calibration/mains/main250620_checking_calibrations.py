from bronte.startup import set_data_dir
from bronte.package_data import reconstructor_folder
from bronte.calibration.utils.experimental_push_pull_optimizer import ExperimentalPushPullOptimizer
from astropy.io import fits
from bronte.calibration.mains.main250609_on_axis_calibratrion import eris_like_calib
import matplotlib.pyplot as plt
import numpy as np 


def main():
    set_data_dir()
    
    subap_tag = '250612_143100'
    intmat_tag = '250619_141800'#'250617_165500'
    calib_tag = '_bronte_calib_config'
    
    pp_vect_in_nm, epp, im, ss, s, mean_slope =_get_stuff_from_calib(subap_tag, intmat_tag)
    
    Nmodes = im.shape[0]
    j_noll_vector = np.arange(0,Nmodes) + 2
    
    epp._dsm.display_all_slope_maps(size=45, ncols=5, nrows=5)
    
    plt.figure()
    plt.clf()
    plt.plot(s/s.max(), label = intmat_tag)
    plt.grid('--', alpha=0.3)
    plt.legend(loc='best')
    plt.xlabel('Eigen-Mode index')
    plt.ylabel('Eigen-Value [Normalized]')
    print(f"{intmat_tag}: Eigen-value ratio (MAX/MIN) = {s.max()/s.min()} ")

    sx_map, sy_map = epp._dsm.get_slope_maps()

    plt.figure()
    plt.clf()
    plt.plot(epp._dsm._ifs[0], label = 'Tip')
    plt.plot(epp._dsm._ifs[1], label = 'Tilt')
    plt.grid('--', alpha=0.3)
    plt.legend(loc='best')
    plt.xlabel('2Nsubaps index')
    plt.ylabel('Slopes [Normalized]')
    
    plt.figure()
    plt.clf()
    plt.plot(j_noll_vector[2:], mean_slope[2:], '.-', label = intmat_tag)
    plt.grid('--', alpha=0.3)
    plt.legend(loc='best')
    plt.xlabel('j Mode index')
    plt.ylabel('Slopes mean [normalized]')
    
    return sx_map, sy_map
    
def _get_pp_vector_in_nm(intmat_tag):
    calib_tag = '_bronte_calib_config'
    file_name = reconstructor_folder() / (intmat_tag + calib_tag + '.fits')
    config_data = fits.open(file_name)
    pp_vect_in_nm = config_data[0].data
    return pp_vect_in_nm


def _get_stuff_from_calib(subap_tag, intmat_tag):
    
    pp_vect_in_nm = _get_pp_vector_in_nm(intmat_tag)
    epp = ExperimentalPushPullOptimizer(subap_tag, intmat_tag, pp_vect_in_nm)
    im = epp._dsm._intmat._intmat
    u,s,vh = np.linalg.svd(im)
    ss = (s - s.min())/(s.max() - s.min())
    mean_slope = epp._dsm._ifs.mean(axis=-1)
    return pp_vect_in_nm, epp, im, ss, s,  mean_slope