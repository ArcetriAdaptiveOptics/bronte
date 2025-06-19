from bronte.startup import set_data_dir
from bronte.package_data import reconstructor_folder
from bronte.calibration.utils.experimental_push_pull_optimizer import ExperimentalPushPullOptimizer
from astropy.io import fits
from bronte.calibration.mains.main250609_on_axis_calibratrion import eris_like_calib
import matplotlib.pyplot as plt
import numpy as np 

def main():
    set_data_dir()
    
    subap_tag = '250612_143100'#'250610_140500'#
    intmat_tag = '250613_094300'#'250616_103300'#'250613_102700'#'250613_094300'#'250612_150500'#'250611_155700'#
    calib_tag = '_bronte_calib_config'
    
    file_name = reconstructor_folder() / (intmat_tag + calib_tag + '.fits')
    config_data = fits.open(file_name)
    pp_vect_in_nm = config_data[0].data
    #pp_vect_in_nm = eris_like_calib()
    epp = ExperimentalPushPullOptimizer(subap_tag, intmat_tag, pp_vect_in_nm)
    
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
    intmat_tag1 = '250613_102700' # targ_val = 0.1 one iteration
    intmat_tag2 = '250613_111400' # targ_val = 0.16 one iteration
    
    pp_vect_in_nm0 = _get_pp_vector_in_nm(intmat_tag0)
    pp_vect_in_nm1 = _get_pp_vector_in_nm(intmat_tag1)
    pp_vect_in_nm2 = _get_pp_vector_in_nm(intmat_tag2)
    
    epp0 = ExperimentalPushPullOptimizer(subap_tag, intmat_tag0, pp_vect_in_nm0)
    epp1 = ExperimentalPushPullOptimizer(subap_tag, intmat_tag1, pp_vect_in_nm1)
    epp2 = ExperimentalPushPullOptimizer(subap_tag, intmat_tag2, pp_vect_in_nm2)
    
    im0 = epp0._dsm._intmat._intmat.T
    im1 = epp1._dsm._intmat._intmat.T
    im2 = epp2._dsm._intmat._intmat.T
    
    u0,s0,vh0 = np.linalg.svd(im0)
    u1,s1,vh1 = np.linalg.svd(im1)
    u2,s2,vh2 = np.linalg.svd(im2)
    
    print(f"{intmat_tag0}: Eigen-value ratio (MAX/MIN) = {s0.max()/s0.min()} ")
    print(f"{intmat_tag1}: Eigen-value ratio (MAX/MIN) = {s1.max()/s1.min()} ")
    print(f"{intmat_tag2}: Eigen-value ratio (MAX/MIN) = {s2.max()/s2.min()} ")
    j_noll_vector = np.arange(200) + 2
    
    plt.figure()
    plt.clf()
    plt.plot(j_noll_vector, epp0._im_std, label=intmat_tag0)
    plt.plot(j_noll_vector, epp1._im_std, label=intmat_tag1)
    plt.plot(j_noll_vector, epp2._im_std, label=intmat_tag2)
    plt.grid('--', alpha=0.3)
    plt.legend(loc='best')
    plt.ylabel('Slopes (IFS) std [normalized]')
    plt.xlabel(' j Mode index')
    
    plt.figure()
    plt.clf()
    plt.plot(j_noll_vector, pp_vect_in_nm0, label=intmat_tag0)
    plt.plot(j_noll_vector, pp_vect_in_nm1, label=intmat_tag1)
    plt.plot(j_noll_vector, pp_vect_in_nm2, label=intmat_tag2)
    plt.ylabel('Push-Pull [nm] rms wf')
    plt.xlabel('j Mode index')
    plt.grid('--', alpha=0.3)
    plt.legend(loc='best')
    
    epp0._dsm.display_all_slope_maps(size = 45, ncols=5, nrows=40, title=intmat_tag0)
    epp1._dsm.display_all_slope_maps(size = 45, ncols=5, nrows=40, title=intmat_tag1)
    epp2._dsm.display_all_slope_maps(size = 45, ncols=5, nrows=40, title=intmat_tag2)
    
    epp0._dsm._compute_full_slopes_map(size=45, ncols=5, nrows=40)
    epp1._dsm._compute_full_slopes_map(size=45, ncols=5, nrows=40)
    epp2._dsm._compute_full_slopes_map(size=45, ncols=5, nrows=40)
    
    mean_slope0 = epp0._dsm._ifs.mean(axis=-1)
    mean_slope1 = epp1._dsm._ifs.mean(axis=-1)
    mean_slope2 = epp2._dsm._ifs.mean(axis=-1)
    
    plt.figure()
    plt.clf()
    plt.plot(j_noll_vector[2:], mean_slope0[2:], '.-', label = intmat_tag0)
    plt.plot(j_noll_vector[2:], mean_slope1[2:], '.-', label = intmat_tag1)
    plt.plot(j_noll_vector[2:], mean_slope2[2:], '.-', label = intmat_tag2)
    plt.grid('--', alpha=0.3)
    plt.legend(loc='best')
    plt.xlabel('j Mode index')
    plt.ylabel('Slopes (IFS) mean [normalized]')
    
    
    ss0 = (s0 - s0.min())/(s0.max() - s0.min())
    ss1 = (s1 - s1.min())/(s1.max() - s1.min())
    ss2 = (s2 - s2.min())/(s2.max() - s2.min())
    
    plt.figure()
    plt.clf()
    plt.plot(ss0, label = intmat_tag0)
    plt.plot(ss1, label = intmat_tag1)
    plt.plot(ss2, label = intmat_tag2)
    plt.grid('--', alpha=0.3)
    plt.legend(loc='best')
    plt.xlabel('Eigen-Mode index')
    plt.ylabel('Eigen-Value [Normalized]')
    
    fig, axes = plt.subplots(1, 3, sharex = True, sharey = True)
    ima0 = axes[0].imshow(epp0._dsm._full_slopes_map)
    axes[0].set_title(intmat_tag0)
    axes[0].axis('off')
    cbar0 = fig.colorbar(ima0, ax=axes[0], orientation='horizontal')#, fraction=0.046, pad=0.08)
    #cbar0.ax.tick_params(labelsize=6)
    ima1 = axes[1].imshow(epp1._dsm._full_slopes_map)
    axes[1].set_title(intmat_tag1)
    axes[1].axis('off')
    cbar1 = fig.colorbar(ima1, ax=axes[1], orientation='horizontal')#, fraction=0.046, pad=0.08)
    #cbar1.ax.tick_params(labelsize=6)
    ima2 = axes[2].imshow(epp2._dsm._full_slopes_map)
    axes[2].set_title(intmat_tag2)
    axes[2].axis('off')
    cbar2 = fig.colorbar(ima2, ax=axes[2], orientation='horizontal')
    return epp1

def _get_stuff_from_calib(subap_tag, intmat_tag):
    
    pp_vect_in_nm = _get_pp_vector_in_nm(intmat_tag)
    epp = ExperimentalPushPullOptimizer(subap_tag, intmat_tag, pp_vect_in_nm)
    im = epp._dsm._intmat._intmat
    u,s,vh = np.linalg.svd(im)
    ss = (s - s.min())/(s.max() - s.min())
    mean_slope = epp._dsm._ifs.mean(axis=-1)
    return pp_vect_in_nm, epp, im, ss, s,  mean_slope

def compare_calib_iterations_with_subapset_250612_143100():
    
    set_data_dir()
    subap_tag = '250612_143100'
   
    intmat_tag0 = '250613_094300' 
    intmat_tag1 = '250613_102700'
    intmat_tag2 = '250616_103300'
    intmat_tag3 = '250616_113900'
    
    pp_vect_in_nm0, epp0, im0, ss0, s0, mean_slope0 = _get_stuff_from_calib(subap_tag, intmat_tag0)
    pp_vect_in_nm1, epp1, im1, ss1, s1, mean_slope1 = _get_stuff_from_calib(subap_tag, intmat_tag1)
    pp_vect_in_nm2, epp2, im2, ss2, s2, mean_slope2 = _get_stuff_from_calib(subap_tag, intmat_tag2)
    pp_vect_in_nm3, epp3, im3, ss3, s3, mean_slope3 = _get_stuff_from_calib(subap_tag, intmat_tag3)
    
    Nmodes = im0.shape[0]
    j_noll_vector = np.arange(Nmodes) + 2
    
    plt.figure()
    plt.clf()
    plt.plot(j_noll_vector, epp0._im_std, label=intmat_tag0)
    plt.plot(j_noll_vector, epp1._im_std, label=intmat_tag1)
    plt.plot(j_noll_vector, epp2._im_std, label=intmat_tag2)
    plt.plot(j_noll_vector, epp3._im_std, label=intmat_tag3)
    plt.grid('--', alpha=0.3)
    plt.legend(loc='best')
    plt.ylabel('Slopes (IFS) std [normalized]')
    plt.xlabel(' j Mode index')
    
    plt.figure()
    plt.clf()
    plt.plot(j_noll_vector, pp_vect_in_nm0, label=intmat_tag0)
    plt.plot(j_noll_vector, pp_vect_in_nm1, label=intmat_tag1)
    plt.plot(j_noll_vector, pp_vect_in_nm2, label=intmat_tag2)
    plt.plot(j_noll_vector, pp_vect_in_nm3, label=intmat_tag3)
    plt.ylabel('Push-Pull [nm] rms wf')
    plt.xlabel('j Mode index')
    plt.grid('--', alpha=0.3)
    plt.legend(loc='best')
    
    plt.figure()
    plt.clf()
    plt.plot(j_noll_vector[2:], mean_slope0[2:], '.-', label = intmat_tag0)
    plt.plot(j_noll_vector[2:], mean_slope1[2:], '.-', label = intmat_tag1)
    plt.plot(j_noll_vector[2:], mean_slope2[2:], '.-', label = intmat_tag2)
    plt.plot(j_noll_vector[2:], mean_slope3[2:], '.-', label = intmat_tag3)
    plt.grid('--', alpha=0.3)
    plt.legend(loc='best')
    plt.xlabel('j Mode index')
    plt.ylabel('Slopes (IFS) mean [normalized]')
    
    plt.figure()
    plt.clf()
    plt.plot(s0, label = intmat_tag0)
    plt.plot(s1, label = intmat_tag1)
    plt.plot(s2, label = intmat_tag2)
    plt.plot(s3, label = intmat_tag3)
    plt.grid('--', alpha=0.3)
    plt.legend(loc='best')
    plt.xlabel('Eigen-Mode index')
    plt.ylabel('Eigen-Value [Normalized]')
    
    print(f"{intmat_tag0}: Eigen-value ratio (MAX/MIN) = {s0.max()/s0.min()} ")
    print(f"{intmat_tag1}: Eigen-value ratio (MAX/MIN) = {s1.max()/s1.min()} ")
    print(f"{intmat_tag2}: Eigen-value ratio (MAX/MIN) = {s2.max()/s2.min()} ")
    print(f"{intmat_tag3}: Eigen-value ratio (MAX/MIN) = {s2.max()/s3.min()} ")
    
    epp0._dsm._compute_full_slopes_map(size=45, ncols=5, nrows=40)
    epp1._dsm._compute_full_slopes_map(size=45, ncols=5, nrows=40)
    epp2._dsm._compute_full_slopes_map(size=45, ncols=5, nrows=40)
    epp3._dsm._compute_full_slopes_map(size=45, ncols=5, nrows=40)
    
    fig, axes = plt.subplots(1, 4, sharex = True, sharey = True)
    ima0 = axes[0].imshow(epp0._dsm._full_slopes_map)
    axes[0].set_title(intmat_tag0)
    axes[0].axis('off')
    cbar0 = fig.colorbar(ima0, ax=axes[0], orientation='horizontal')#, fraction=0.046, pad=0.08)
    #cbar0.ax.tick_params(labelsize=6)
    ima1 = axes[1].imshow(epp1._dsm._full_slopes_map)
    axes[1].set_title(intmat_tag1)
    axes[1].axis('off')
    cbar1 = fig.colorbar(ima1, ax=axes[1], orientation='horizontal')#, fraction=0.046, pad=0.08)
    #cbar1.ax.tick_params(labelsize=6)
    ima2 = axes[2].imshow(epp2._dsm._full_slopes_map)
    axes[2].set_title(intmat_tag2)
    axes[2].axis('off')
    cbar2 = fig.colorbar(ima2, ax=axes[2], orientation='horizontal')
    ima3 = axes[3].imshow(epp3._dsm._full_slopes_map)
    axes[3].set_title(intmat_tag3)
    axes[3].axis('off')
    cbar3 = fig.colorbar(ima3, ax=axes[3], orientation='horizontal')