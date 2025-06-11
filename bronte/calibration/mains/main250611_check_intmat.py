from bronte.calibration.display_slope_maps_from_intmat import DisplaySlopeMapsFromInteractionMatrix
from bronte.startup import set_data_dir
import numpy as np 
import matplotlib.pyplot as plt
from bronte.utils.noll_to_radial_order import from_noll_to_radial_order
from bronte.calibration.mains import main250609_on_axis_calibratrion

def main():
    '''
    this script is meant to check and validate 
    the measured interaction matrix 250610_170500
    '''
    set_data_dir()
    subap_tag = '250610_140500'
    intmat_tag = '250610_170500'

    j_noll_vector = np.arange(200) + 2
    radial_order = from_noll_to_radial_order(j_noll_vector)
    pp_in_nm = 1000/radial_order
    
    
    dsm = DisplaySlopeMapsFromInteractionMatrix(intmat_tag, subap_tag,  pp_in_nm)
    im = dsm._intmat._intmat
    
    u,s,vh = np.linalg.svd(im)
    plt.figure()
    plt.clf()
    plt.plot(s)
    plt.ylabel('Eigenvalues')
    plt.xlabel('index')
    plt.grid('--', alpha=0.3)
    dsm.display_all_slope_maps()
    
    
    return dsm

def compare_calib():
    
    set_data_dir()
    subap_tag = '250610_140500'
    
    intmat1_tag = '250611_155700'#'250610_170500'
    intmat2_tag = '250611_123500'
    
    j_noll_vector = np.arange(200) + 2
    radial_order = from_noll_to_radial_order(j_noll_vector)
    #pp1_in_nm = 1000/radial_order #'250610_170500'
    pp1_in_nm = main250609_on_axis_calibratrion.eris_like_calib()
    plt.close('all')
    pp2_in_nm = 5000/(radial_order)**2
    pp2_in_nm[:2] = 5000
    
    dsm1 = DisplaySlopeMapsFromInteractionMatrix(intmat1_tag, subap_tag, pp1_in_nm)
    im1 = dsm1._intmat._intmat
    u1,s1,vh1 = np.linalg.svd(im1)
    dsm2 = DisplaySlopeMapsFromInteractionMatrix(intmat2_tag, subap_tag, pp2_in_nm)
    im2 = dsm2._intmat._intmat
    u2,s2,vh2 = np.linalg.svd(im2)
    
    im2std = im2.std(axis=1)*pp2_in_nm
    im1std = im1.std(axis=1)*pp1_in_nm
    
    target_val = 0.1
    
    new_pp_vect_in_nm = (0.1/im2std)*pp2_in_nm
    
    plt.figure()
    plt.clf()
    plt.plot(im2std, label=intmat2_tag)
    plt.figure()
    plt.clf()
    plt.plot(im1std, label=intmat1_tag)

    
    plt.figure()
    plt.clf()
    plt.plot(j_noll_vector, pp1_in_nm, label=intmat1_tag)
    plt.plot(j_noll_vector, pp2_in_nm, label=intmat2_tag)
    plt.ylabel('Push-Pull [nm] rms wf')
    plt.xlabel('j index')
    plt.grid('--', alpha=0.3)
    plt.legend(loc='best')
    
    plt.figure()
    plt.clf()
    plt.plot(s1, label = intmat1_tag)
    plt.plot(s2, label = intmat2_tag)
    plt.ylabel('Eigenvalues')
    plt.xlabel('index')
    plt.grid('--', alpha=0.3)
    plt.legend(loc='best')
    
    dsm1.display_all_slope_maps(size = 45, ncols = 5, nrows = 40,title=intmat1_tag)
    dsm2.display_all_slope_maps(size = 45, ncols = 5, nrows = 40,title=intmat2_tag)
    return new_pp_vect_in_nm