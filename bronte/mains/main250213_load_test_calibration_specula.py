import numpy as np 
from bronte.mains.main250207_test_calibration_specula import load_data_from_main250210_z11
from bronte.utils.noll_to_radial_order import from_noll_to_radial_order
import matplotlib.pyplot as plt

def display_results(ftag, amp_in_nm):
    
    modes_zero, rec_mode_list, rec_tag = load_data_from_main250210_z11(ftag)
    
    Nmodes2check = 10
    j_vect = np.arange(2,len(modes_zero)+2)
    n_vect = from_noll_to_radial_order(j_vect)
    amp_per_mode = amp_in_nm/n_vect**2
    
    plt.figure()
    plt.clf()
    plt.title('Calibration '+rec_tag)
    plt.plot(j_vect, rec_mode_list[0] - modes_zero, 'o-', label = 'c2 = %d nm rms'%int(amp_per_mode[0]))
    plt.plot(j_vect, rec_mode_list[1] - modes_zero, 'o-', label = 'c3 = %d nm rms'%int(amp_per_mode[1]))
    plt.plot(j_vect, rec_mode_list[2] - modes_zero, 'o-', label = 'c4 = %d nm rms'%int(amp_per_mode[2]))
    plt.plot(j_vect, rec_mode_list[4] - modes_zero, 'o-', label = 'c6 = %d nm rms'%int(amp_per_mode[4]))
    plt.plot(j_vect, rec_mode_list[6] - modes_zero, 'o-', label = 'c8 = %d nm rms'%int(amp_per_mode[6]))
    plt.plot(j_vect, rec_mode_list[-1] - modes_zero, 'o-', label = 'c11 = %d nm rms'%int(amp_per_mode[-1]))
    plt.ylabel('modal coefficient difference wrt zero [nm rms wf]')
    plt.xlabel('Noll index')
    plt.legend(loc='best')
    plt.grid('--',alpha=0.3)
    
    ca = rec_mode_list.copy()
    for idx in range(Nmodes2check-1):
        ca[idx] -= modes_zero 
    plt.figure()
    plt.clf()
    plt.imshow(ca[:Nmodes2check,:Nmodes2check])
    plt.colorbar(label='nm rms wf')
    plt.xlabel('Reconstructed mode index')
    plt.ylabel('Applied mode index')