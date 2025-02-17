# import specula
# specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
# from specula import np
import numpy as np 
from bronte.wfs.specula_zernike_mode_measurer_new import ZernikeModesMeasurer
from bronte.utils.noll_to_radial_order import from_noll_to_radial_order
from bronte.package_data import other_folder
import matplotlib.pyplot as plt
from astropy.io import fits
from bronte.startup import startup


def get_modes_from_test_calib(zmm, amp):
    zmm.run(amp)
    return 2*zmm._rec.outputs['out_modes'].value


def do_plot(modes_zero, modes_100, modes_1000, str_title):
    
    jmodes = np.arange(2, len(modes_zero)+2)
    plt.figure()
    plt.clf()
    plt.title(str_title)
    plt.plot(jmodes ,modes_100 - modes_zero, 'o-', label = 'c3 = 100 nm rms wf')
    plt.plot(jmodes, modes_1000 - modes_zero, 'o-', label = 'c3 = 1000 nm rms wf')
    plt.ylabel('modal coefficient difference wrt zero [nm rms wf]')
    plt.xlabel('Noll index')
    plt.legend(loc='best')
    plt.grid('--',alpha=0.3)
           
def main250207(ftag=None):
    '''
    Tests the 3 different reconstructors applying 100 and 1000 nm 
    rms of Tilt and displays the reconstructed modes.
    
    Parameter:
    ftag(string), file tag of the reconstructed modes to be saved. If None,
    the data wont be saved 
    '''
    bf = startup()
    bf.N_MODES_TO_CORRECT = 200
    rec_tag8 = '250211_154500' #pp=8um/n*n
    zmm = ZernikeModesMeasurer(bf, rec_tag8)
    amp = np.zeros(200) # flat of the calibration not the WFC
    modes_zero_pp8 = get_modes_from_test_calib(zmm, amp)
    amp = np.zeros(200) 
    amp[1] = 100 # 100 nm rms of tilt
    modes_100_pp8 = get_modes_from_test_calib(zmm, amp)
    amp = np.zeros(200) 
    amp[1] = 1000 # 1000 nm rms of tilt
    modes_1000_pp8 = get_modes_from_test_calib(zmm, amp)
    do_plot(modes_zero_pp8, modes_100_pp8, modes_1000_pp8, 'Calibration pp=8um/n^2')
    
    
    rec_tag3 = '250211_155500' #pp=3um/n*n
    zmm = ZernikeModesMeasurer(bf, rec_tag3)
    amp = np.zeros(200) # flat of the calibration not the WFC
    modes_zero_pp3 = get_modes_from_test_calib(zmm, amp)
    amp = np.zeros(200) 
    amp[1] = 100 # 100 nm rms of tilt
    modes_100_pp3 = get_modes_from_test_calib(zmm, amp)
    amp = np.zeros(200) 
    amp[1] = 1000 # 1000 nm rms of tilt
    modes_1000_pp3 = get_modes_from_test_calib(zmm, amp)
    do_plot(modes_zero_pp3, modes_100_pp3, modes_1000_pp3, 'Calibration pp=3um/n^2')
    
    rec_tag1 = '250211_160100' #pp=1um/n*n
    zmm = ZernikeModesMeasurer(bf, rec_tag1)
    amp = np.zeros(200) # flat of the calibration not the WFC
    modes_zero_pp1 = get_modes_from_test_calib(zmm, amp)
    amp = np.zeros(200) 
    amp[1] = 100 # 100 nm rms of tilt# 100 nm rms of tilt
    modes_100_pp1 = get_modes_from_test_calib(zmm, amp)
    amp = np.zeros(200) 
    amp[1] = 1000 # 100 nm rms of tilt# 1000 nm rms of tilt
    modes_1000_pp1 = get_modes_from_test_calib(zmm, amp)
    do_plot(modes_zero_pp1, modes_100_pp1, modes_1000_pp1, 'Calibration pp=1um/n^2')
    
    print('\n + Calibration 8um/n^2:')
    print(modes_100_pp8[:3]-modes_zero_pp8[:3])
    print(modes_1000_pp8[:3]-modes_zero_pp8[:3])
    print('\n + Calibration 3um/n^2:')
    print(modes_100_pp3[:3]-modes_zero_pp3[:3])
    print(modes_1000_pp3[:3]-modes_zero_pp3[:3])
    print('\n + Calibration 1um/n^2:')
    print(modes_100_pp1[:3]-modes_zero_pp1[:3])
    print(modes_1000_pp1[:3]-modes_zero_pp1[:3])
    
    if ftag is not None:
        file_name = other_folder() / (ftag + '.fits')
        hdr = fits.Header()
        hdr['REC_TAG1'] = rec_tag1
        hdr['REC_TAG3'] = rec_tag3
        hdr['REC_TAG8'] = rec_tag8
        
        fits.writeto(file_name, modes_zero_pp1, hdr)
        fits.append(file_name, modes_100_pp1)
        fits.append(file_name, modes_1000_pp1)
        
        fits.append(file_name, modes_zero_pp3)
        fits.append(file_name, modes_100_pp3)
        fits.append(file_name, modes_1000_pp3)
        
        fits.append(file_name, modes_zero_pp8)
        fits.append(file_name, modes_100_pp8)
        fits.append(file_name, modes_1000_pp8)
    
def main250210_tt(ftag = None):
    '''
    Tests the reconstructor of 2 modes (tip-tilt) applying the same amplitude
    of the push-pull and displays the reconstructed modes.
    
    Parameter:
    ftag(string), file tag of the reconstructed modes to be saved. If None,
    the data wont be saved 
    '''
    bf = startup()
    bf.N_MODES_TO_CORRECT = 2
    rec_tag = '250211_140400'# # pp=8 um rms for tip-tilt
    zmm = ZernikeModesMeasurer(bf, rec_tag)
    amp = np.zeros(2)
    modes_zero_tt = get_modes_from_test_calib(zmm, amp)
    amp[0]= 8000 # 8000 nm rms of tip
    modes_8000_tip = get_modes_from_test_calib(zmm, amp)
    amp = np.zeros(2)
    amp[1]= 8000 # 8000 nm rms of tilt
    modes_8000_tilt = get_modes_from_test_calib(zmm, amp)
   
    j_vect = np.array([2,3])
    plt.figure()
    plt.clf()
    plt.title('TT Calibration 8um rms/n^2')
    plt.plot(j_vect, modes_8000_tip - modes_zero_tt, 'o-', label = 'c2=8 um rms wf')
    plt.plot(j_vect, modes_8000_tilt - modes_zero_tt, 'o-', label = 'c3=8 um rms wf')
    plt.ylabel('modal coefficient difference wrt zero [nm rms wf]')
    plt.xlabel('noll index')
    plt.legend(loc='best')
    plt.grid('--',alpha=0.3)
    
    print('\n + TT Calibration 8um/n^2:')
    print(modes_8000_tip-modes_zero_tt)
    print(modes_8000_tilt-modes_zero_tt)
    
    if ftag is not None:
        file_name = other_folder() / (ftag + '.fits')
        hdr = fits.Header()
        hdr['REC_TAG'] = rec_tag
        fits.writeto(file_name, modes_zero_tt, hdr)
        fits.append(file_name, modes_8000_tip)
        fits.append(file_name, modes_8000_tilt)

def load_data_from_main250210_tt(ftag):
        file_name = other_folder() / (ftag + '.fits')
        header = fits.getheader(file_name)
        hduList = fits.open(file_name)
        rec_tag = header['REC_TAG']
        modes_zero_tt = hduList[0].data
        modes_8000_tip = hduList[1].data
        modes_8000_tilt = hduList[2].data
        return modes_zero_tt, modes_8000_tip, modes_8000_tilt, rec_tag
    
def main250210_z11(amp_in_nm =1000, ftag=None):
    '''
    Tests a reconstructor by applying an amplitude
    that scales with the radial order of the zernike
    and displays the reconstructed modes also in a map(up to z11).
    
    Parameter:
    amp_in_nm(float), amplitude of the command to be applied in nm rms
    ftag(string), file tag of the reconstructed modes to be saved. If None,
    the data wont be saved 
    '''
    bf = startup()
    rec_tag = '250217_124500'#'250211_154500'#'250211_143700' # 10 modes pp=8um rms/n^2
    bf.N_MODES_TO_CORRECT = 10
    pp = amp_in_nm
    Nmodes2check = 10
    rec_mode_list = []
    zmm = ZernikeModesMeasurer(bf, rec_tag)
    amp = np.zeros(Nmodes2check)
    modes_zero = get_modes_from_test_calib(zmm, amp)
    j_vect  = np.arange(2,len(modes_zero)+2)
    n_vect = from_noll_to_radial_order(j_vect)
    pp_per_mode = pp*np.ones(len(j_vect))#/n_vect**2
    
    for idx in range(Nmodes2check):
        amp = np.zeros(Nmodes2check)
        amp[idx] = pp_per_mode[idx] 
        rec_mode = get_modes_from_test_calib(zmm, amp)
        rec_mode_list.append(rec_mode)
        
    plt.figure()
    plt.clf()
    plt.title('Calibration up to Z11 (pp=8um rms/n^2)')
    plt.plot(j_vect, rec_mode_list[0] - modes_zero, 'o-', label = 'c2 = %d nm rms'%int(pp_per_mode[0]))
    plt.plot(j_vect, rec_mode_list[1] - modes_zero, 'o-', label = 'c3 = %d nm rms'%int(pp_per_mode[1]))
    plt.plot(j_vect, rec_mode_list[2] - modes_zero, 'o-', label = 'c4 = %d nm rms'%int(pp_per_mode[2]))
    plt.plot(j_vect, rec_mode_list[4] - modes_zero, 'o-', label = 'c6 = %d nm rms'%int(pp_per_mode[4]))
    plt.plot(j_vect, rec_mode_list[6] - modes_zero, 'o-', label = 'c8 = %d nm rms'%int(pp_per_mode[6]))
    plt.plot(j_vect, rec_mode_list[-1] - modes_zero, 'o-', label = 'c11 = %d nm rms'%int(pp_per_mode[-1]))
    plt.ylabel('modal coefficient difference wrt zero [nm rms wf]')
    plt.xlabel('Noll index')
    plt.legend(loc='best')
    plt.grid('--',alpha=0.3)
    
    ca = np.array(rec_mode_list)
    for idx in range(Nmodes2check):
        ca[idx] -= modes_zero 
    plt.figure()
    plt.clf()
    plt.imshow(ca[:10,:10])
    plt.colorbar(label='nm rms wf')
    plt.xlabel('Reconstructed mode index')
    plt.ylabel('Applied mode index')
    
    print('\n + Calibration up to Z11 (pp=8um/n^2):')
    print('Z1-Tip')
    print(rec_mode_list[0][0] - modes_zero[0])
    print('Z2-Tilt')
    print(rec_mode_list[1][1] - modes_zero[1])
    print('Z4-Tip')
    print(rec_mode_list[2][2] - modes_zero[2])
    print('Z6-Astig')
    print(rec_mode_list[4][4] - modes_zero[4])
    print('Z8-Coma')
    print(rec_mode_list[6][6] - modes_zero[6])
    print('Z11-Sphere')
    print(rec_mode_list[-1][-1] - modes_zero[-1])
    
    if ftag is not None:
        file_name = other_folder() / (ftag + '.fits')
        hdr = fits.Header()
        hdr['REC_TAG'] = rec_tag
        fits.writeto(file_name, modes_zero, hdr)
        for idx in range(Nmodes2check):
            fits.append(file_name, rec_mode_list[idx])

def load_data_from_main250210_z11(ftag):
        file_name = other_folder() / (ftag + '.fits')
        header = fits.getheader(file_name)
        hduList = fits.open(file_name)
        rec_tag = header['REC_TAG']
        modes_zero = hduList[0].data
        
        rec_mode_list = []
        Nmodes2check = 10
        for idx in range(Nmodes2check-1):
            rec_mode_list.append(hduList[idx+1].data)
    
        return modes_zero, np.array(rec_mode_list), rec_tag
