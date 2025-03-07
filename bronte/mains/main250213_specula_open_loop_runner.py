import numpy as np 
from bronte.startup import startup
from bronte.wfs.specula_zernike_mode_measurer_new import ZernikeModesMeasurer
import matplotlib.pyplot as plt
from astropy.io import fits
from bronte.package_data import other_folder

def main(zc_vect_in_nm = np.array([2000, 0, 0]), Nsteps=100, ftag = None):
    '''
    Performs an open loop on an inpunt disturb expressed as a vector of
    zernike coefficients. 
    
    Parameters:
    - zc_vect_in_nm (numpy array): vector of zernike coefficients in nm of the disturb
        to inject into the testbench
    - Nsteps (int): Number of steps of the loop
    - ftag (string): file tag of applied and reference commands and of the 
        reconstructed modes to be saved. If None, the data wont be saved
    '''
    bf = startup()
    bf.SH_FRAMES2AVERAGE = 1
    rec_tag = '250218_085300'#'250217_124500'#'250214_164100' #'250211_154500'#
    zmm = ZernikeModesMeasurer(bf, rec_tag, do_plots=False)
    
    ref_cmd = np.zeros(3) 
    zmm.run(ref_cmd)
    ref_modes = 2*zmm._rec.outputs['out_modes'].value
    
    Nmodes = len(ref_modes)
    rec_modes = np.zeros((Nsteps, Nmodes))
    j_vect = np.arange(2,Nmodes+2)
    
    for idx in range(Nsteps):
        
        zmm.run(zc_vect_in_nm)
        rec_modes[idx] = 2*zmm._rec.outputs['out_modes'].value - ref_modes
    
    plt.figure()
    plt.clf()
    plt.semilogy(j_vect, abs(rec_modes).mean(axis=0), '.-')
    plt.grid('--', alpha=0.3)
    plt.xlabel('Noll index')
    plt.ylabel('Mean reconstracted modes'+r'$ |<c_j>|$'+' [nm rms wf]')
    
    plt.figure()
    plt.clf()
    plt.semilogy(j_vect, rec_modes.std(axis=0), '.-')
    plt.grid('--', alpha=0.3)
    plt.xlabel('Noll index')
    plt.ylabel('Error on reconstracted modes [nm rms wf]')
    
    if ftag is not None:
        file_name = other_folder() / (ftag + '.fits')
        hdr = fits.Header()
        hdr['REC_TAG'] = rec_tag
        fits.writeto(file_name, rec_modes, hdr)
        fits.append(file_name, ref_modes)
        fits.append(file_name, zc_vect_in_nm)
        fits.append(file_name, ref_cmd)
    
    return rec_modes

def load_data(ftag):
    file_name = other_folder() / (ftag + '.fits')
    header = fits.getheader(file_name)
    hduList = fits.open(file_name)
    rec_tag = header['REC_TAG']
    rec_modes = hduList[0].data
    ref_modes = hduList[1].data
    zc_vect_in_nm = hduList[2].data
    ref_cmd = hduList[3].data
    return rec_tag, rec_modes, ref_modes, zc_vect_in_nm, ref_cmd

def display_results(ftag):
    
    rec_tag, rec_modes, ref_modes, zc_vect_in_nm, ref_cmd = load_data(ftag)
    
    Nmodes = len(ref_modes)

    j_vect = np.arange(2,Nmodes+2)
    
    plt.figure()
    plt.clf()
    plt.semilogy(j_vect, abs(rec_modes).mean(axis=0), '.-', label=r'$|<c_j>|$')
    plt.grid('--', alpha=0.3)
    plt.xlabel('Noll index')
    plt.ylabel('Mean reconstracted modes [nm rms wf]')
    plt.legend(loc='best')
    
    plt.figure()
    plt.clf()
    plt.semilogy(j_vect, rec_modes.std(axis=0), '.-', label=r'$\sigma^{std}_j$')
    plt.grid('--', alpha=0.3)
    plt.xlabel('Noll index')
    plt.ylabel('Error on reconstracted modes [nm rms wf]')
    plt.legend(loc='best')
    
    err = rec_modes.std(axis=0)
    var = err**2
    tot_err = np.sqrt(var.sum())
    
    zernike_var = (rec_modes.mean(axis=0))**2
    tot_wf_err_rec = np.sqrt(zernike_var.sum())
    print(tot_wf_err_rec)