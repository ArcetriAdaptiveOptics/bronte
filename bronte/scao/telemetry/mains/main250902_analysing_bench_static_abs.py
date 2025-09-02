import numpy as np 
from bronte.scao.telemetry.scao_telemetry_data_analyser import ScaoTelemetryDataAnalyser
from bronte.utils.slopes_covariance_matrix_analyser import SlopesCovariaceMatrixAnalyser
from astropy.io import fits
import matplotlib.pyplot as plt
from bronte.startup import set_data_dir
from bronte.package_data import other_folder

def main(subap_ftag, rec_ftag, ol_ftag, pix_thr_ratio = 0.18):
    '''
    computes the mean and std  of rec coefficients from meas OL (no turb) slopes
    '''
    slope_cube = load_slopes_cube(ol_ftag)
    scma = SlopesCovariaceMatrixAnalyser(subap_ftag)
    scma.set_slopes_from_slopes_cube(slope_cube, pix_thr_ratio, abs_pix_thr=0)
    scma.load_reconstructor(rec_ftag)
    scma.compute_delta_modal_command()
    dcmd_cube_in_nm =  scma.get_delta_modal_command()
    mean_dcmd_in_nm = dcmd_cube_in_nm.mean(axis=0)
    err_dcmd_in_nm = dcmd_cube_in_nm.std(axis=0)
    
    return mean_dcmd_in_nm, err_dcmd_in_nm

def load_slopes_cube(ol_ftag):
    
    stda_ol = ScaoTelemetryDataAnalyser(ol_ftag)
    slopes_cube = stda_ol._slopes_vect # (Nstep,2*Nsubap)
    return slopes_cube

def save_ol_rec_modes(subap_ftag, rec_ftag, ol_ftag, pix_thr_ratio, mean_rec_modes, std_rec_modes):
    
    set_data_dir()
    fname = other_folder() / (ol_ftag + '_ol_noturb_rec_modes_'+ rec_ftag+'.fits')
    
    hdr = fits.Header()
    hdr['SUBA_TAG'] = subap_ftag
    hdr['REC_TAG'] = rec_ftag
    hdr['OL_TAG'] = ol_ftag
    hdr['SH_THR'] = pix_thr_ratio
    
    fits.writeto(fname, mean_rec_modes, hdr)
    fits.append(fname, std_rec_modes)
    
def load_ol_rec_modes(ftag_ol, ftag_rec):
    
    set_data_dir()
    fname = other_folder() / (ftag_ol + '_ol_noturb_rec_modes_'+ ftag_rec +'.fits')
    hdr = fits.getheader(fname)
    hdulist = fits.open(fname)
    mean_rec_modes_in_nm  = hdulist[0].data
    std_rec_modes_in_nm = hdulist[1].data
    return mean_rec_modes_in_nm, std_rec_modes_in_nm, hdr

###
##### mains to compute rec modes from ol telemetry data with no turb
def main250901_100400():
    '''
    This is wrong, the ol ftag refers to OL with turbulence
    '''
    subap_ftag = '250612_143100'
    rec_ftag_kl = '250808_144900' #kl
    rec_ftag_zern = '250616_103300'#zern
    ol_ftag = '250901_100400'
    pix_thr_ratio = 0.18
    
    mean_rec_modes_in_nm, std_rec_modes_in_nm = main(subap_ftag, rec_ftag_kl, ol_ftag, pix_thr_ratio) 
    save_ol_rec_modes(subap_ftag, rec_ftag_kl, ol_ftag, pix_thr_ratio, mean_rec_modes_in_nm, std_rec_modes_in_nm)
    
    mean_rec_modes_in_nm, std_rec_modes_in_nm = main(subap_ftag, rec_ftag_zern, ol_ftag, pix_thr_ratio) 
    save_ol_rec_modes(subap_ftag, rec_ftag_zern, ol_ftag, pix_thr_ratio, mean_rec_modes_in_nm, std_rec_modes_in_nm)
    
def main250829_111600():
    
    subap_ftag = '250612_143100'
    rec_ftag_kl = '250808_144900' #kl
    rec_ftag_zern = '250616_103300'#zern
    ol_ftag = '250829_111600'
    pix_thr_ratio = 0.18
    
    mean_rec_modes_in_nm, std_rec_modes_in_nm = main(subap_ftag, rec_ftag_kl, ol_ftag, pix_thr_ratio) 
    save_ol_rec_modes(subap_ftag, rec_ftag_kl, ol_ftag, pix_thr_ratio, mean_rec_modes_in_nm, std_rec_modes_in_nm)
    
    mean_rec_modes_in_nm, std_rec_modes_in_nm = main(subap_ftag, rec_ftag_zern, ol_ftag, pix_thr_ratio) 
    save_ol_rec_modes(subap_ftag, rec_ftag_zern, ol_ftag, pix_thr_ratio, mean_rec_modes_in_nm, std_rec_modes_in_nm)
    
def main250828_133300():
    
    subap_ftag = '250612_143100'
    rec_ftag_kl = '250808_144900' #kl
    rec_ftag_zern = '250616_103300'#zern
    ol_ftag = '250828_133300'
    pix_thr_ratio = 0.18
    
    mean_rec_modes_in_nm, std_rec_modes_in_nm = main(subap_ftag, rec_ftag_kl, ol_ftag, pix_thr_ratio) 
    save_ol_rec_modes(subap_ftag, rec_ftag_kl, ol_ftag, pix_thr_ratio, mean_rec_modes_in_nm, std_rec_modes_in_nm)
    
    mean_rec_modes_in_nm, std_rec_modes_in_nm = main(subap_ftag, rec_ftag_zern, ol_ftag, pix_thr_ratio) 
    save_ol_rec_modes(subap_ftag, rec_ftag_zern, ol_ftag, pix_thr_ratio, mean_rec_modes_in_nm, std_rec_modes_in_nm)
    
def main250808_161900():
    
    subap_ftag = '250612_143100'
    rec_ftag_kl = '250808_144900' #kl
    rec_ftag_zern = '250616_103300'#zern
    ol_ftag = '250808_161900'
    pix_thr_ratio = 0.18
    
    mean_rec_modes_in_nm, std_rec_modes_in_nm = main(subap_ftag, rec_ftag_kl, ol_ftag, pix_thr_ratio) 
    save_ol_rec_modes(subap_ftag, rec_ftag_kl, ol_ftag, pix_thr_ratio, mean_rec_modes_in_nm, std_rec_modes_in_nm)
    
    mean_rec_modes_in_nm, std_rec_modes_in_nm = main(subap_ftag, rec_ftag_zern, ol_ftag, pix_thr_ratio) 
    save_ol_rec_modes(subap_ftag, rec_ftag_zern, ol_ftag, pix_thr_ratio, mean_rec_modes_in_nm, std_rec_modes_in_nm)
    
def main250808_151100():
    
    subap_ftag = '250612_143100'
    rec_ftag_kl = '250808_144900' #kl
    rec_ftag_zern = '250616_103300'#zern
    ol_ftag = '250808_151100'
    pix_thr_ratio = 0.18
    
    mean_rec_modes_in_nm, std_rec_modes_in_nm = main(subap_ftag, rec_ftag_kl, ol_ftag, pix_thr_ratio) 
    save_ol_rec_modes(subap_ftag, rec_ftag_kl, ol_ftag, pix_thr_ratio, mean_rec_modes_in_nm, std_rec_modes_in_nm)
    
    mean_rec_modes_in_nm, std_rec_modes_in_nm = main(subap_ftag, rec_ftag_zern, ol_ftag, pix_thr_ratio) 
    save_ol_rec_modes(subap_ftag, rec_ftag_zern, ol_ftag, pix_thr_ratio, mean_rec_modes_in_nm, std_rec_modes_in_nm)

def main250902_101600():
    
    subap_ftag = '250612_143100'
    rec_ftag_kl = '250808_144900' #kl
    rec_ftag_zern = '250616_103300'#zern
    ol_ftag = '250902_101600'
    pix_thr_ratio = 0.18
    
    mean_rec_modes_in_nm, std_rec_modes_in_nm = main(subap_ftag, rec_ftag_kl, ol_ftag, pix_thr_ratio) 
    save_ol_rec_modes(subap_ftag, rec_ftag_kl, ol_ftag, pix_thr_ratio, mean_rec_modes_in_nm, std_rec_modes_in_nm)
    
    mean_rec_modes_in_nm, std_rec_modes_in_nm = main(subap_ftag, rec_ftag_zern, ol_ftag, pix_thr_ratio) 
    save_ol_rec_modes(subap_ftag, rec_ftag_zern, ol_ftag, pix_thr_ratio, mean_rec_modes_in_nm, std_rec_modes_in_nm)


###
##### diplay results

def display_results_250902_101200():
    
    rec_ftag_kl = '250808_144900' 
    rec_ftag_zern = '250616_103300'
    ol_ftag_list = ['250808_151100','250808_161900','250828_133300','250829_111600','250902_101600']
    
    mean_rec_modes_in_nm_list = []
    std_rec_modes_in_nm_list = []
    
    meas_error_list = []
    
    ### rec modes from kl measured base
    for ol_ftag in ol_ftag_list:
        mean_rec_modes_in_nm, std_rec_modes_in_nm, _ = load_ol_rec_modes(ol_ftag, rec_ftag_kl)
        meas_error_in_nm = np.sqrt((std_rec_modes_in_nm**2).sum())
        meas_error_list.append(meas_error_in_nm)
        mean_rec_modes_in_nm_list.append(mean_rec_modes_in_nm)
        std_rec_modes_in_nm_list.append(std_rec_modes_in_nm)
    
    plt.figure()
    plt.clf()
    plt.title('KL Base')
    for idx in range(len(ol_ftag_list)):
        plt.plot(mean_rec_modes_in_nm_list[idx], label = ol_ftag_list[idx])
        print(f"Measurment error for ol_tag {ol_ftag_list[idx]} and rec_tag {rec_ftag_kl}: {meas_error_list[idx]:.0f} nm rms wf ")
    
    plt.xlabel('Mode index')
    plt.ylabel('rec modes [nm rms wf]')
    plt.legend(loc='best')
    plt.grid('--', alpha=0.3)
    
    plt.title('KL Base')
    plt.figure()
    plt.clf()
    for idx in range(len(ol_ftag_list)):
        plt.plot(std_rec_modes_in_nm_list[idx], label = ol_ftag_list[idx])
        
    plt.xlabel('Mode index')
    plt.ylabel('std of rec modes [nm rms wf]')
    plt.legend(loc='best')
    plt.grid('--', alpha=0.3)
    
    
    ### rec modes from zernike measured base
    for ol_ftag in ol_ftag_list:
        mean_rec_modes_in_nm, std_rec_modes_in_nm, _ = load_ol_rec_modes(ol_ftag, rec_ftag_zern)
        meas_error_in_nm = np.sqrt((std_rec_modes_in_nm**2).sum())
        meas_error_list.append(meas_error_in_nm)
        mean_rec_modes_in_nm_list.append(mean_rec_modes_in_nm)
        std_rec_modes_in_nm_list.append(std_rec_modes_in_nm)
    
    plt.figure()
    plt.clf()
    plt.title('Zernike Base')
    for idx in range(len(ol_ftag_list)):
        plt.plot(mean_rec_modes_in_nm_list[idx], label = ol_ftag_list[idx])
        print(f"Measurment error for ol_tag {ol_ftag_list[idx]} and rec_tag {rec_ftag_zern}: {meas_error_list[idx]:.0f} nm rms wf ")
    
    plt.xlabel('Mode index')
    plt.ylabel('rec modes [nm rms wf]')
    plt.legend(loc='best')
    plt.grid('--', alpha=0.3)
    
    
    plt.figure()
    plt.clf()
    plt.title('Zernike Base')
    for idx in range(len(ol_ftag_list)):
        plt.plot(std_rec_modes_in_nm_list[idx], label = ol_ftag_list[idx])
        
    plt.xlabel('Mode index')
    plt.ylabel('std of rec modes [nm rms wf]')
    plt.legend(loc='best')
    plt.grid('--', alpha=0.3)
        