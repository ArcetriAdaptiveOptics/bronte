from bronte.startup import measured_calibration_startup
from bronte.calibration.runners.measured_pp_modes_reconstructor import PushPullModesMeasurer
from bronte.package_data import reconstructor_folder
from astropy.io import fits
import numpy as np

def main(ftag = 'pippo'):
    
    calib_factory = measured_calibration_startup()
    recmat_tag = '250619_141800'#'250617_165500'#'250616_103300'#'250617_170100'
    
    calib_tag = '_bronte_calib_config'
    file_name = reconstructor_folder() / (recmat_tag + calib_tag + '.fits')
    config_data = fits.open(file_name)
    pp_vector_in_nm = config_data[0].data
    Nmodes = len(pp_vector_in_nm)#200
   
    #SLM_RADIUS = 545 # set on base factory
    calib_factory.N_MODES_TO_CORRECT = Nmodes
    calib_factory.SUBAPS_TAG = '250612_143100'#250610_140500'
    calib_factory.SLOPE_OFFSET_TAG = '250613_140600'#None
    calib_factory.LOAD_HUGE_TILT_UNDER_MASK  = True
    calib_factory.SH_PIX_THR = 0  # in ADU
    calib_factory.PIX_THR_RATIO = 0.18
    calib_factory.SOURCE_COORD = [0.0, 0.0] # [radius(in_arcsec), angle(in_deg)]
    calib_factory.FOV = 2*calib_factory.SOURCE_COORD[0] # diameter in arcsec
    calib_factory.SH_FRAMES2AVERAGE = 6 #
    
    calib_factory.load_custom_pp_amp_vector(pp_vector_in_nm[:Nmodes])
    
    ppm = PushPullModesMeasurer(calib_factory, recmat_tag)
    return ppm
    ppm.run()
    
    ppm.save(ftag)
    
    #return ppm


def check_reconstructed_modes(ftag):
    import matplotlib.pyplot as plt
    rec_modes, pp_vector_in_nm, slopes_vector, hdr = PushPullModesMeasurer.load(ftag)
    
    Nmodes = len(pp_vector_in_nm)
   
    rec_modes_push_norm = np.zeros((Nmodes,Nmodes))
    rec_modes_pull_norm = np.zeros((Nmodes,Nmodes))
    even_index = np.arange(0, 2*Nmodes, 2)
    odd_index = np.arange(1, 2*Nmodes, 2)
  
    for idx in range(Nmodes):
        
        rec_modes_push_norm[idx,:] = rec_modes[even_index[idx],:]/pp_vector_in_nm[idx]
        rec_modes_pull_norm[idx,:] = rec_modes[odd_index[idx],:]/(-1*pp_vector_in_nm[idx])
        
    plt.figure()
    plt.clf()
    plt.title(f'Push/Pull ({ftag})')
    plt.imshow(0.5*(rec_modes_push_norm+rec_modes_pull_norm))
    plt.colorbar(label='Normalized')
    plt.xlabel('Mode index')
    plt.ylabel('Reconstructed Mode Index')
    plt.figure()
    plt.clf()
    plt.title(f'Reconstructed Push modal commands ({ftag})')
    plt.imshow(rec_modes_push_norm)
    plt.colorbar(label='Normalized')
    plt.xlabel('Mode index')
    plt.ylabel('Reconstructed Mode Index')
    
    plt.figure()
    plt.clf()
    plt.title(f'Reconstructed Pull modal commands ({ftag})')
    plt.imshow(rec_modes_pull_norm)
    plt.colorbar(label='Normalized')
    plt.xlabel('Mode index')
    plt.ylabel('Reconstructed Mode Index')
    
    