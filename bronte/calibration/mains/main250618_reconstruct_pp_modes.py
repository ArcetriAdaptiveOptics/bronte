from bronte.startup import measured_calibration_startup, specula_startup
from bronte.calibration.runners.measured_pp_modes_reconstructor import PushPullModesMeasurer
from bronte.package_data import reconstructor_folder
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

def main(ftag = 'pippo'):
    
    calib_factory = measured_calibration_startup()
    recmat_tag = '250616_103300'#'250619_141800'#'250617_165500'#'250616_103300'#'250617_170100'
    
    calib_tag = '_bronte_calib_config'
    file_name = reconstructor_folder() / (recmat_tag + calib_tag + '.fits')
    config_data = fits.open(file_name)
    pp_vector_in_nm = config_data[0].data
    Nmodes = len(pp_vector_in_nm)#200
   
    #SLM_RADIUS = 545 # set on base factory
    calib_factory.N_MODES_TO_CORRECT = Nmodes
    calib_factory.SUBAPS_TAG = '250612_143100'#250610_140500'
    calib_factory.SLOPE_OFFSET_TAG = '250625_145500'#'250613_140600'#None
    calib_factory.LOAD_HUGE_TILT_UNDER_MASK  = True
    calib_factory.SH_PIX_THR = 0  # in ADU
    calib_factory.PIX_THR_RATIO = 0.18
    calib_factory.SOURCE_COORD = [0.0, 0.0] # [radius(in_arcsec), angle(in_deg)]
    calib_factory.FOV = 2*calib_factory.SOURCE_COORD[0] # diameter in arcsec
    calib_factory.SH_FRAMES2AVERAGE = 6 #
    
    calib_factory.load_custom_pp_amp_vector(pp_vector_in_nm[:Nmodes])
    
    ppm = PushPullModesMeasurer(calib_factory, recmat_tag)
    
    ppm.run()
    
    ppm.save(ftag)
    
    return ppm


def check_reconstructed_modes(ftag):
    
    rec_modes, pp_vector_in_nm, slopes_vector, hdr = PushPullModesMeasurer.load(ftag)
    
    Nmodes = len(pp_vector_in_nm)
   
    rec_modes_push_norm = np.zeros((Nmodes,Nmodes))
    rec_modes_pull_norm = np.zeros((Nmodes,Nmodes))
    
    rec_modes_push = np.zeros((Nmodes,Nmodes))
    rec_modes_pull = np.zeros((Nmodes,Nmodes))
    even_index = np.arange(0, 2*Nmodes, 2)
    odd_index = np.arange(1, 2*Nmodes, 2)
  
    for idx in range(Nmodes):
        
        rec_modes_push_norm[idx,:] = rec_modes[even_index[idx],:]/pp_vector_in_nm[idx]
        rec_modes_pull_norm[idx,:] = rec_modes[odd_index[idx],:]/(-1*pp_vector_in_nm[idx])
        rec_modes_push[idx,:] = rec_modes[even_index[idx],:]
        rec_modes_pull[idx,:] = rec_modes[odd_index[idx],:]
        
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
    
    return rec_modes_push_norm, rec_modes_pull_norm,rec_modes_push, rec_modes_pull
    
    
def comapare_similar_dataset_with_different_slopeoffset():
    
    fdata_with_new_slope_offset = '250626_112700'
    fdata_with_old_slope_offset = '250619_092400'
    
    rec_push_new,_,_,_ = check_reconstructed_modes(fdata_with_new_slope_offset)
    rec_push_old,_,_,_ = check_reconstructed_modes(fdata_with_old_slope_offset)
    
    vmin = np.min((rec_push_new.min(),rec_push_old.min()))
    vmax = np.max((rec_push_new.max(),rec_push_old.max()))
    
    plt.subplots(1, 2, sharex=True, sharey=True)
    plt.subplot(1,2,1)
    plt.imshow(rec_push_new, vmin=vmin, vmax=vmax)
    plt.xlabel('mode index')
    plt.ylabel('Reconstructed mode index')
    plt.title(fdata_with_new_slope_offset)
    plt.colorbar(orientation = 'horizontal', label = 'normalized')
    
    plt.subplot(1,2,2)
    plt.imshow(rec_push_old, vmin=vmin, vmax=vmax)
    plt.xlabel('mode index')
    plt.title(fdata_with_old_slope_offset)
    plt.colorbar(orientation = 'horizontal', label = 'normalized')
    
    plt.figure()
    plt.clf()
    plt.title('Difference (New-Old)')
    plt.imshow(rec_push_new-rec_push_old)
    plt.xlabel('mode index')
    plt.ylabel('Reconstructed mode index')
    plt.colorbar(label='normalized')

def compare_slope_offset(
        rec_tag = '250619_141800',
        subap_tag = '250612_143100'):
    
    slope_offset_tag_old = '250613_140600'
    slope_offset_tag_new = '250625_145500'
    
    sf_old = specula_startup()
    sf_old.SUBAPS_TAG = subap_tag
    sf_old.REC_MAT_TAG = rec_tag
    sf_old.SLOPE_OFFSET_TAG = slope_offset_tag_old
    
    slope_offset_old = sf_old.slope_offset
    rec_mat = sf_old.reconstructor.recmat.recmat
    c_offset_old_in_nm = np.dot(rec_mat, slope_offset_old)
    
    sf_new = specula_startup()
    sf_new.SUBAPS_TAG = subap_tag
    sf_new.REC_MAT_TAG = rec_tag
    sf_new.SLOPE_OFFSET_TAG = slope_offset_tag_new
    
    slope_offset_new = sf_new.slope_offset
    rec_mat = sf_new.reconstructor.recmat.recmat
    c_offset_new_in_nm = np.dot(rec_mat, slope_offset_new)
    
    wf_ref_old = sf_old.slm_rasterizer.zernike_coefficients_to_raster(c_offset_old_in_nm).toNumpyArray()
    wf_ref_new = sf_new.slm_rasterizer.zernike_coefficients_to_raster(c_offset_new_in_nm).toNumpyArray()
    
    
    plt.figure()
    plt.clf()
    plt.plot(slope_offset_new, label = slope_offset_tag_new)
    plt.plot(slope_offset_old, label = slope_offset_tag_old)
    plt.xlabel('2Nsubap index')
    plt.ylabel('Slope Offset [normalized]')
    plt.grid('--', alpha=0.3)
    plt.legend(loc='best')
    
    j_noll_vector = np.arange(0, len(c_offset_old_in_nm))+2
    plt.figure()
    plt.plot(j_noll_vector, c_offset_new_in_nm, '.-',label = r'$c_{offset}=Rs_{offset}$' + f' ({slope_offset_tag_new})')
    plt.plot(j_noll_vector, c_offset_old_in_nm, '.-',label = r'$c_{offset}=Rs_{offset}$' + f' ({slope_offset_tag_old})')
    plt.xlabel('j mode index')
    plt.ylabel('cj nm rms wf')
    plt.grid('--', alpha=0.3)
    plt.legend(loc='best')
    
    vmin=np.min((wf_ref_new.min(),wf_ref_old.min()))
    vmax=np.max((wf_ref_new.max(), wf_ref_old.max()))
    
    plt.subplots(1,2,sharex=True, sharey=True)
    plt.subplot(1,2,1)
    plt.imshow(wf_ref_new, vmin=vmin, vmax=vmax)
    plt.title(slope_offset_tag_new)
    plt.imshow(wf_ref_new)
    plt.colorbar(orientation = 'horizontal', label = 'nm rms wf')
    
    plt.subplot(1,2,2)
    plt.imshow(wf_ref_old, vmin=vmin, vmax=vmax)
    plt.title(slope_offset_tag_old)
    plt.imshow(wf_ref_old)
    plt.colorbar(orientation = 'horizontal', label = 'nm rms wf')
    
    wf_diff = wf_ref_new-wf_ref_old
    plt.figure()
    plt.clf()
    plt.imshow(wf_diff)
    plt.colorbar(label = 'nm rms wf')
    plt.title(f'Difference std %g nm rms'%wf_diff.std())
    
    print(c_offset_new_in_nm[:12])
    print(c_offset_old_in_nm[:12])