import numpy as np
import matplotlib.pyplot as plt
from tesi_slm.utils import my_tools

def sharpening_psf_histo():
    # sharpening from j=4 to j=11
    # misure ripetute 30 volte
    fpath = "D:\\phd_slm_edo\\old_data\\non_common_path_abs\\230911\\"
    fnflat30 = fpath + "230911spoc_coef_matrix_flat_texp1.0ms_30times_v0.fits"
    fntiltm5um = fpath + "230912spoc_coef_matrix_c2_m5umrms_texp1.0ms_v0.fits"
    fntiltm10um = fpath + "230911spoc_coef_matrix_c2_m10umrms_texp1.0ms_30times_v0.fits"
    fntiltm20um = fpath + "230911spoc_coef_matrix_c2_m20umrms_texp1.0ms_30times_v0.fits"
    
    hh , dd = my_tools.open_fits_file(fnflat30)
    coeff_flat = dd[0].data.mean(axis=1)
    err_coeff_flat = dd[0].data.std(axis=1)
    
    hh , dd = my_tools.open_fits_file(fntiltm5um)
    coeff_z2_m5um = dd[0].data.mean(axis=1)
    err_coeff_z2_m5um = dd[0].data.std(axis=1)
    
    hh , dd = my_tools.open_fits_file(fntiltm10um)
    coeff_z2_m10um = dd[0].data.mean(axis=1)
    err_coeff_z2_m10um = dd[0].data.std(axis=1)
    
    hh , dd = my_tools.open_fits_file(fntiltm20um)
    coeff_z2_m20um = dd[0].data.mean(axis=1)
    err_coeff_z2_m20um = dd[0].data.std(axis=1)
    
    j_index = np.arange(2,11+1)

    plt.figure()
    plt.clf()
    nm = 1e-9
    dj = 0.2
    dw = 0.4
    
    plt.bar(j_index, coeff_flat/nm, width = -dw ,align='edge', color='r',label='c2 = 0 um rms')
    plt.bar(j_index,coeff_z2_m10um/nm, width=dw ,align='edge',color='g',label='c2 = -10 um rms')
    
    plt.errorbar(j_index-dj, coeff_flat/nm, err_coeff_flat/nm, fmt='ko', ecolor='k',linestyle='')
    plt.errorbar(j_index+dj, coeff_z2_m10um/nm, err_coeff_z2_m10um/nm, fmt='ko', ecolor ='k', linestyle='')
    
    plt.xlabel('j index')
    plt.xticks(j_index)
    plt.ylabel('$c_j$'+'' '[nm rms]')
    plt.xlim(4-0.5,11+0.5)
    plt.ylim(-75,75)
    plt.legend(loc='best')
    plt.grid(ls='--',alpha = 0.3)
    
    plt.figure()
    plt.clf()
    dj = 0.2*0.5
    dw = 0.2
    plt.bar(j_index-3*dj, coeff_flat/nm, width = -dw ,align='center', color='r',label='c2 = 0 um rms')
    plt.bar(j_index-dj,coeff_z2_m5um/nm, width=-dw ,align='center',color='c',label='c2 = -5 um rms')
    plt.bar(j_index+dj,coeff_z2_m10um/nm, width=dw ,align='center',color='g',label='c2 = -10 um rms')
    plt.bar(j_index+3*dj,coeff_z2_m20um/nm, width=dw ,align='center',color='m',label='c2 = -20 um rms')
    
    plt.errorbar(j_index-3*dj, coeff_flat/nm, err_coeff_flat/nm, fmt='ko', ecolor='k',linestyle='')
    plt.errorbar(j_index-dj, coeff_z2_m5um/nm, err_coeff_z2_m5um/nm, fmt='ko', ecolor ='k', linestyle='')
    plt.errorbar(j_index+dj, coeff_z2_m10um/nm, err_coeff_z2_m10um/nm, fmt='ko', ecolor ='k', linestyle='')
    plt.errorbar(j_index+3*dj, coeff_z2_m20um/nm, err_coeff_z2_m20um/nm, fmt='ko', ecolor ='k', linestyle='')
    
    plt.xlabel('j index')
    plt.xticks(j_index)
    plt.ylabel('$c_j$'+'' '[nm rms]')
    plt.xlim(4-0.5,11+0.5)
    plt.ylim(-75,75)
    plt.legend(loc='best')
    plt.grid(ls='--',alpha = 0.3)
    
    plt.figure()
    plt.clf()
    dw = 0.5
    dj = 0
    plt.bar(j_index,coeff_z2_m10um/nm, width=dw ,align='center',color='g',label='c2 = -10 um rms')
    plt.errorbar(j_index+dj, coeff_z2_m10um/nm, err_coeff_z2_m10um/nm, fmt='ko', ecolor ='k', linestyle='')
    plt.xlabel('j index')
    plt.xticks(j_index)
    plt.ylabel('$c_j$'+'' '[nm rms]')
    plt.xlim(4-0.5,11+0.5)
    plt.ylim(-75,75)
    plt.legend(loc='best')
    plt.grid(ls='--',alpha = 0.3)
    
    
def sharped_and_unsharped_psf():
    fpath = "D:\\phd_slm_edo\\old_data\\non_common_path_abs\\230911\\"
    fsharp_c2 = fpath + "230911ima_cleansharppsf_c2_m10umrms_texp1.0ms_30times_v0.fits"
    funsharp_c2 = fpath + "230912ima_unsharpedpsf_c2_m10umrms_texp1.0ms_v0.fits"
    #fsharp = fpath + "230911ima_cleansharppsf_flat_texp1.0ms_30times_v0.fits"
    #funsharp = fpath + "230911ima_unsharppsf_flat_texp1.0ms_v0.fits"
    fname = fsharp_c2
    h, psf = my_tools.open_fits_file(fname)
    #h, u_psf_tilt = my_tools.open_fits_file(funsharp_c2)
    #h, s_psf_flat = my_tools.open_fits_file(fsharp)
    #h, u_psf_flat = my_tools.open_fits_file(funsharp) 
    
    ima = psf[0].data
    par = psf[1].data
    err = psf[2].data
    y,x = my_tools.get_index_from_image(ima)
    cut_ima = my_tools.cut_image_around_coord(ima, y, x, 15)
    
    print(fname)
    print(par)
    print(err)
    
    plt.figure()
    plt.clf()
    plt.imshow(cut_ima, cmap='jet')
    plt.colorbar(label='ADU')
    plt.figure()
    plt.clf()
    #cut_ima[cut_ima<=0] = 1e-11
    plt.imshow(np.log(cut_ima + 10), cmap='jet')
    plt.colorbar(label='Log scale')
    