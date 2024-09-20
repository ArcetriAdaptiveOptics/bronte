import numpy as np 
import matplotlib.pyplot as plt
from bronte.oao_school.image_reduction import main_ir_data_reduction

def main_ao_image_resolution_estimation():
    
    sl_image, ao_image = main_ir_data_reduction.main()
    plt.close('all')
    
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    
    axs[0].imshow(np.log10(sl_image))
    axs[0].title.set_text('Seeing Limited')
    axs[1].imshow(np.log10(ao_image))
    axs[1].title.set_text('AO compensated')
    fig.tight_layout()
    
    Dtel = 1.52 
    #cred3
    wl = 1310e-9
    dl_fwhm_in_rad = 1.028*wl/Dtel
    dl_fwhm_in_pixels = 3.6
    pixel_scale_in_arcsec = (dl_fwhm_in_rad*(180/np.pi)*60*60)/dl_fwhm_in_pixels 
    
    
    # AO resolution/FWHM estimation
    
    #roi selection
    
    star_roi = ao_image[240:296, 335:392]
    
    from astropy.modeling import models, fitting
    
    #fitting gaussian 2D
    
    amp = star_roi.max()
    x0 = np.where(star_roi == star_roi.max())[1][0]
    y0 = np.where(star_roi == star_roi.max())[0][0]
    sigma_x = 3.6/(2*np.sqrt(2*np.log(2)))
    sigma_y = sigma_x
    
    model_gauss = models.Gaussian2D(amp, x0, y0, sigma_x, sigma_y)
    
    fitter = fitting.LevMarLSQFitter()
    
    roi_shape = star_roi.shape
    y, x = np.mgrid[:roi_shape[0], :roi_shape[1]]
    
    best_fit_gauss = fitter(model_gauss, x, y, z = star_roi)
    

    
    fwhm_fit = best_fit_gauss.parameters[3:5]*(2*np.sqrt(2*np.log(2)))
    ao_res_gauss2d = fwhm_fit * pixel_scale_in_arcsec
    
    #fitting moffat 2D
    
    model_moffat = models.Moffat2D(amp, x0, y0, 3.6)
    best_fit_moffat = fitter(model_moffat, x, y, z = star_roi)
    
    fwhm_fit  = best_fit_moffat.parameters[3] * 2
    ao_res_moffat2d = fwhm_fit * pixel_scale_in_arcsec
    
    #SR estimation assuming a gaussaian model
    
    Ntot = star_roi.sum()
    sigma_dl = dl_fwhm_in_pixels/(2*np.sqrt(2*np.log(2)))
    amp_dl = Ntot/(2*np.pi*sigma_dl**2)
    
    sr_gaussian = best_fit_gauss.parameters[0]/amp_dl
    
    return ao_res_gauss2d, ao_res_moffat2d, best_fit_moffat, best_fit_gauss