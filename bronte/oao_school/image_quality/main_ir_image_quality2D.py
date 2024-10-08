import numpy as np 
import matplotlib.pyplot as plt
from bronte.oao_school.image_reduction import main_ir_data_reduction

def main_ao_image_resolution_estimation():
    
    sl_image, ao_image = main_ir_data_reduction.main()
    plt.close('all')
    
    # fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    #
    # axs[0].imshow(np.log10(sl_image))
    # axs[0].title.set_text('Seeing Limited')
    # axs[1].imshow(np.log10(ao_image))
    # axs[1].title.set_text('AO compensated')
    # fig.tight_layout()
    
    Dtel = 1.52 
    #cred3
    wl = 1.65e-6#1310e-9
    dl_fwhm_in_rad = wl/Dtel
    dl_fwhm_in_pixels = 2.5 # or from fov
    pixel_scale_in_arcsec = (dl_fwhm_in_rad*(180/np.pi)*60*60)/dl_fwhm_in_pixels 
    
    
    # AO resolution/FWHM estimation
    
    #roi selection
    
    star_roi = ao_image[240:296, 335:391]
    
    from astropy.modeling import models, fitting
    
    #fitting gaussian 2D
    
    amp = star_roi.max()
    x0 = np.where(star_roi == star_roi.max())[1][0]
    y0 = np.where(star_roi == star_roi.max())[0][0]
    sigma_x =  dl_fwhm_in_pixels/(2*np.sqrt(2*np.log(2)))
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
    
    
    
    # alpha = np.pi/(28*wl)*15e-6
    # amp_dl  = 3 *alpha* Ntot / 32
    # sr = best_fit_gauss.parameters[0]/amp_dl
    
    #computing SR from fft of pupil
    
    from arte.types.mask import CircularMask
    
    Npix = 200
    pupil = np.zeros((Npix,Npix))
    pupil_radius_in_pix = 100
    #pupil_center = (99,99)
    obs_radius_in_pix = int(np.round(0.33*pupil_radius_in_pix))
    
    cmask = CircularMask(
        frameShape=pupil.shape,
        maskRadius=pupil_radius_in_pix
        )
    
    pupil_mask = cmask.mask()
    
    obs_cmask = CircularMask(
        frameShape=pupil.shape,
        maskRadius=obs_radius_in_pix
        )
    obs_mask = obs_cmask.mask()
    
    pupil_mask[obs_mask == False] = True
    
    phase = np.ma.array(pupil, mask = pupil_mask)
    
    #computing transmitted electric field
    Ui = 1
    transmission_amplitude = 1
    transmission = transmission_amplitude * np.exp(1j * phase)
    Ut = transmission * Ui
    
    Ut.fill_value = 0
    Ut.data[Ut.mask == True] = 0
    
    #padding transmitted electric field
    Npad = 2.5
    padded_frame_size = int(np.round(np.max(phase.shape) * Npad))
    padded_Ut = np.zeros((padded_frame_size, padded_frame_size), dtype=complex)
    padded_Ut[0 : Ut.shape[0], 0 : Ut.shape[1]] = Ut   
    
    #computing angular frequencies
    dxi = 0.5*Dtel/pupil_radius_in_pix
    deta = dxi 
    
    #in rad
    x = np.fft.fftshift(np.fft.fftfreq(padded_Ut.shape[1], dxi)) * wl 
    y = np.fft.fftshift(np.fft.fftfreq(padded_Ut.shape[0], deta)) * wl
    

    #computing psf
    dl_psf = np.abs(np.fft.fftshift(np.fft.fft2(padded_Ut)))**2
    
    total_dl_flux = dl_psf.sum()
    total_meas_flux = star_roi.sum()
    
    dl_psf_norm = dl_psf * total_meas_flux/total_dl_flux
    
    from scipy.ndimage import zoom
    
    pix_scale_dl_in_rad = (wl/Dtel)/Npad
    pix_scale_meas_in_rad = (wl/Dtel)
    
    scale_factor = pix_scale_meas_in_rad/pix_scale_dl_in_rad
    measured_shape = star_roi.shape
    
    dl_shape = dl_psf_norm.shape
    
    zoom_factors = (measured_shape[0] / dl_shape[0] *scale_factor,
                    measured_shape[1] / dl_shape[1] *scale_factor)
    dl_psf_rebinned = zoom(dl_psf_norm, zoom_factors)
    

    sr = best_fit_gauss.parameters[0]/dl_psf_rebinned.max()
    return sr, phase, dl_psf_norm, x, dl_psf_rebinned, star_roi