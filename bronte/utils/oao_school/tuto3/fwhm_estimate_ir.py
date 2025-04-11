import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from bronte.oao_school.utils.package_data import set_data_root_dir, InfraredExampleData
from bronte.oao_school.utils import image_processing


EB_folder = "C:\\Users\\labot\\Desktop\\TP OAO\\oao24\\data"
set_data_root_dir(EB_folder)


def main():
    
    background_image = InfraredExampleData.get_camera_dark_data()
    cl_raw_image_cube = InfraredExampleData.get_close_loop_data_cube()
    ao_image = image_processing.make_master_image(cl_raw_image_cube, background_image)
    
    RAD2ARCSEC=180/np.pi*3600
    pupil_diameter = 1.52 
    wavelength = 1.65e-6 # 1310e-9
    cred3_pixel_in_meter = 15e-6
    f_number =  23
    telescope_focal_length = f_number * pupil_diameter
    # so the size of the pixel in arcsec is 
    pixel_scale_in_arcsec =cred3_pixel_in_meter / telescope_focal_length * RAD2ARCSEC 
    
    # the DL size (=lambda/D) in units of arcsec or pixels are therefore
    dl_size_in_arcsec = wavelength / pupil_diameter * RAD2ARCSEC 
    dl_size_in_pixels = dl_size_in_arcsec / pixel_scale_in_arcsec
    
    print("C-Red3 pixel scale: %g arcsec/pixel" % pixel_scale_in_arcsec)
    print("DL PSF size: %g arcsec" % dl_size_in_arcsec)
    print("DL PSF size: %g pixels" % dl_size_in_pixels)
    
    
    # We work on a roi of the science image with the star image centered in the field
    star_roi = ao_image[240:296, 335:391]
    star_roi_cut_index = (29, slice(15, 45))
    
    #We start simple with a Gaussian fit Note that the actual PSF is quite similar
    #to a Airy function, as the AO correction is high Check PSF fitting residual
    # (PSF-GaussianFit)
    
    amp = star_roi.max()
    x0 = np.where(star_roi == star_roi.max())[1][0]
    y0 = np.where(star_roi == star_roi.max())[0][0]
    sigma_x = 2.5/(2*np.sqrt(2*np.log(2)))
    sigma_y = sigma_x
    
    model_gauss = models.Gaussian2D(amp, x0, y0, sigma_x, sigma_y)
    
    fitter = fitting.LevMarLSQFitter()
    
    roi_shape = star_roi.shape
    y, x = np.mgrid[:roi_shape[0], :roi_shape[1]]
    
    best_fit_gauss = fitter(model_gauss, x, y, z = star_roi)
    psf_residual_gauss= best_fit_gauss(x,y)-star_roi

    display_psf_fit(star_roi, best_fit_gauss, psf_residual_gauss, star_roi_cut_index, 'Gaussian fit')
    
    #Repeat with a Moffat2D model It is capable of better fitting the PSF wing 
    #Moffat is typically used to fit Seeing limited PSF
    
    model_moffat = models.Moffat2D(amp, x0, y0, 2.5)
    best_fit_moffat = fitter(model_moffat, x, y, z = star_roi)
    psf_residual_moffat= best_fit_moffat(x,y)-star_roi

    display_psf_fit(star_roi, best_fit_moffat, psf_residual_moffat, star_roi_cut_index, 'Moffat fit')
    
    # Compare Gaussian and Moffat
    plt.figure()
    plt.plot(best_fit_gauss(x,y)[star_roi_cut_index], label='Gaussian fit')
    plt.plot(best_fit_moffat(x,y)[star_roi_cut_index], label='Moffat fit')
    plt.plot(star_roi[star_roi_cut_index], label='PSF')
    plt.plot(psf_residual_gauss[star_roi_cut_index], label='Gaussian residual')
    plt.plot(psf_residual_moffat[star_roi_cut_index], label='Moffat residual')
    plt.legend()
    
    ###
    # convert sigma to FWHM!
    # average major and minor axis of the Gaussian
    # convert from px to arcsec using the plate scale (given or calibrated)
    #
    # Advanced: PSF ellipticity can tell you something about anisoplanatism, residual astigmatism, wind direction, ...
    ###
    
    fwhm_gaussian_fit_px = np.mean(best_fit_gauss.parameters[3:5]*(2*np.sqrt(2*np.log(2))))
    fwhm_gaussian_fit_arcsec = fwhm_gaussian_fit_px * pixel_scale_in_arcsec
    print('FWHM from Gaussian fit %g arcsec' % (fwhm_gaussian_fit_arcsec))
    
    ###
    # compute FWHM for the Moffat
    # convert from px to arcsec using the plate scale (given or calibrated)
    ###
    
    fwhm_moffat_fit_px  = best_fit_moffat.parameters[3] * 2
    fwhm_moffat_fit_arcsec = fwhm_moffat_fit_px * pixel_scale_in_arcsec
    print('FWHM from Moffat fit %g arcsec' % fwhm_moffat_fit_arcsec)
    print('DL FWHM %g arcsec' % dl_size_in_arcsec) 

def display_psf_fit(star_roi, best_fit, psf_residual, star_roi_cut_index, label_fit):
    
    roi_shape = star_roi.shape
    y, x = np.mgrid[:roi_shape[0], :roi_shape[1]]
    
    plt.figure()
    plt.imshow(star_roi)
    plt.colorbar()
    plt.title('PSF')

    plt.figure()
    plt.imshow(psf_residual)
    plt.colorbar()
    plt.title('PSF fitting residual (PSF-%s) std %g' % (label_fit, psf_residual.std())) 

    plt.figure()
    plt.plot(best_fit(x,y)[star_roi_cut_index], label=label_fit)
    plt.plot(star_roi[star_roi_cut_index], label='PSF')
    plt.plot(psf_residual[star_roi_cut_index], label='PSF fitting residual')
    plt.legend()