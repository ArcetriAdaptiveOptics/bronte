import numpy as np 
import matplotlib.pyplot as plt
from bronte.oao_school.image_reduction import main_ir_data_reduction

def main_seeing_estimation():
    
    sl_image, ao_image = main_ir_data_reduction.main()
    plt.close('all')
    
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    
    axs[0].imshow(np.log10(sl_image))
    axs[0].title.set_text('Seeing Limited')
    axs[1].imshow(np.log10(ao_image))
    axs[1].title.set_text('AO compensated')
    fig.tight_layout()
    
    # seeing estimation:
    # The commonly accepted measure of the seeing is the full width at 
    # half maximum (FWHM) of the star image profile, expressed in 
    # arcseconds. 
    
    #define a roi enclosing the star image and a little background
    
    star_roi = sl_image[210:370, 270:440]
    
    # We must now draw the profile along any direction in the plane. For 
    # an in-focus star the profile will be reasonably isotropic (but what if its not?), and all the 
    # directions must be equivalent, so that it is convenient to sum along 
    # the rows or the columns
    
    profile_column_sum  = star_roi.sum(axis=0)
    profile_row_sum = star_roi.sum(axis=1)
    
    fig, axs = plt.subplots(1, 2)
    fig.suptitle('Seeing-limited Star Profile')
    axs[0].plot(profile_column_sum)
    axs[0].title.set_text("sum along columns")
    axs[1].plot(profile_row_sum)
    axs[1].title.set_text("sum along rows")
    fig.tight_layout()
    
    # To measure the seeing, we must estimate the FWHM of this profile
    # We can approximately find the half-maximum y coordinate by averaging 
    # the minimum and maximum values
    # and then look for the x coordinates of the pixels whose y 
    # coordinates are closest to the result (to the right and to the left of 
    # the maximum), maybe performing a rough interpolation
    
    profile = profile_row_sum
    
    half_max = 0.5 * (profile.max() + profile.min())
    axs[1].hlines(half_max, 0 , len(profile), ls = '--', colors='red', label = 'Half Max' )
    axs[1].legend(loc='best')

    # linear interpolation but you can do better than this!
    
    x  = np.arange(0, len(profile))
    x_close2half_max1 = x[61:64]
    x_close2half_max2 = x[85:88]
    y_close2half_max1 = profile[61:64]
    y_close2half_max2 = profile[85:88]
    
    fwhm_x1 = np.interp(half_max, y_close2half_max1, x_close2half_max1)
    fwhm_x2 = np.interp(half_max, y_close2half_max2, x_close2half_max2)
    
    fwhm_in_pixels = fwhm_x2 - fwhm_x1 #86.57-62.19 by eye
    
    Dtel = 1.52 
    #cred3
    wl = 1.65e-6#1310e-9
    dl_fwhm_in_rad = 1.028*wl/Dtel
    dl_fwhm_in_pixels = 3.6
    
    pixel_scale_in_arcsec = (dl_fwhm_in_rad*(180/np.pi)*60*60)/dl_fwhm_in_pixels 
    
    seeing = fwhm_in_pixels * pixel_scale_in_arcsec
    
    return seeing


def main_ao_image_resolution_estimation():
    
    sl_image, ao_image = main_ir_data_reduction.main()
    plt.close('all')
    
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    
    axs[0].imshow(np.log10(sl_image))
    axs[0].title.set_text('Seeing Limited')
    axs[1].imshow(np.log10(ao_image))
    axs[1].title.set_text('AO compensated')
    fig.tight_layout()
    
    # AO resolution/FWHM estimation
    
    #1st approach as in seeing estimation
    
    #roi selection
    
    star_roi = ao_image[240:296, 335:392]
    
    profile_column_sum  = star_roi.sum(axis=0)
    profile_row_sum = star_roi.sum(axis=1)
    
    fig, axs = plt.subplots(1, 2)
    fig.suptitle('Seeing-limited Star Profile')
    axs[0].plot(profile_column_sum,'.-')
    axs[0].title.set_text("sum along columns")
    axs[1].plot(profile_row_sum, '.-')
    axs[1].title.set_text("sum along rows")
    fig.tight_layout()
    
    profile = profile_row_sum
    x  = np.arange(0, len(profile))
    x_aroud_max = x[28:31]
    y_aroud_max = profile[28:31]
    
    z = np.polyfit(x_aroud_max, y_aroud_max, 2)
    x_max = -0.5*z[1]/z[0]
    y = np.poly1d(z)
    half_max = y(x_max) * 0.5
    
    axs[1].hlines(half_max, 0 , len(profile), ls = '--', colors = 'm', label='Half Max polyfit')
    x_fit = np.linspace(27.5, 30.5, 19)
    y_fit = y(x_fit)
    axs[1].plot(x_fit, y_fit,'m-', label='polyfit')
    
    
    z[2]-=half_max
    x1, x2 = np.roots(z)
    fwhm_in_pixel = np.abs(x1-x2)
    
    Dtel = 1.52 
    #cred3
    wl = 1.65e-6#1310e-9
    dl_fwhm_in_rad = 1.028*wl/Dtel
    dl_fwhm_in_pixels = 3.6
    
    pixel_scale_in_arcsec = (dl_fwhm_in_rad*(180/np.pi)*60*60)/dl_fwhm_in_pixels 
    
    ao_resolution = fwhm_in_pixel * pixel_scale_in_arcsec
    
    
    #however in this way you may underestimate (or overestimate, it 
    #depends on the selected point used for fitting) the fwhm
    #do better with a gaussian fit
    
    from astropy.modeling import models, fitting
    #from astropy.stats.funcs import gaussian_fwhm_to_sigma
    
    xdata2fit = x[27:32]
    ydata2fit = profile[27:32]
    
    amp = half_max * 2
    x0 = x_max
    gauss_std = fwhm_in_pixel/(2*np.sqrt(2*np.log(2)))
    
    model_gauss = models.Gaussian1D(amp, x0, gauss_std)
    fitter_gauss = fitting.LevMarLSQFitter()
    best_fit_gauss = fitter_gauss(
        model_gauss, xdata2fit, ydata2fit
    )
    
    xfit = np.linspace(27 , 31, 80)
    yfit = models.Gaussian1D(*best_fit_gauss.parameters)
    
    axs[1].plot(xfit, yfit(xfit), 'c-', label='Gaussian fit')
    axs[1].hlines(best_fit_gauss.parameters[0]*0.5, 0 , len(profile), ls = '--', colors = 'b', label='Half Max gauss')
    #axs[1].legend(loc='best')
    
    best_fit_std = best_fit_gauss.parameters[-1]
    
    fwhm_in_pixel_gauss = 2 * best_fit_std * np.sqrt(2*np.log(2))
    ao_res_gauss = fwhm_in_pixel_gauss * pixel_scale_in_arcsec
    
    #lorentz1d
    model_lorentz = models.Lorentz1D(amp, x0, fwhm_in_pixel)
    fitter_lorentz = fitting.LevMarLSQFitter()
    best_fit_lorentz = fitter_lorentz(
        model_lorentz, x[20:40], profile[20:40])
    
    yfit = models.Lorentz1D(*best_fit_lorentz.parameters)
    xfit = np.linspace(20,40,80)
    
    axs[1].plot(xfit, yfit(xfit), 'g-', label='Lorentz fit')
    axs[1].hlines(best_fit_lorentz.parameters[0]*0.5, 0 , len(profile), ls = '--', colors = 'g', label='Half Max Lorentz')
    # axs[1].legend(loc='best')
    ao_res_lor = best_fit_lorentz.parameters[-1]*pixel_scale_in_arcsec
    
    #moffat1d
    model_moffat = models.Moffat1D(amp, x0, fwhm_in_pixel, 1)
    fitter_moffat = fitting.LevMarLSQFitter()
    best_fit_moffat = fitter_moffat(
        model_moffat, x[20:40], profile[20:40])
    
    yfit = models.Moffat1D(*best_fit_moffat.parameters)
    axs[1].plot(xfit, yfit(xfit), 'r-', label='Moffat fit')
    axs[1].hlines(best_fit_moffat.parameters[0]*0.5, 0 , len(profile), ls = '--', colors = 'r', label='Half Max moffat')
    axs[1].legend(loc='best')
    ao_res_moffat = 2 * best_fit_moffat.parameters[2] * pixel_scale_in_arcsec
        
    return ao_resolution, ao_res_gauss, ao_res_lor, ao_res_moffat