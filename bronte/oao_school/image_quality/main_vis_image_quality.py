import numpy as np 
import matplotlib.pyplot as plt
from bronte.oao_school.image_reduction import main_vis_data_reduction

def main_seeing_estimation():
    
    sl_image, ao_image = main_vis_data_reduction.main()
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
    
    star_roi = sl_image[70:, 65:]
    
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
    axs[1].plot(profile_row_sum, '.-')
    axs[1].title.set_text("sum along rows")
    fig.tight_layout()
    
    # To measure the seeing, we must estimate the FWHM of this profile
    # We can approximately find the half-maximum y coordinate by averaging 
    # the minimum and maximum values
    # and then look for the x coordinates of the pixels whose y 
    # coordinates are closest to the result (to the right and to the left of 
    # the maximum), maybe performing a rough interpolation
    
    # which profile is suitable? 
    profile = profile_row_sum
    
     
    half_max = 0.5 * (profile.max() + profile.min())
    axs[1].hlines(half_max, 0 , len(profile), ls = '--', colors='red', label = 'Half Max' )
    axs[1].legend(loc='best')
    
    # linear interpolation but you can do better than this!
    
    x  = np.arange(0, len(profile))
    x_close2half_max1 = x[112:115]
    x_close2half_max2 = x[159:163]
    y_close2half_max1 = profile[112:115]
    y_close2half_max2 = profile[159:163]
    
    fwhm_x1 = np.interp(half_max, y_close2half_max1, x_close2half_max1)
    fwhm_x2 = np.interp(half_max, y_close2half_max2, x_close2half_max2)
    
    fwhm_in_pixels = fwhm_x2 - fwhm_x1 #161.8-108.3 by eye
    
    Dtel = 1.52
    #orca 
    wl = 630e-9
    dl_fwhm_in_rad = 1.028*wl/Dtel
    dl_fwhm_in_pixels = 1.028*(12.12e-6/1.22)/6.5e-6
    
    pixel_scale_in_arcsec = (dl_fwhm_in_rad*(180/np.pi)*60*60)/dl_fwhm_in_pixels 
    
    seeing = fwhm_in_pixels * pixel_scale_in_arcsec
    
    return seeing
    
    def main_ao_image_quality_estimation():
        sl_image, ao_image = main_vis_data_reduction.main()
        plt.close('all')
    
        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    
        axs[0].imshow(np.log10(sl_image))
        axs[0].title.set_text('Seeing Limited')
        axs[1].imshow(np.log10(ao_image))
        axs[1].title.set_text('AO compensated')
        fig.tight_layout()