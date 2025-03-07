from bronte import startup
from bronte.package_data import shframes_folder, subaperture_set_folder
from bronte.wfs.subaperture_set import ShSubapertureSet
import numpy as np 
from astropy.io import fits
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.modeling import models, fitting

def main():
    
    subap_tag = '250120_122000'
    ftag_meas = '250224_173200'
    ftag_synth = '250225_124000'#'250224_171700'
    
    bronte_subap = ShSubapertureSet.restore(
            subaperture_set_folder() / (subap_tag+'.fits'))
    subap_grid = bronte_subap.subapertures_map()
    meas_data = fits.open(shframes_folder() / (ftag_meas + '.fits'))
    synth_data = fits.open(shframes_folder() / (ftag_synth + '.fits'))
    
    meas_fr = meas_data[0].data
    synth_fr = synth_data[0].data
    
    norm_meas = _get_normalized(meas_fr)
    norm_synth = _get_normalized(synth_fr)
    
    _do_plot(norm_meas, norm_synth)
    
    fr_norm_with_grid = norm_meas - norm_synth + subap_grid
    
    plt.figure()
    plt.clf()
    plt.imshow(fr_norm_with_grid)
    plt.colorbar()

    _display_around_central_subap(fr_norm_with_grid)
    
    yc = 861
    xc = 959
    size = 26*3 + 4
    hsize = int(round(size*0.5))
    
    norm_meas_wgrd = norm_meas + subap_grid
    norm_synth_wgrd = norm_synth + subap_grid
    _do_plot(norm_meas_wgrd[yc-hsize:yc+hsize,xc-hsize:xc+hsize],
            norm_synth_wgrd[yc-hsize:yc+hsize,xc-hsize:xc+hsize])
    
    hsize = 11
    single_subap = norm_synth_wgrd[yc-hsize:yc+hsize,xc-hsize:xc+hsize+2]
    plt.figure()
    plt.clf()
    plt.imshow(single_subap)
    plt.colorbar()
    
    amp = 1
    x0 = 12
    y0 = 10.5
    radius0 = (1.22*(633e-9/144e-6)*8.31477e-3)/5.5e-6
    
    model_airy = models.AiryDisk2D(amp, x0, y0, radius0)
    fitter = fitting.LevMarLSQFitter()
    
    roi_shape = single_subap.shape
    y, x = np.mgrid[:roi_shape[0], :roi_shape[1]]
    best_fit_airy = fitter(model_airy, x, y, z = single_subap)
    
    psf_residual_airy = best_fit_airy(x,y)-single_subap
    
    # plt.figure()
    # plt.clf()
    # plt.imshow(psf_residual_airy);
    # plt.colorbar()
    
    sing_meas_subap = norm_meas_wgrd[yc-hsize:yc+hsize,xc-hsize:xc+hsize+2]
    
    plt.figure()
    plt.clf()
    plt.plot(sing_meas_subap[12,:],label='meas')
    plt.plot(single_subap[10,:], label='synth')
    plt.ylabel('Normalized intensity')
    plt.legend(loc='best')
    plt.grid('--',alpha=0.3)
    
    
    print(best_fit_airy.parameters)
    return best_fit_airy
    
    
def _do_plot(meas_fr, synth_fr):
    
    fig, axs = plt.subplots(1, 2, sharex = True,
                                 sharey = True)
        
    axs[0].set_title('Measured')
    im_map_x = axs[0].imshow(meas_fr)
    # Use make_axes_locatable to create a colorbar of the same height
    divider_x = make_axes_locatable(axs[0])
    cax_x = divider_x.append_axes("right", size="5%", pad=0.15)  # Adjust size and padding
    fig.colorbar(im_map_x, cax=cax_x, label='a.u.')
    
    axs[1].set_title('Synthetic')
    im_map_y = axs[1].imshow(synth_fr)
    
    divider_y = make_axes_locatable(axs[1])
    cax_y = divider_y.append_axes("right", size="5%", pad=0.15)
    fig.colorbar(im_map_y, cax=cax_y, label='a.u.')
    fig.subplots_adjust(wspace=0.5)
    
    fig.suptitle(f"Tilt Pull 2 um rms")
    fig.tight_layout()

def _get_normalized(data):
    norm_data = (data - data.min())/(data.max()-data.min())
    return norm_data

def _display_around_central_subap(frame):
    #see main250117_subap_initialiser.py
    yc = 861
    xc = 959
    size = 26*3 + 4
    hsize = int(round(size*0.5))
    plt.figure()
    plt.clf()
    plt.imshow(frame[yc-hsize:yc+hsize,xc-hsize:xc+hsize])
    plt.colorbar()
    plt.title('ROI around central subap')