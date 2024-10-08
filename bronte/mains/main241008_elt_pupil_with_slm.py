import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from bronte.wfs import slm_rasterizer


def main():
    
    fpath = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\bronte\\elt_pupil\\"
    fname = "EELT480pp0.0803m_obs0.283_spider2023.fits"
    
    header = fits.getheader(fpath+fname)
    hduList = fits.open(fpath+fname)
    
    elt_idl_mask = hduList[1].data.astype(bool)
    elt_mask_shape = elt_idl_mask.shape
    plt.figure()
    plt.clf()
    plt.title("IDL mask")
    plt.imshow(hduList[1].data.astype(bool));
    plt.colorbar()
    elt_mask = elt_idl_mask.copy()
    
    elt_mask[elt_idl_mask == True] = False
    elt_mask[elt_idl_mask == False] = True
    plt.figure()
    plt.clf()
    plt.title("for SLM mask")
    plt.imshow(elt_mask);
    plt.colorbar()
    
    slm_mask = np.ones((1152,1920))
    h_size = int(elt_mask_shape[0] / 2)
    yc = int(1152/2)
    xc = int(1920/2)
    slm_mask[yc-h_size:yc+h_size, xc-h_size:xc+h_size] = elt_mask
    
    fig, axs =plt.subplots() 
    im = axs.imshow(elt_mask);
    axs.set_axis_off()
    
    wf_on_pupil = np.ma.array(np.zeros((1152,1920)), mask = slm_mask)
    sr = slm_rasterizer.SlmRasterizer()
    
    pupil_with_tilt_under_mask = sr.get_slm_raster_from_wf(wf_on_pupil)
    plt.figure()
    plt.clf()
    plt.imshow(pupil_with_tilt_under_mask)
    plt.colorbar()
    
    
    pupil_8bit = (pupil_with_tilt_under_mask/635e-9*255) %256
    print("Max gray = %d"%pupil_8bit.max())
    
    #roi_8bit = pupil_8bit[yc-h_size-50:yc+h_size+50, xc-h_size-50:xc+h_size+50]
    #roe_wf = pupil_with_tilt_under_mask[yc-h_size-50:yc+h_size+50, xc-h_size-50:xc+h_size+50]
    plt.figure()
    plt.clf()
    plt.imshow(pupil_8bit, cmap='grey',vmin=0,vmax=255)
    plt.colorbar()
    
    roi_8bit = pupil_8bit[yc-h_size-50:yc+h_size+50, xc-h_size-50:xc+h_size+50]
    roe_wf = pupil_with_tilt_under_mask[yc-h_size-50:yc+h_size+50, xc-h_size-50:xc+h_size+50]

    fig, axs = plt.subplots( 2, 1, sharex=True, sharey=True)
    
    im0 = axs[0].imshow(roe_wf/1e-6)
    fig.colorbar(im0, ax=axs[0])
    
    im1 = axs[1].imshow(roi_8bit , cmap='grey',vmin=0,vmax=255)
    fig.colorbar(im1, ax=axs[1])
    axs[0].set_axis_off()
    axs[1].set_axis_off()
    
    
