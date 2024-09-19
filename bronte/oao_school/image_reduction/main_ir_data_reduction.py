import numpy as np
import matplotlib.pyplot as plt
from bronte.oao_school.backup_data.load_backup_data import LoadIrBackUpData

def main():
    
    plt.close('all')
    
    load_data = LoadIrBackUpData()
    
    dark_data = load_data.get_camera_dark_data()
    ol_raw_data = load_data.get_open_loop_data()
    cl_raw_dataCube = load_data.get_close_loop_dataCube()
    
    Nframes = cl_raw_dataCube.shape[-1]
    
    raw_long_exp_psf_ol = ol_raw_data
    raw_long_exp_psf_cl = cl_raw_dataCube.sum(axis=-1)
    
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    axs[0].imshow(raw_long_exp_psf_ol, vmin=1000,vmax=2000)
    axs[1].imshow(raw_long_exp_psf_cl)
    fig.suptitle('Long exposure PSF (Raw)')
    axs[0].title.set_text('Open Loop')
    axs[1].title.set_text('Close Loop')
    fig.tight_layout()
    
    master_dark = dark_data
    
    plt.figure()
    plt.clf()
    plt.title("Dark")
    plt.imshow(master_dark, vmin=1000,vmax=2000)
    plt.colorbar(label = 'ADU')
    
    
    # subtracting master dark from raw data
    
    ol_darksub_dataCube = ol_raw_data - master_dark
    cl_darksub_dataCube = np.zeros(cl_raw_dataCube.shape)
    
    for frame in np.arange(Nframes):
        
        #ol_darksub_dataCube[frame, :, :]  = ol_raw_dataCube[frame, :, : ] - master_dark
        cl_darksub_dataCube[:, :, frame] = cl_raw_dataCube[ :, :, frame] - master_dark
    
    #clipping negative values to zero
    ol_darksub_dataCube[ol_darksub_dataCube < 0] = 0
    cl_darksub_dataCube[cl_darksub_dataCube < 0] = 0
    
    #computing masters of open and close loop data
    
    master_open_loop = ol_darksub_dataCube
    master_close_loop = cl_darksub_dataCube.sum(axis = -1)
    
    
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    fig.suptitle('Long exposure PSF (Dark Sub)')
    axs[0].imshow(master_open_loop)
    axs[1].imshow(master_close_loop)
    axs[0].title.set_text('Open Loop')
    axs[1].title.set_text('Close Loop')
    fig.tight_layout()
    
        # reducing image from background
    #roi selection
    bkg_roi1 = master_open_loop[50:100, 50:100]
    bkg_roi2 = master_open_loop[250:300, 50:100]
    bkg_roi3 = master_open_loop[25:75, 225:275]
    bkg_roi4 = master_open_loop[260:300, 260:300]
    
    bkg_ol = np.median([bkg_roi1.mean(), bkg_roi2.mean(), bkg_roi3.mean(), bkg_roi4.mean()]) 
    
    bkg_roi1 = master_close_loop[50:100, 50:100]
    bkg_roi2 = master_close_loop[250:300, 50:100]
    bkg_roi3 = master_close_loop[25:75, 225:275]
    bkg_roi4 = master_close_loop[260:300,260:300]
    
    bkg_cl = np.median([bkg_roi1.mean(), bkg_roi2.mean(), bkg_roi3.mean(), bkg_roi4.mean()]) 
    
    seeing_limited_science_image = master_open_loop - bkg_ol
    ao_corrected_science_image = master_close_loop - bkg_cl
    
    seeing_limited_science_image[seeing_limited_science_image < 0] = 0
    ao_corrected_science_image[ao_corrected_science_image < 0] = 0
    
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    fig.suptitle('Long exposure PSF (Bkg and Dark Sub)')
    axs[0].imshow(np.log10(seeing_limited_science_image))
    axs[1].imshow(np.log10(ao_corrected_science_image))
    axs[0].title.set_text('Open Loop')
    axs[1].title.set_text('Close Loop')
    fig.tight_layout()
    
    return seeing_limited_science_image, ao_corrected_science_image