import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
from bronte.oao_school.backup_data.load_backup_data import LoadVisBackupData
#from bronte.oao_school.image_reduction.image_reduction_tools import ImageReduction

def main():
    
    plt.close('all')
    
    load_data = LoadVisBackupData()
    
    dark_dataCube = load_data.get_camera_dark_dataCube()
    ol_raw_dataCube = load_data.get_open_loop_dataCube()
    cl_raw_dataCube = load_data.get_close_loop_dataCube()
    
    Nframes = cl_raw_dataCube.shape[0]
    
    #displaying short exposure psf
    
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    fig.suptitle('Short exposure PSF (Raw)')
    im_ol = axs[0].imshow(ol_raw_dataCube[0])
    im_cl = axs[1].imshow(cl_raw_dataCube[0])
    axs[0].title.set_text("Open Loop")
    axs[1].title.set_text("Close Loop")
    fig.tight_layout()
    
    # #animation
    # def update(frame):
    #     fig.suptitle('Short exposure PSF (Raw), frame: %d'%frame)
    #     im_ol = axs[0].imshow(ol_raw_dataCube[frame])
    #     im_cl = axs[1].imshow(cl_raw_dataCube[frame])
    #     return im_ol, im_cl
    #
    # ani = animation.FuncAnimation(
    #     fig = fig, func = update, frames = Nframes, interval = 500)    
    # plt.show()
    
    #displaying long exposure psf
    
    raw_long_exp_psf_ol = ol_raw_dataCube.sum(axis=0)
    raw_long_exp_psf_cl = cl_raw_dataCube.sum(axis=0)
    
    
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    axs[0].imshow(raw_long_exp_psf_ol)
    axs[1].imshow(raw_long_exp_psf_cl)
    fig.suptitle('Long exposure PSF (Raw)')
    axs[0].title.set_text('Open Loop')
    axs[1].title.set_text('Close Loop')
    fig.tight_layout()
    #fig.colorbar(ol_im, ax = axs[0])
    #fig.colorbar(cl_im, ax = axs[1])
    
    
    #computing master dark
    
    master_dark = np.median(dark_dataCube, axis = 0)
    
    # plt.figure()
    # plt.clf()
    # plt.title("Dark")
    # plt.imshow(master_dark)
    # plt.colorbar(label = 'ADU')
    
    # subtracting master dark from raw data
    
    ol_darksub_dataCube = np.zeros(ol_raw_dataCube.shape)
    cl_darksub_dataCube = np.zeros(cl_raw_dataCube.shape)
    
    for frame in np.arange(Nframes):
        
        ol_darksub_dataCube[frame, :, :]  = ol_raw_dataCube[frame, :, : ] - master_dark
        cl_darksub_dataCube[frame, :, :] = cl_raw_dataCube[frame, :, :] - master_dark
    
    #clipping negative values to zero
    ol_darksub_dataCube[ol_darksub_dataCube < 0] = 0
    cl_darksub_dataCube[cl_darksub_dataCube < 0] = 0
    
    #computing masters of open and close loop data
    
    master_open_loop = ol_darksub_dataCube.sum(axis = 0)
    master_close_loop = cl_darksub_dataCube.sum(axis = 0)
    
    
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
    
    
    #there is a drift on the bkg?
    plt.figure()
    plt.plot(np.arange(0,300),master_open_loop[10:30,:].sum(axis=0))
    
    plt.figure()
    plt.plot(np.arange(0,300),master_open_loop[-15:-1,:].sum(axis=0))
    
    plt.figure()
    plt.plot(np.arange(0,300),master_open_loop[:, 10:30].sum(axis=1))
    
    plt.plot(np.arange(0,300),master_open_loop[:, 40:60].sum(axis=1))
    
    return seeing_limited_science_image, ao_corrected_science_image