import numpy as np
from astropy.io import fits
from bronte.startup import set_data_dir
from bronte.package_data import other_folder
from bronte.scao.specula_scao_runner import SpeculaScaoRunner
from bronte.utils.raw_strehl_ratio_computer import StrehlRatioComputer
import matplotlib.pyplot as plt
import matplotlib as mpl

def main(ftag):
    
    psf_cube, hdr = SpeculaScaoRunner.load_acquired_psf_frames(ftag)
    Nframes = psf_cube.shape[0]
    
    sr_pc = StrehlRatioComputer()
    
    measured_sr_vector = np.zeros(Nframes)
    
    for idx in range(Nframes):
        roi_psf = psf_cube[idx, 400:440, 735:765]
        measured_sr_vector[idx] = sr_pc.get_SR_from_image(roi_psf)
    
    plt.figure()
    plt.clf()
    plt.plot(measured_sr_vector)
    plt.grid('--', alpha=0.3)
    plt.xlabel('N Steps')
    plt.ylabel('Measured SR @633nm')
    
    
    return measured_sr_vector, psf_cube 
    
def compute_sr_from250903_111200():
    
    ftag = '250903_111200'
    measured_sr_vector, psf_cube_roi = main(ftag)
    return measured_sr_vector, psf_cube_roi

def compute_sr_from250903_155600():
    
    ftag = '2250903_155600'
    measured_sr_vector, psf_cube_roi  = main(ftag)
    return measured_sr_vector, psf_cube_roi

def _setup_matplotlib_for_thesis():
    mpl.rcParams.update({
        "figure.dpi": 170,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "lines.linewidth": 1.6,
        "errorbar.capsize": 2.5,
    })


def plot_ol_cl_psf():
    
    #_setup_matplotlib_for_thesis()
    measured_sr_vector, psf_cube_roi = compute_sr_from250903_111200()
    
    idx_ol = 0
    idx_cl = 36
    short_exp_ol_psf = psf_cube_roi[idx_ol, 260:600,525:865]
    short_exp_cl_psf = psf_cube_roi[idx_cl, 390:450, 725:785]
    print('SR short exp OL %g'%measured_sr_vector[idx_ol])
    print('SR short exp CL %g'%measured_sr_vector[idx_cl])
    print('SR integ CL %g'%measured_sr_vector[idx_cl:].mean())
    
    long_exp_cl_psf = psf_cube_roi[idx_cl:, 390:450, 725:785].mean(axis=0)
    
    plt.figure()
    plt.title('Short exposure Open-Loop PSF')
    plt.imshow(short_exp_ol_psf, cmap='inferno')
    plt.colorbar(label='ADU')
    
    plt.figure()
    plt.title('Short exposure Closed-Loop PSF')
    plt.imshow(short_exp_cl_psf, cmap='inferno')
    plt.colorbar(label='ADU')
    
    plt.figure()
    plt.title('Integrated Closed-Loop PSF')
    plt.imshow(long_exp_cl_psf, cmap='inferno')
    plt.colorbar(label='ADU')
    
    

    title_size=18
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im0 = axes[0].imshow(short_exp_ol_psf, cmap='inferno')
    axes[0].set_title('Short exposure Open-Loop PSF', size=title_size)
    fig.colorbar(im0, ax=axes[0], label='ADU', fraction=0.046, pad=0.04)
    
    im1 = axes[1].imshow(short_exp_cl_psf, cmap='inferno')
    axes[1].set_title('Short exposure Closed-Loop PSF', size=title_size)
    fig.colorbar(im1, ax=axes[1], label='ADU', fraction=0.046, pad=0.04)
    
    im2 = axes[2].imshow(long_exp_cl_psf, cmap='inferno', vmin=0)
    axes[2].set_title('Average Closed-Loop PSF', size=title_size)
    fig.colorbar(im2, ax=axes[2], label='ADU', fraction=0.046, pad=0.04)
    
    plt.show()