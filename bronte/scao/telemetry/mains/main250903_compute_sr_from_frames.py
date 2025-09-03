import numpy as np
from astropy.io import fits
from bronte.startup import set_data_dir
from bronte.package_data import other_folder
from bronte.scao.specula_scao_runner import SpeculaScaoRunner
from bronte.utils.raw_strehl_ratio_computer import StrehlRatioComputer
import matplotlib.pyplot as plt


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
    
    
    return measured_sr_vector 
    
def compute_sr_from250903_111200():
    
    ftag = '250903_111200'
    measured_sr_vector = main(ftag)
    

def compute_sr_from250903_155600():
    
    ftag = '2250903_155600'
    measured_sr_vector = main(ftag)