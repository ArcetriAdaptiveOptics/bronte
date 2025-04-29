from bronte.startup import set_data_dir
from bronte.package_data import other_folder
from bronte.utils.load_psf_cross_section_from_zemax import load_position_value_array
import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def main():
    set_data_dir()
    ftag = '250429_101200_hugens_psf_profile_single_lenslet.txt'
    file_name = other_folder() /(ftag)
    data = load_position_value_array(file_name)
    pos_in_um = data[:, 0]
    norm_intensity = data[:, 1]
    d_la= 144e-6
    wl = 633e-9
    f_la = 8.31477e-3

    plt.figure()
    plt.clf()
    plt.plot(pos_in_um, norm_intensity, '-')
    plt.xlabel('Position [um]')
    plt.ylabel('Normalized Intensity')
    plt.xlim(-144, 144)
    plt.grid('--', alpha=0.3)
    
    cs = CubicSpline(norm_intensity[1007:1025], pos_in_um[1007:1025])
    
    FWHM = 2*np.abs(cs(0.5))
    plt.figure()
    plt.clf()
    plt.plot(norm_intensity, '-')
    return FWHM