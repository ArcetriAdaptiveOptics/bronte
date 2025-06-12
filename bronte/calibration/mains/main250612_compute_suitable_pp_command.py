from bronte.startup import set_data_dir
from bronte.package_data import reconstructor_folder
from bronte.calibration.experimental_push_pull_amplitude_computer import ExperimentalPushPullAmplitudeComputer
from astropy.io import fits

def main():
    set_data_dir()
    
    subap_tag = '250612_143100'
    intmat_tag = '250612_150500'
    calib_tag = '_bronte_calib_config'
    
    file_name = reconstructor_folder() / (intmat_tag + calib_tag + '.fits')
    config_data = fits.open(file_name)
    pp_vect_in_nm = config_data[0].data
    
    epp = ExperimentalPushPullAmplitudeComputer(subap_tag, intmat_tag, pp_vect_in_nm)
    
    epp.display_ifs_std()
    epp.compute_rescaled_pp_vector(target_val = 0.1)
    epp.display_pp_amplitude_vector()
    
    epp._dsm.display_all_slope_maps(size = 45, ncols=5, nrows=40)
    
    return epp
    #epp.save_rescaled_pp_vector(ftag='250612_161500')