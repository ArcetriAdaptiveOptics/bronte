from bronte.startup import measured_calibration_startup
from bronte.calibration.measured_pp_modes_reconstructor import PushPullModesMeasurer
from bronte.package_data import reconstructor_folder
from astropy.io import fits

def main(ftag = 'pippo'):
    
    calib_factory = measured_calibration_startup()
    recmat_tag = '250617_170100'
    
    calib_tag = '_bronte_calib_config'
    file_name = reconstructor_folder() / (recmat_tag + calib_tag + '.fits')
    config_data = fits.open(file_name)
    pp_vector_in_nm = config_data[0].data
    Nmodes = len(pp_vector_in_nm)
   
    #SLM_RADIUS = 545 # set on base factory
    calib_factory.N_MODES_TO_CORRECT = Nmodes
    calib_factory.SUBAPS_TAG = '250612_143100'#250610_140500'
    calib_factory.SLOPE_OFFSET_TAG = '250613_140600'#None
    calib_factory.LOAD_HUGE_TILT_UNDER_MASK  = True
    calib_factory.SH_PIX_THR = 0  # in ADU
    calib_factory.PIX_THR_RATIO = 0.18
    calib_factory.SOURCE_COORD = [0.0, 0.0] # [radius(in_arcsec), angle(in_deg)]
    calib_factory.FOV = 2*calib_factory.SOURCE_COORD[0] # diameter in arcsec
    calib_factory.SH_FRAMES2AVERAGE = 6 #
    
    calib_factory.load_custom_pp_amp_vector(pp_vector_in_nm[:Nmodes])
    
    ppm = PushPullModesMeasurer(calib_factory, recmat_tag)
    ppm.run()
    
    ppm.save(ftag)
    
    return ppm