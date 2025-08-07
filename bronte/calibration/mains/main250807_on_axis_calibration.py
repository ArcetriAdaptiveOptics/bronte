from bronte.startup import measured_calibration_startup
#from bronte.package_data import pp_amp_vector_folder
from bronte.calibration.runners.measured_control_matrix_calibration import MeasuredControlMatrixCalibrator
from bronte.utils.noll_to_radial_order import from_noll_to_radial_order
import numpy as np
from bronte.calibration.utils.display_slope_maps_from_intmat import DisplaySlopeMapsFromInteractionMatrix
#from bronte.startup import set_data_dir
from bronte.calibration.utils.experimental_push_pull_optimizer import ExperimentalPushPullOptimizer
#import matplotlib.pyplot as plt 

def main(calib_factory, ftag):
    
    mcmc = MeasuredControlMatrixCalibrator(
    calib_factory,
    ftag,
    pp_amp_in_nm = None)
    
    mcmc.run()
    
# def _get_pp_vector_in_nm(intmat_tag):
#     calib_tag = '_bronte_calib_config'
#     file_name = reconstructor_folder() / (intmat_tag + calib_tag + '.fits')
#     config_data = fits.open(file_name)
#     pp_vect_in_nm = config_data[0].data
#     return pp_vect_in_nm
    

def main250807_142000():
    
    ftag = '250807_142000'
    calib_factory = measured_calibration_startup()
    Nmodes = 4
    
    calib_factory.N_MODES_TO_CORRECT = Nmodes
    calib_factory.MODAL_BASE_TYPE = 'kl'
    calib_factory.KL_MODAL_IFS_TAG = '250807_100700'
    calib_factory.SUBAPS_TAG = '250612_143100'
    calib_factory.SLOPE_OFFSET_TAG = None
    calib_factory.LOAD_HUGE_TILT_UNDER_MASK  = True
    calib_factory.SH_PIX_THR = 0  # in ADU
    calib_factory.PIX_THR_RATIO = 0.18
    calib_factory.SOURCE_COORD = [0.0, 0.0] # [radius(in_arcsec), angle(in_deg)]
    calib_factory.FOV = 2*calib_factory.SOURCE_COORD[0] # diameter in arcsec
    calib_factory.SH_FRAMES2AVERAGE = 6
    
    pp_tag = '250613_102900'
    pp_vector_in_nm, _ = ExperimentalPushPullOptimizer.load_pp_vector(pp_tag)
    pp_vector_in_nm[:2] = 5000
    
    calib_factory.load_custom_pp_amp_vector(pp_vector_in_nm[:Nmodes])
    main(calib_factory, ftag)