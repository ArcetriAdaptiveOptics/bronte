from bronte.startup import measured_calibration_startup
from bronte.calibration.measured_control_matrix_calibration import MeasuredControlMatrixCalibrator
from bronte.utils.noll_to_radial_order import from_noll_to_radial_order
import numpy as np

def main(ftag):
    
    ftag_calib = ftag
    calib_factory = measured_calibration_startup()
    Nmodes = 100
    #SLM_RADIUS = 545 # set on base factory
    calib_factory.N_MODES_TO_CORRECT = Nmodes
    calib_factory.SUBAPS_TAG = '250120_122000'
    calib_factory.SLOPE_OFFSET_TAG = None
    calib_factory.LOAD_HUGE_TILT_UNDER_MASK  = True
    calib_factory.SH_PIX_THR = 0  # in ADU
    calib_factory.PIX_THR_RATIO = 0.2
    calib_factory.SOURCE_COORD = [0.0, 0.0] # [radius(in_arcsec), angle(in_deg)]
    calib_factory.FOV = 2*calib_factory.SOURCE_COORD[0] # diameter in arcsec
    calib_factory.SH_FRAMES2AVERAGE = 6 # the first 3 frames are discarded
    
    pp_amp_in_nm = 2000
    j_noll_vector = np.arange(Nmodes) + 2
    radial_order = from_noll_to_radial_order(j_noll_vector)
    pp_vector_in_nm = pp_amp_in_nm /(radial_order)
    calib_factory.load_custom_pp_amp_vector(pp_vector_in_nm)
    
    mcmc = MeasuredControlMatrixCalibrator(
        calib_factory,
        ftag_calib,
        pp_amp_in_nm = None)
    
    mcmc.run()