from bronte.startup import measured_calibration_startup
#from bronte.package_data import pp_amp_vector_folder
from bronte.calibration.runners.measured_control_matrix_calibration import MeasuredControlMatrixCalibrator
from bronte.utils.noll_to_radial_order import from_noll_to_radial_order
import numpy as np
from bronte.calibration.utils.display_slope_maps_from_intmat import DisplaySlopeMapsFromInteractionMatrix
from bronte.startup import set_data_dir
from bronte.calibration.utils.experimental_push_pull_optimizer import ExperimentalPushPullOptimizer
import matplotlib.pyplot as plt 

def main(ftag, pp_tag = None):
    
    ftag_calib = ftag
    calib_factory = measured_calibration_startup()
    Nmodes = 25#100#200
    #SLM_RADIUS = 545 # set on base factory
    calib_factory.N_MODES_TO_CORRECT = Nmodes
    calib_factory.MODAL_BASE_TYPE = 'zernike'
    calib_factory.SUBAPS_TAG = '250612_143100'#250610_140500'
    calib_factory.SLOPE_OFFSET_TAG = None
    calib_factory.LOAD_HUGE_TILT_UNDER_MASK  = True
    calib_factory.SH_PIX_THR = 0  # in ADU
    calib_factory.PIX_THR_RATIO = 0.18
    calib_factory.SOURCE_COORD = [0.0, 0.0] # [radius(in_arcsec), angle(in_deg)]
    calib_factory.FOV = 2*calib_factory.SOURCE_COORD[0] # diameter in arcsec
    calib_factory.SH_FRAMES2AVERAGE = 6 # the first 3 frames are discarded
    
    if pp_tag is not None:
        pp_vector_in_nm, _ = ExperimentalPushPullOptimizer.load_pp_vector(pp_tag)
        pp_vector_in_nm[:2] = 5000
    else:
        pp_amp_in_nm = 5000 #1000
        j_noll_vector = np.arange(Nmodes) + 2
        radial_order = from_noll_to_radial_order(j_noll_vector)
        pp_vector_in_nm = pp_amp_in_nm /(radial_order)**2
    
    #pp_vector_in_nm = eris_like_calib()
    plt.close('all')
    
    calib_factory.load_custom_pp_amp_vector(pp_vector_in_nm[:Nmodes])
    
    mcmc = MeasuredControlMatrixCalibrator(
        calib_factory,
        ftag_calib,
        pp_amp_in_nm = None)
    
    mcmc.run()
 
    
def eris_like_calib():
    
    set_data_dir()
    subap_tag = '250610_140500'
    intmat_tag = '250611_123500'
    
    j_noll_vector = np.arange(200) + 2
    radial_order = from_noll_to_radial_order(j_noll_vector)
    pp_in_nm = 5000/(radial_order)**2
    
    dsm = DisplaySlopeMapsFromInteractionMatrix(intmat_tag, subap_tag, pp_in_nm)
    im = dsm._intmat._intmat

    imstd = im.std(axis=1)*pp_in_nm
    target_val = 0.1
    new_pp_vect_in_nm = (target_val/imstd)*pp_in_nm
    new_pp_vect_in_nm[:2]=5000
    
    plt.figure()
    plt.clf()
    plt.plot(imstd, label=intmat_tag)
    
    plt.figure()
    plt.clf()
    plt.plot(j_noll_vector, pp_in_nm, label=intmat_tag)
    plt.plot(j_noll_vector, new_pp_vect_in_nm, label='new pp')
    plt.ylabel('Push-Pull [nm] rms wf')
    plt.xlabel('j index')
    plt.grid('--', alpha=0.3)
    plt.legend(loc='best')
    
    new_pp_vect_in_nm[:2]=5000
    return new_pp_vect_in_nm