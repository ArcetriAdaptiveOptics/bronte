from bronte.startup import measured_calibration_startup
from bronte.calibration.measured_control_matrix_calibration import MeasuredControlMatrixCalibrator
import matplotlib.pyplot as plt


def main(ftag='250310_145000', how_many = 10):
    
    #sf = measured_calibration_startup()
    pp_amp_in_nm = 2000
    #sf.PP_AMP_IN_NM = pp_amp_in_nm
    for idx in range(how_many):
        ftag_calib = ftag + f'_{idx}'
        mcmc = MeasuredControlMatrixCalibrator(ftag_calib, pp_amp_in_nm)
        mcmc.run()
        plt.close('all')
    
    