from bronte.startup import set_data_dir
from bronte.package_data import shframes_folder
from bronte.utils.slopes_covariance_matrix_analyser import SlopesCovariaceMatrixAnalyser
from astropy.io import fits

def main():
    
    set_data_dir()
    #loading frame cubes
    ftag_flat = '250505_151700'
    ftag_tip = '250505_152300'
    ftag = ftag_tip
    fname = shframes_folder() / (ftag + '.fits')
    hdl = fits.open(fname)
    frame_cube = hdl[0].data
    
    subap_tag = '250120_122000'
    scma = SlopesCovariaceMatrixAnalyser(subap_tag)
    scma.set_slopes_from_frame_cube(frame_cube, pix_thr_ratio=0.2, abs_pix_thr=0)
    
    scma.display_rms_slopes()
    scma.display_slope_covariance_matrix()
    
    
    
    
    