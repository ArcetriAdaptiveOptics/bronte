from bronte.startup import specula_startup
from bronte.scao.specula_scao_runner import SpeculaScaoRunner
import numpy as np 

def main(ftag='pippo', seeing=0, gain_vector=None, total_time=0.1):
    
    sf = specula_startup()
    
    flat = np.zeros(1920*1152)
    sf.deformable_mirror.set_shape(flat)
    
    #SLM_RADIUS = 545 # set on base factory
    sf.SUBAPS_TAG = '250612_143100'#'250610_140500'
    sf.REC_MAT_TAG = '250616_103300'#'250619_141800'#'250616_103300'#'250617_103800'#'250613_102700'#'250613_111400'#'250611_155700'#'250611_123500' # Nmodes=200
    sf.SLOPE_OFFSET_TAG = None #'250613_140600'#
    sf.LOAD_HUGE_TILT_UNDER_MASK = True
    
    sf.SH_FRAMES2AVERAGE = 1
    sf.SH_PIX_THR = 0
    sf.PIX_THR_RATIO = 0.18
    
    sf.ATMO_SEED = 1
    sf.TELESCOPE_PUPIL_DIAMETER = 8
    sf._pupil_pixel_pitch = sf.TELESCOPE_PUPIL_DIAMETER/sf._pupil_diameter_in_pixel
    sf.OUTER_SCALE_L0 = 40             # m
    sf.SEEING = seeing #1                  # arcsec
    sf.WIND_SPEED_LIST = [5.0]
    sf.WIND_DIR_LIST = [0, 0]
    sf.LAYER_HEIGHTS_LIST = [0.0] # in m
    sf.Cn2_WEIGHTS_LIST = [1.]
    
    #SETTING NGS SOURCE PARAMETERS
    sf.ONAXIS_SOURCE_COORD = [0.0, 0.0]
    sf.ONAXIS_SOURCE_MAG = 8
    sf.ONAXIS_SOURCE_WL_IN_NM = 750
    
    #SETTING AO PARAMETERS
    sf.INT_DELAY = 2
    sf.N_MODES_TO_CORRECT = 200# 25#
    sf.TIME_STEP_IN_SEC = 0.001
    
    if gain_vector is None:
        gain_vector =  -0.1*np.ones(sf.N_MODES_TO_CORRECT)    
    
    sf.INT_GAIN = gain_vector
    
    Nsteps = int(total_time/sf.TIME_STEP_IN_SEC)
    
    ssr = SpeculaScaoRunner(sf) 
    ssr.run(Nsteps)
    ssr.save_telemetry(ftag)
    
def main_cl_s10():
    main('250626_184100', seeing=1)
    
def main_cl_s05():
    main('250626_184200', seeing=0.5)
    
def main_ol_s05():
    main('250626_184300', seeing=0.5, gain_vector=0)
    
def main_cl_s025():
    main('250626_184400', seeing=0.25)
    
def main_cl_50m():
    g=np.zeros(200)
    g[0:50]=-0.1
    main('250626_184500', seeing=0.5, gain_vector=g)
    
    
def main_cl_50m_2():
    g=np.zeros(200)
    g[0:50]=-0.1
    main('250627_114600', seeing=0.5, gain_vector=g)
    
def main_ol_no_trub250627_141900():
    g = 0
    main('250627_142500', seeing = 0, gain_vector=g)

def main_ol_no_trub250804_111500():
    g = 0
    main('250804_111500', seeing = 0, gain_vector=g)
    
def main_cl_no_trub250804_112600():
    g = -0.1
    main('250804_112600', seeing = 0, gain_vector=g)
    
def main_cl_no_turb250804_151500():
    from astropy.io import fits
    fname = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\bronte\\other_data\\250804_151400.fits"
    hduList = fits.open(fname)
    rejection_ratio = hduList[0].data
    index_of_noisy_modes = np.where(rejection_ratio < 2)[0]
    
    g = -0.1*np.ones(200)
    g[index_of_noisy_modes] = 0
    main('250804_151500', seeing = 0, gain_vector=g)