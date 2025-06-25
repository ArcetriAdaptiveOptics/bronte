from bronte.startup import specula_startup
from bronte.scao.specula_scao_runner import SpeculaScaoRunner
import numpy as np 

def main(ftag='pippo'):
    
    sf = specula_startup()
    
    flat = np.zeros(1920*1152)
    sf.deformable_mirror.set_shape(flat)
    
    #SLM_RADIUS = 545 # set on base factory
    sf.SUBAPS_TAG = '250612_143100'#'250610_140500'
    sf.REC_MAT_TAG = '250616_103300'#'250619_141800'#'250616_103300'#'250617_103800'#'250613_102700'#'250613_111400'#'250611_155700'#'250611_123500' # Nmodes=200
    sf.SLOPE_OFFSET_TAG = None #'250613_140600'#'250610_150900'
    sf.LOAD_HUGE_TILT_UNDER_MASK = True
    
    sf.SH_FRAMES2AVERAGE = 1
    sf.SH_PIX_THR = 0
    sf.PIX_THR_RATIO = 0.18
    
    sf.ATMO_SEED = 1
    sf.TELESCOPE_PUPIL_DIAMETER = 8
    sf._pupil_pixel_pitch = sf.TELESCOPE_PUPIL_DIAMETER/sf._pupil_diameter_in_pixel
    sf.OUTER_SCALE_L0 = 40             # m
    sf.SEEING = 0#1                  # arcsec
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
    
    gain_vector =  -0.1*np.ones(sf.N_MODES_TO_CORRECT)
    gain_vector[:3] = -0.10
    gain_vector[3:53] = -0.10
    gain_vector[53:100] = -0.08#-0.025
    gain_vector[100:] = -0.01
    
    
    sf.INT_GAIN = gain_vector
    
    T = 0.1#0.05 #10 # in sec
    Nsteps = int(T/sf.TIME_STEP_IN_SEC)
    
    ssr = SpeculaScaoRunner(sf) 
    ssr.run(Nsteps)
    ssr.save_telemetry(ftag)