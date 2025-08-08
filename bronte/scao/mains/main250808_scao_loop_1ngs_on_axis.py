from bronte.startup import specula_startup
from bronte.scao.specula_scao_runner import SpeculaScaoRunner
import numpy as np 

def main(sf, total_time, ftag):
    
    flat = np.zeros(1920*1152)
    sf.deformable_mirror.set_shape(flat)
    
    Nsteps = int(total_time/sf.TIME_STEP_IN_SEC)
    ssr = SpeculaScaoRunner(sf) 
    ssr.run(Nsteps)
    ssr.save_telemetry(ftag)


def main250808_XXXXXX():
    
    sf  = _factory_setup250808_XXXXXX()
    total_time = 0.1 #sec
    ftag = '250808_XXXXXX'
    
    # load control matrices zc or kl
    sf.REC_MAT_TAG = 'XXXX'
    sf.MODAL_BASE_TYPE = 'kl'
    sf.KL_MODAL_IFS_TAG = 'XXXXXXX'
    
    sf.SH_FRAMES2AVERAGE = 1
    
    #opening or closing the loop with/without turb
    sf.SEEING  = 0
    gain_vector =  -0.1*np.ones(sf.N_MODES_TO_CORRECT)
    sf.INT_GAIN = gain_vector 
    
    main(sf, total_time, ftag)

#### Get a factory setup
def _factory_setup250808_XXXXXX():
    '''
    Using kl modes on circular pupil
    correcting 200 modes
    '''
    sf = specula_startup()
    
    #SLM_RADIUS = 545 # set on base factory
    sf.SUBAPS_TAG = '250612_143100'
    sf.REC_MAT_TAG = 'XXXX'
    sf.MODAL_BASE_TYPE = 'kl'
    sf.KL_MODAL_IFS_TAG = 'XXXXXXX'
    sf.SLOPE_OFFSET_TAG = None 
    sf.LOAD_HUGE_TILT_UNDER_MASK = True
    
    sf.SH_FRAMES2AVERAGE = 1
    sf.SH_PIX_THR = 0
    sf.PIX_THR_RATIO = 0.18
    
    sf.ATMO_SEED = 1
    sf.TELESCOPE_PUPIL_DIAMETER = 8
    sf._pupil_pixel_pitch = sf.TELESCOPE_PUPIL_DIAMETER/sf._pupil_diameter_in_pixel
    sf.OUTER_SCALE_L0 = 40             # m
    sf.SEEING = 0.5                  # arcsec
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
    sf.INT_GAIN = gain_vector 
    
    return sf