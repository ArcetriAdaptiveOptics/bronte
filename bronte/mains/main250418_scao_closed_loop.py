from bronte import startup
from bronte.scao.specula_scao_runner import SpeculaScaoRunner
import numpy as np 

def main(ftag = '250418'):
    '''
    Closing a scao loop on a single NGS on axis with 
    atmospheric parameters equal to '250410_115000' data 
    (in phase_screen folder) 
    '''
    sf = startup.specula_startup()
    
    sf.ATMO_SEED = 1
    sf.TELESCOPE_PUPIL_DIAMETER = 8
    sf._pupil_pixel_pitch = sf.TELESCOPE_PUPIL_DIAMETER/sf._pupil_diameter_in_pixel
    sf.OUTER_SCALE_L0 = 40             # m
    sf.SEEING = 1                  # arcsec
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
    sf.N_MODES_TO_CORRECT = 200
    sf.TIME_STEP_IN_SEC = 0.001
    # TODO: set ad hoc gains 
    gain = -0.3 * np.ones(sf.N_MODES_TO_CORRECT) #* 1e-9
    bad_mode_index = 187 
    gain[bad_mode_index] = -0.3*0.5
    #sf.INT_GAIN = -0.3
    #sf.INT_GAIN = np.zeros(sf.N_MODES_TO_CORRECT)
    sf.INT_GAIN = gain
    T = 10
    Nsteps = int(T/sf.TIME_STEP_IN_SEC)
    
    ssr = SpeculaScaoRunner(sf)
    ssr.run(Nsteps)
    ssr.save_telemetry(ftag)