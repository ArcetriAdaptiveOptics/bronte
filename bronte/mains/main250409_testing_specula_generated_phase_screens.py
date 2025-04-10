from bronte import startup
from bronte.scao.phase_screen.phase_screen_generator import PhaseScreenGenerator


def kolmogorov_like_phase_screens(ftag = '250409_090000'):
    '''
    the aim in to generate phase screens using SPECULA and to 
    verify the compatibility with Kolomogorov theory
    '''
    sf = startup.specula_startup()
    
    #SETTING ATMO PARAMETERS
    sf.ATMO_SEED = 1    
    sf.TELESCOPE_PUPIL_DIAMETER = 8   # m
    sf._pupil_pixel_pitch = sf.TELESCOPE_PUPIL_DIAMETER/sf._pupil_diameter_in_pixel
    sf.OUTER_SCALE_L0 = 99999             # m
    sf.SEEING = 1.                  # arcsec
    sf.WIND_SPEED_LIST = [5.0]
    sf.WIND_DIR_LIST = [0, 0]
    sf.LAYER_HEIGHTS_LIST = [0.0] # in m
    sf.Cn2_WEIGHTS_LIST = [1.]
    
    #SETTING NGS SOURCE PARAMETERS
    sf.ONAXIS_SOURCE_COORD = [0.0, 0.0]
    sf.ONAXIS_SOURCE_MAG = 8
    sf.ONAXIS_SOURCE_WL_IN_NM = 750
    
    #SETTING AO PARAMETERS
    sf.INT_DELAY = 0
    sf.N_MODES_TO_CORRECT = 200
    sf.TIME_STEP_IN_SEC = 0.001
    
    
    psg = PhaseScreenGenerator(sf)
    psg.PROPAGATION_DIR = 'on_axis'
    psg.run(10)
    psg.save(ftag)