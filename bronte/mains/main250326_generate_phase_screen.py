from bronte import startup
from bronte.scao.phase_screen.phase_screen_generator import PhaseScreenGenerator

def phase_screens_like_250317_134400_telemetry_data(ftag = '250327_160000'):
    '''
    computes and saves the generated phase screens propagated in the 
    direction of the on axis source, with the same configuration of the
    scao telemetry data 250317_134400 (see header)
    '''
    sf = startup.specula_startup()
    sf.N_MODES_TO_CORRECT = 1000
    sf.WIND_SPEED_LIST = [10.0, 5.5]
    sf.TIME_STEP_IN_SEC = 0.001
    psg = PhaseScreenGenerator(sf)
    psg.PROPAGATION_DIR = 'on_axis'
    psg.run(5000, storePhaseScreens = False)
    psg.save(ftag)
    
def phase_screens_like_250311_151100_telemetry_data(ftag = '250326_151700'):
    '''
    computes and saves the generated phase screens propagated in the 
    direction of the on axis source, with the same configuration of the
    scao telemetry data 250317_151100 and 250310_124300 (see header)
    '''
    sf = startup.specula_startup()
    sf.N_MODES_TO_CORRECT = 1000
    sf.WIND_SPEED_LIST = [25.5, 5.5]
    sf.TIME_STEP_IN_SEC = 0.001
    psg = PhaseScreenGenerator(sf)
    psg.PROPAGATION_DIR = 'on_axis'
    psg.run(5000, storePhaseScreens = False)
    psg.save(ftag)
 