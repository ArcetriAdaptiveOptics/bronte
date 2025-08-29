from bronte.startup import specula_startup
from bronte.scao.specula_scao_runner import SpeculaScaoRunner
import numpy as np 

def main(sf, total_time, ftag, do_plots = True, save_disp_wf = False):
    
    flat = np.zeros(1920*1152)
    sf.deformable_mirror.set_shape(flat)
    
    Nsteps = int(total_time/sf.TIME_STEP_IN_SEC)
    ssr = SpeculaScaoRunner(
        scao_factory= sf,
        display_plots = do_plots) 
    ssr.run(Nsteps)
    ssr.save_telemetry(ftag, save_disp_wf)


def main250808_134700():
    '''
    open loop, no turb, with KL modes
    100 step ad dt 1ms
    '''
    sf  = _factory_setup250808_130000()
    total_time = 0.1 #sec
    ftag = '250808_134700'
    
    # load control matrices zc or kl
    sf.REC_MAT_TAG = '250808_123600'
    sf.MODAL_BASE_TYPE = 'kl'
    sf.KL_MODAL_IFS_TAG = '250808_092602'
    
    sf.SH_FRAMES2AVERAGE = 1
    
    #opening or closing the loop with/without turb
    sf.SEEING  = 0
    gain_vector =  0 #0-0.1*np.ones(sf.N_MODES_TO_CORRECT)
    sf.INT_GAIN = gain_vector 
    
    main(sf, total_time, ftag)


def main250808_135900():
    '''
    closed loop, no turb, with KL modes
    100 step ad dt 1ms
    '''
    sf  = _factory_setup250808_130000()
    total_time = 0.1 #sec
    ftag = '250808_135900'
    
    # load control matrices zc or kl
    sf.REC_MAT_TAG = '250808_123600'
    sf.MODAL_BASE_TYPE = 'kl'
    sf.KL_MODAL_IFS_TAG = '250808_092602'
    
    sf.SH_FRAMES2AVERAGE = 6
    
    #opening or closing the loop with/without turb
    sf.SEEING  = 0
    gain_vector =  -0.1*np.ones(sf.N_MODES_TO_CORRECT)
    sf.INT_GAIN = gain_vector 
    
    main(sf, total_time, ftag)

def main250808_140700():
    '''
    open loop, with turb seeing 0.5, with KL modes
    100 step ad dt 1ms
    '''
    sf  = _factory_setup250808_130000()
    total_time = 0.1 #sec
    ftag = '250808_140700'
    
    # load control matrices zc or kl
    sf.REC_MAT_TAG = '250808_123600'
    sf.MODAL_BASE_TYPE = 'kl'
    sf.KL_MODAL_IFS_TAG = '250808_092602'
    
    sf.SH_FRAMES2AVERAGE = 1
    
    #opening or closing the loop with/without turb
    sf.SEEING  = 0.5
    gain_vector =  0 
    sf.INT_GAIN = gain_vector 
    
    main(sf, total_time, ftag)

def main250808_141500():
    '''
    closed loop, with turb seeing 0.5, with KL modes
    100 step ad dt 1ms
    gain =-0.3
    '''
    sf  = _factory_setup250808_130000()
    total_time = 0.1 #sec
    ftag = '250808_141500'
    
    # load control matrices zc or kl
    sf.REC_MAT_TAG = '250808_123600'
    sf.MODAL_BASE_TYPE = 'kl'
    sf.KL_MODAL_IFS_TAG = '250808_092602'
    
    sf.SH_FRAMES2AVERAGE = 6
    
    #opening or closing the loop with/without turb
    sf.SEEING  = 0.5
    gain_vector =  -0.1*np.ones(sf.N_MODES_TO_CORRECT)
    sf.INT_GAIN = gain_vector 
    
    main(sf, total_time, ftag)

def main250808_142500():
    '''
    closed loop, with turb seeing 0.5, with KL modes
    100 step ad dt 1ms
    gain  =-0.3
    '''
    sf  = _factory_setup250808_130000()
    total_time = 0.1 #sec
    ftag = '250808_142500'
    
    # load control matrices zc or kl
    sf.REC_MAT_TAG = '250808_123600'
    sf.MODAL_BASE_TYPE = 'kl'
    sf.KL_MODAL_IFS_TAG = '250808_092602'
    
    sf.SH_FRAMES2AVERAGE = 6
    
    #opening or closing the loop with/without turb
    sf.SEEING  = 0.5
    gain_vector =  -0.3*np.ones(sf.N_MODES_TO_CORRECT)
    sf.INT_GAIN = gain_vector 
    
    main(sf, total_time, ftag)

def main250808_143900():
    '''
    closed loop, with turb seeing 0.5, with KL modes
    1000 step ad dt 1ms
    gain  =-0.3
    '''
    sf  = _factory_setup250808_130000()
    total_time = 1 #sec
    ftag = '250808_143900'
    
    # load control matrices zc or kl
    sf.REC_MAT_TAG = '250808_123600'
    sf.MODAL_BASE_TYPE = 'kl'
    sf.KL_MODAL_IFS_TAG = '250808_092602'
    
    sf.SH_FRAMES2AVERAGE = 6
    
    #opening or closing the loop with/without turb
    sf.SEEING  = 0.5
    gain_vector =  -0.3*np.ones(sf.N_MODES_TO_CORRECT)
    sf.INT_GAIN = gain_vector 
    
    main(sf, total_time, ftag)

def main250808_151100():
    '''
    open loop, no turb, with KL modes
    100 step ad dt 1ms
    L0=25m, r0=0.15m,D=8.2m
    '''
    sf  = _factory_setup250808_130000()
    total_time = 0.1 #sec
    ftag = '250808_151100'
    
    # load control matrices zc or kl
    sf.REC_MAT_TAG = '250808_144900'
    sf.MODAL_BASE_TYPE = 'kl'
    sf.KL_MODAL_IFS_TAG = '250806_170800'
    
    sf.SH_FRAMES2AVERAGE = 1
    
    #opening or closing the loop with/without turb
    sf.SEEING  = 0
    gain_vector =  0 
    sf.INT_GAIN = gain_vector 
    
    main(sf, total_time, ftag)


def main250808_151500():
    '''
    closed loop, no turb, with KL modes
    100 step ad dt 1ms
    L0=25m, r0=0.15m,D=8.2m
    '''
    sf  = _factory_setup250808_130000()
    total_time = 0.1 #sec
    ftag = '250808_151500'
    
    # load control matrices zc or kl
    sf.REC_MAT_TAG = '250808_144900'
    sf.MODAL_BASE_TYPE = 'kl'
    sf.KL_MODAL_IFS_TAG = '250806_170800'
    
    sf.SH_FRAMES2AVERAGE = 6
    
    #opening or closing the loop with/without turb
    sf.SEEING  = 0
    gain_vector =  -0.3*np.ones(sf.N_MODES_TO_CORRECT)
    sf.INT_GAIN = gain_vector 
    
    main(sf, total_time, ftag)
    
def main250808_152700():
    '''
    open loop, with turb, with KL modes
    300 step ad dt 1ms
    L0=25m, r0=0.15m,D=8.2m
    '''
    sf  = _factory_setup250808_130000()
    total_time = 0.3 #sec
    ftag = '250808_152700_tris'
    
    # load control matrices zc or kl
    sf.REC_MAT_TAG = '250808_144900'
    sf.MODAL_BASE_TYPE = 'kl'
    sf.KL_MODAL_IFS_TAG = '250806_170800'
    
    sf.SH_FRAMES2AVERAGE = 1
    
    #opening or closing the loop with/without turb
    sf.TELESCOPE_PUPIL_DIAMETER = 8.2
    sf._pupil_pixel_pitch = sf.TELESCOPE_PUPIL_DIAMETER/sf._pupil_diameter_in_pixel
    sf.OUTER_SCALE_L0 = 25            # m
    wl  = 500e-9
    r0 = 0.15
    seeing = (wl/r0)*(180/np.pi)*60*60
    sf.SEEING = seeing
    gain_vector =  0 
    sf.INT_GAIN = gain_vector 
    
    main(sf, total_time, ftag)
    
def main250808_153900():
    '''
    close loop, with turb, with KL modes
    300 step ad dt 1ms
    L0=25m, r0=0.15m,D=8.2m
    '''
    sf  = _factory_setup250808_130000()
    total_time = 0.3 #sec
    ftag = '250808_153900_tris'
    
    # load control matrices zc or kl
    sf.REC_MAT_TAG = '250808_144900'
    sf.MODAL_BASE_TYPE = 'kl'
    sf.KL_MODAL_IFS_TAG = '250806_170800'
    
    sf.SH_FRAMES2AVERAGE = 6
    
    #opening or closing the loop with/without turb
    sf.TELESCOPE_PUPIL_DIAMETER = 8.2
    sf._pupil_pixel_pitch = sf.TELESCOPE_PUPIL_DIAMETER/sf._pupil_diameter_in_pixel
    sf.OUTER_SCALE_L0 = 25            # m
    wl  = 500e-9
    r0 = 0.15
    seeing = (wl/r0)*(180/np.pi)*60*60
    sf.SEEING = seeing
    gain_vector =  -0.3*np.ones(sf.N_MODES_TO_CORRECT)
    sf.INT_GAIN = gain_vector 
    
    main(sf, total_time, ftag)
    
    
def main250808_155500():
    '''
    open loop, with turb, with Zernike modes
    300 step ad dt 1ms
    L0=25m, r0=0.15m,D=8.2m
    '''
    sf  = _factory_setup250808_130000()
    total_time = 0.3 #sec
    ftag = '250808_155500'
    
    # load control matrices zc or kl
    sf.REC_MAT_TAG = '250616_103300'
    sf.MODAL_BASE_TYPE = 'zernike'
    sf.KL_MODAL_IFS_TAG = None
    
    sf.SH_FRAMES2AVERAGE = 1
    
    #opening or closing the loop with/without turb
    sf.TELESCOPE_PUPIL_DIAMETER = 8.2
    sf.OUTER_SCALE_L0 = 25            # m
    wl  = 500e-9
    r0 = 0.15
    seeing = (wl/r0)*(180/np.pi)*60*60
    sf.SEEING = seeing
    gain_vector =  0 
    sf.INT_GAIN = gain_vector 
    
    main(sf, total_time, ftag)
    
def main250808_160500():
    '''
    close loop, with turb, with Zernike modes
    100 step ad dt 1ms
    L0=25m, r0=0.15m,D=8.2m
    '''
    sf  = _factory_setup250808_130000()
    total_time = 0.3 #sec
    ftag = '250808_160500'
    
    # load control matrices zc or kl
    sf.REC_MAT_TAG = '250616_103300'
    sf.MODAL_BASE_TYPE = 'zernike'
    sf.KL_MODAL_IFS_TAG = None
    
    sf.SH_FRAMES2AVERAGE = 6
    
    #opening or closing the loop with/without turb
    sf.TELESCOPE_PUPIL_DIAMETER = 8.2
    sf.OUTER_SCALE_L0 = 25            # m
    wl  = 500e-9
    r0 = 0.15
    seeing = (wl/r0)*(180/np.pi)*60*60
    sf.SEEING = seeing
    gain_vector =  -0.3*np.ones(sf.N_MODES_TO_CORRECT)
    sf.INT_GAIN = gain_vector 
    
    main(sf, total_time, ftag)
    
def main250808_161900():
    '''
    open loop, no turb, with Zernike modes
    300 step ad dt 1ms
    L0=25m, r0=0.15m,D=8.2m
    '''
    sf  = _factory_setup250808_130000()
    total_time = 0.3 #sec
    ftag = '250808_161900'
    
    # load control matrices zc or kl
    sf.REC_MAT_TAG = '250616_103300'
    sf.MODAL_BASE_TYPE = 'zernike'
    sf.KL_MODAL_IFS_TAG = None
    
    sf.SH_FRAMES2AVERAGE = 1
    
    #opening or closing the loop with/without turb
    sf.TELESCOPE_PUPIL_DIAMETER = 8.2
    sf.OUTER_SCALE_L0 = 25            # m
    seeing = 0
    sf.SEEING = seeing
    gain_vector =  0 
    sf.INT_GAIN = gain_vector 
    
    main(sf, total_time, ftag)
    
def main250808_162500():
    '''
    close loop, with turb, with Zernike modes
    100 step ad dt 1ms
    L0=25m, r0=0.15m,D=8.2m
    '''
    sf  = _factory_setup250808_130000()
    total_time = 0.3 #sec
    ftag = '250808_162500'
    
    # load control matrices zc or kl
    sf.REC_MAT_TAG = '250616_103300'
    sf.MODAL_BASE_TYPE = 'zernike'
    sf.KL_MODAL_IFS_TAG = None
    
    sf.SH_FRAMES2AVERAGE = 6
    
    #opening or closing the loop with/without turb
    sf.TELESCOPE_PUPIL_DIAMETER = 8.2
    sf.OUTER_SCALE_L0 = 25            # m
   
    seeing = 0
    sf.SEEING = seeing
    gain_vector =  -0.3*np.ones(sf.N_MODES_TO_CORRECT)
    sf.INT_GAIN = gain_vector 
    
    main(sf, total_time, ftag)



def main250825_153300():
    '''
    open loop, with turb, with KL modes
    300 step ad dt 1ms
    L0=25m, r0=0.15m,D=8.2m
    '''
    sf  = _factory_setup250808_130000()
    total_time = 0.3 #sec
    ftag = '250825_153300_bis'
    
    # load control matrices zc or kl
    sf.REC_MAT_TAG = '250808_144900'
    sf.MODAL_BASE_TYPE = 'kl'
    sf.KL_MODAL_IFS_TAG = '250806_170800'
    
    sf.SH_FRAMES2AVERAGE = 1
    
    #opening or closing the loop with/without turb
    sf.TELESCOPE_PUPIL_DIAMETER = 8.2
    sf._pupil_pixel_pitch = sf.TELESCOPE_PUPIL_DIAMETER/sf._pupil_diameter_in_pixel
    sf.OUTER_SCALE_L0 = 25            # m
    wl  = 500e-9
    r0 = 0.15
    seeing = (wl/r0)*(180/np.pi)*60*60
    sf.SEEING = seeing
    gain_vector =  0 
    sf.INT_GAIN = gain_vector 
    sf.ONAXIS_SOURCE_WL_IN_NM = 633
    
    main(sf, total_time, ftag)
    
def main250825_154200():
    '''
    close loop, with turb, with KL modes
    300 step ad dt 1ms
    L0=25m, r0=0.15m,D=8.2m
    '''
    sf  = _factory_setup250808_130000()
    total_time = 0.3 #sec
    ftag = '250825_154200_bis'
    
    # load control matrices zc or kl
    sf.REC_MAT_TAG = '250808_144900'
    sf.MODAL_BASE_TYPE = 'kl'
    sf.KL_MODAL_IFS_TAG = '250806_170800'
    
    sf.SH_FRAMES2AVERAGE = 6
    
    #opening or closing the loop with/without turb
    sf.TELESCOPE_PUPIL_DIAMETER = 8.2
    sf._pupil_pixel_pitch = sf.TELESCOPE_PUPIL_DIAMETER/sf._pupil_diameter_in_pixel
    sf.OUTER_SCALE_L0 = 25            # m
    wl  = 500e-9
    r0 = 0.15
    seeing = (wl/r0)*(180/np.pi)*60*60
    sf.SEEING = seeing
    gain_vector =  -0.3*np.ones(sf.N_MODES_TO_CORRECT)
    sf.INT_GAIN = gain_vector 
    sf.ONAXIS_SOURCE_WL_IN_NM = 633
    main(sf, total_time, ftag)

def main250828_133300():
    '''
    open loop, NO turb, with KL modes
    300 step ad dt 1ms
    L0=25m, r0=0.15m,D=8.2m
    '''
    sf  = _factory_setup250808_130000()
    total_time = 0.3 #sec
    ftag = '250828_133300'
    
    # load control matrices zc or kl
    sf.REC_MAT_TAG = '250808_144900'
    sf.MODAL_BASE_TYPE = 'kl'
    sf.KL_MODAL_IFS_TAG = '250806_170800'
    
    sf.SH_FRAMES2AVERAGE = 1
    
    #opening or closing the loop with/without turb
    sf.TELESCOPE_PUPIL_DIAMETER = 8.2
    sf._pupil_pixel_pitch = sf.TELESCOPE_PUPIL_DIAMETER/sf._pupil_diameter_in_pixel
    sf.OUTER_SCALE_L0 = 25            # m

    sf.SEEING = 0
    gain_vector =  0 
    sf.INT_GAIN = gain_vector 
    sf.ONAXIS_SOURCE_WL_IN_NM = 633
    
    main(sf, total_time, ftag)
    
def main250828_135000():
    '''
    close loop, NO turb, with KL modes
    300 step ad dt 1ms
    L0=25m, r0=0.15m,D=8.2m
    '''
    sf  = _factory_setup250808_130000()
    total_time = 0.3 #sec
    ftag = '250828_135000'
    
    # load control matrices zc or kl
    sf.REC_MAT_TAG = '250808_144900'
    sf.MODAL_BASE_TYPE = 'kl'
    sf.KL_MODAL_IFS_TAG = '250806_170800'
    
    sf.SH_FRAMES2AVERAGE = 6
    
    #opening or closing the loop with/without turb
    sf.TELESCOPE_PUPIL_DIAMETER = 8.2
    sf._pupil_pixel_pitch = sf.TELESCOPE_PUPIL_DIAMETER/sf._pupil_diameter_in_pixel
    sf.OUTER_SCALE_L0 = 25            # m

    sf.SEEING = 0
    gain_vector =  -0.3 
    sf.INT_GAIN = gain_vector 
    sf.ONAXIS_SOURCE_WL_IN_NM = 633
    
    main(sf, total_time, ftag)


def main250828_141200():
    '''
    Open loop, with turb, with KL modes
    300 step ad dt 1ms
    L0=25m, r0=0.15m,D=8.2m
    '''
    sf  = _factory_setup250808_130000()
    total_time = 0.3 #sec
    ftag = '250828_141200'
    
    # load control matrices zc or kl
    sf.REC_MAT_TAG = '250808_144900'
    sf.MODAL_BASE_TYPE = 'kl'
    sf.KL_MODAL_IFS_TAG = '250806_170800'
    
    sf.SH_FRAMES2AVERAGE = 1
    
    #opening or closing the loop with/without turb
    sf.TELESCOPE_PUPIL_DIAMETER = 8.2
    sf._pupil_pixel_pitch = sf.TELESCOPE_PUPIL_DIAMETER/sf._pupil_diameter_in_pixel
    sf.OUTER_SCALE_L0 = 25            # m
    wl  = 500e-9
    r0 = 0.15
    seeing = (wl/r0)*(180/np.pi)*60*60
    sf.SEEING = seeing
    gain_vector =  0#-0.3*np.ones(sf.N_MODES_TO_CORRECT)
    sf.INT_GAIN = gain_vector 
    sf.ONAXIS_SOURCE_WL_IN_NM = 633
    main(sf, total_time, ftag)
    
def main250828_142600():
    '''
    Close loop, with turb, with KL modes
    300 step ad dt 1ms
    L0=25m, r0=0.15m,D=8.2m
    '''
    sf  = _factory_setup250808_130000()
    total_time = 0.3 #sec 0.3
    ftag = '250828_142600'
    
    # load control matrices zc or kl
    sf.REC_MAT_TAG = '250808_144900'
    sf.MODAL_BASE_TYPE = 'kl'
    sf.KL_MODAL_IFS_TAG = '250806_170800'
    
    sf.SH_FRAMES2AVERAGE = 6
    
    #opening or closing the loop with/without turb
    sf.TELESCOPE_PUPIL_DIAMETER = 8.2
    sf._pupil_pixel_pitch = sf.TELESCOPE_PUPIL_DIAMETER/sf._pupil_diameter_in_pixel
    sf.OUTER_SCALE_L0 = 25            # m
    wl  = 500e-9
    r0 = 0.15
    seeing = (wl/r0)*(180/np.pi)*60*60
    sf.SEEING = seeing
    #sf.WIND_SPEED_LIST = [0.0] #5m/s
    gain_vector =  -0.3*np.ones(sf.N_MODES_TO_CORRECT)
    sf.INT_GAIN = gain_vector 
    sf.ONAXIS_SOURCE_WL_IN_NM = 633
    main(sf, total_time, ftag, False)
    
def main250829_093000():
    '''
    test to save displayed wf on the SLM on
    Close loop, with turb, with KL modes
    300 step ad dt 1ms
    L0=25m, r0=0.15m,D=8.2m
    '''
    sf  = _factory_setup250808_130000()
    total_time = 0.01 
    ftag = '250829_093000'
    
    # load control matrices zc or kl
    sf.REC_MAT_TAG = '250808_144900'
    sf.MODAL_BASE_TYPE = 'kl'
    sf.KL_MODAL_IFS_TAG = '250806_170800'
    
    sf.SH_FRAMES2AVERAGE = 6
    
    #opening or closing the loop with/without turb
    sf.TELESCOPE_PUPIL_DIAMETER = 8.2
    sf._pupil_pixel_pitch = sf.TELESCOPE_PUPIL_DIAMETER/sf._pupil_diameter_in_pixel
    sf.OUTER_SCALE_L0 = 25            # m
    wl  = 500e-9
    r0 = 0.15
    seeing = (wl/r0)*(180/np.pi)*60*60
    sf.SEEING = seeing
    gain_vector =  -0.3*np.ones(sf.N_MODES_TO_CORRECT)
    sf.INT_GAIN = gain_vector 
    sf.ONAXIS_SOURCE_WL_IN_NM = 633
    main(sf, total_time, ftag, False, True)
    
def main250829_100600():
    '''
    test to save displayed wf on the SLM on
    Open loop, with turb, with KL modes
    300 step ad dt 1ms
    L0=25m, r0=0.15m,D=8.2m
    '''
    sf  = _factory_setup250808_130000()
    total_time = 0.01 
    ftag = '250829_100600'
    
    # load control matrices zc or kl
    sf.REC_MAT_TAG = '250808_144900'
    sf.MODAL_BASE_TYPE = 'kl'
    sf.KL_MODAL_IFS_TAG = '250806_170800'
    
    sf.SH_FRAMES2AVERAGE = 1
    
    #opening or closing the loop with/without turb
    sf.TELESCOPE_PUPIL_DIAMETER = 8.2
    sf._pupil_pixel_pitch = sf.TELESCOPE_PUPIL_DIAMETER/sf._pupil_diameter_in_pixel
    sf.OUTER_SCALE_L0 = 25            # m
    wl  = 500e-9
    r0 = 0.15
    seeing = (wl/r0)*(180/np.pi)*60*60
    sf.SEEING = seeing
    gain_vector =  0
    sf.INT_GAIN = gain_vector 
    sf.ONAXIS_SOURCE_WL_IN_NM = 633
    main(sf, total_time, ftag, False, True)
    
    
def main250829_111600():
    '''
    Open loop, with NO turb, with KL modes
    300 step ad dt 1ms
    L0=25m, r0=0.15m,D=8.2m
    No need to save displayed wf on slm because is always zero
    '''
    sf  = _factory_setup250808_130000()
    total_time = 0.3 
    ftag = '250829_111600'
    
    # load control matrices zc or kl
    sf.REC_MAT_TAG = '250808_144900'
    sf.MODAL_BASE_TYPE = 'kl'
    sf.KL_MODAL_IFS_TAG = '250806_170800'
    
    sf.SH_FRAMES2AVERAGE = 1
    
    #opening or closing the loop with/without turb
    sf.TELESCOPE_PUPIL_DIAMETER = 8.2
    sf._pupil_pixel_pitch = sf.TELESCOPE_PUPIL_DIAMETER/sf._pupil_diameter_in_pixel
    sf.OUTER_SCALE_L0 = 25            # m
    seeing = 0
    sf.SEEING = seeing
    gain_vector =  0
    sf.INT_GAIN = gain_vector 
    sf.ONAXIS_SOURCE_WL_IN_NM = 633
    main(sf, total_time, ftag, False, False)
    
def main250829_112700():
    '''
    close loop, with NO turb, with KL modes
    300 step ad dt 1ms
    L0=25m, r0=0.15m,D=8.2m
    No need to save displayed wf on slm because is the integrated modal command 
    '''
    sf  = _factory_setup250808_130000()
    total_time = 0.3 
    ftag = '250829_112700'
    
    # load control matrices zc or kl
    sf.REC_MAT_TAG = '250808_144900'
    sf.MODAL_BASE_TYPE = 'kl'
    sf.KL_MODAL_IFS_TAG = '250806_170800'
    
    sf.SH_FRAMES2AVERAGE = 6
    
    #opening or closing the loop with/without turb
    sf.TELESCOPE_PUPIL_DIAMETER = 8.2
    sf._pupil_pixel_pitch = sf.TELESCOPE_PUPIL_DIAMETER/sf._pupil_diameter_in_pixel
    sf.OUTER_SCALE_L0 = 25            # m

    seeing = 0
    sf.SEEING = seeing
    gain_vector =  -0.3
    sf.INT_GAIN = gain_vector 
    sf.ONAXIS_SOURCE_WL_IN_NM = 633
    main(sf, total_time, ftag, False, False)


def main250829_114300():
    '''
    Open loop, with turb, with KL modes
    300 step ad dt 1ms
    L0=25m, r0=0.15m,D=8.2m
    Saving displayed wf on the slm
    '''
    sf  = _factory_setup250808_130000()
    total_time = 0.3
    ftag = '250829_114300'
    
    # load control matrices zc or kl
    sf.REC_MAT_TAG = '250808_144900'
    sf.MODAL_BASE_TYPE = 'kl'
    sf.KL_MODAL_IFS_TAG = '250806_170800'
    
    sf.SH_FRAMES2AVERAGE = 6
    
    #opening or closing the loop with/without turb
    sf.TELESCOPE_PUPIL_DIAMETER = 8.2
    sf._pupil_pixel_pitch = sf.TELESCOPE_PUPIL_DIAMETER/sf._pupil_diameter_in_pixel
    sf.OUTER_SCALE_L0 = 25            # m
    wl  = 500e-9
    r0 = 0.15
    seeing = (wl/r0)*(180/np.pi)*60*60
    sf.SEEING = seeing
    gain_vector =  0
    sf.INT_GAIN = gain_vector 
    sf.ONAXIS_SOURCE_WL_IN_NM = 633
    main(sf, total_time, ftag, False, True)
    
def main250829_120000():
    '''
    Close loop, with turb, with KL modes
    300 step ad dt 1ms
    L0=25m, r0=0.15m,D=8.2m
    Saving displayed wf on the slm
    '''
    sf  = _factory_setup250808_130000()
    total_time = 0.3
    ftag = '250829_120000'
    
    # load control matrices zc or kl
    sf.REC_MAT_TAG = '250808_144900'
    sf.MODAL_BASE_TYPE = 'kl'
    sf.KL_MODAL_IFS_TAG = '250806_170800'
    
    sf.SH_FRAMES2AVERAGE = 6
    
    #opening or closing the loop with/without turb
    sf.TELESCOPE_PUPIL_DIAMETER = 8.2
    sf._pupil_pixel_pitch = sf.TELESCOPE_PUPIL_DIAMETER/sf._pupil_diameter_in_pixel
    sf.OUTER_SCALE_L0 = 25            # m
    wl  = 500e-9
    r0 = 0.15
    seeing = (wl/r0)*(180/np.pi)*60*60
    sf.SEEING = seeing
    gain_vector =  -0.3
    sf.INT_GAIN = gain_vector 
    sf.ONAXIS_SOURCE_WL_IN_NM = 633
    main(sf, total_time, ftag, False, True)

    
###################################################################
#### Get a factory setup
def _factory_setup250808_130000():
    '''
    Using kl modes on circular pupil
    correcting 200 modes
    '''
    sf = specula_startup()
    
    #SLM_RADIUS = 545 # set on base factory
    sf.SUBAPS_TAG = '250612_143100'
    sf.REC_MAT_TAG = '250808_123600'
    sf.MODAL_BASE_TYPE = 'kl'
    sf.KL_MODAL_IFS_TAG = None
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