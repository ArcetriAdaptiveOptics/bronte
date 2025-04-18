import numpy as np 
from bronte import startup
from bronte.scao.open_loop_runner import OpenLoopRunner

def main_ol_on_tilt(ftag = '250416_161600'):
    
    ff = startup.flattening_startup()
    ff.INT_DELAY = 0
    ff.INT_GAIN = 0
    zc_vect = np.array([0., 2e-6, 0.])
    cmd = ff.slm_rasterizer.m2c(zc_vect)
    
    ff.deformable_mirror.set_shape(cmd)
    
    olr = OpenLoopRunner(ff)
    olr.modal_offset = zc_vect
    olr.run(1000)
    olr.save_telemetry(ftag)
    
def main_ol_on_wfc(ftag = '250416_171600'):
    
    ff = startup.flattening_startup()
    ff.INT_DELAY = 0
    ff.INT_GAIN = 0
    cmd = np.zeros((1920*1152))
    
    ff.deformable_mirror.set_shape(cmd)
    
    olr = OpenLoopRunner(ff)
    olr.run(1000)
    olr.save_telemetry(ftag)
    


def show_plot(ftag_mode, ftag_wfc):
    
    h_mode,s_mode,dzc_mode = OpenLoopRunner.load_telemetry(ftag_mode)
    h_wfc,s_wfc,dzc_wfc = OpenLoopRunner.load_telemetry(ftag_wfc)
    
    dzc_wfc_mean = dzc_wfc.mean(axis=0)
    dzc_mode_mean = dzc_mode.mean(axis=0)
    
    dzc_wfc_std = dzc_wfc.std(axis=0)
    dzc_mode_std = dzc_mode.std(axis=0)
    
    #TODO: controllare std
    noll_index = np.arange(2,len(dzc_wfc_mean)+2)
    import matplotlib.pyplot as plt
    
    plt.figure()
    plt.clf()
    plt.plot(noll_index, dzc_wfc_mean - dzc_mode_mean, '.-',label = 'wfc - mode')
    plt.xlabel('Noll index')
    plt.ylabel('Modal coefficients m rms wf')
    plt.legend(loc = 'best')
    plt.grid('--', alpha = 0.3)
    
    plt.figure()
    plt.clf()
    plt.plot(noll_index, dzc_wfc_std,'.-',label = 'wfc')
    plt.plot(noll_index, dzc_mode_std,'.-',label = 'mode')
    plt.xlabel('Noll index')
    plt.ylabel('Modal coefficients std m rms wf')
    plt.legend(loc = 'best')
    plt.grid('--', alpha = 0.3)