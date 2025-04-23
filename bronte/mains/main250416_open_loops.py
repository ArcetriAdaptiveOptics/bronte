import numpy as np 
from bronte import startup
from bronte.scao.open_loop_runner import OpenLoopRunner
from bronte.package_data import other_folder
import time
from bronte.utils.noll_to_radial_order import from_noll_to_radial_order
from astropy.io import fits
import matplotlib.pyplot as plt

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
    
def main_ol_on_mode(ftag = '250416_161600', zc_vect = np.array([0., 2e-6, 0.])):
    
    ff = startup.flattening_startup()
    ff.INT_DELAY = 0
    ff.INT_GAIN = 0
    cmd = ff.slm_rasterizer.m2c(zc_vect)
    
    ff.deformable_mirror.set_shape(cmd)
    
    olr = OpenLoopRunner(ff)
    olr.modal_offset = zc_vect
    olr.run(5)
    olr.save_telemetry(ftag)
    
def _execute_ol_on_mode(factory, ftag = '250416_161600', zc_vect = np.array([0., 2e-6, 0.])):
    
    cmd = factory.slm_rasterizer.m2c(zc_vect)
    factory.deformable_mirror.set_shape(cmd)
    olr = OpenLoopRunner(factory)
    olr.modal_offset = zc_vect
    olr.run(500)
    olr.save_telemetry(ftag)
    
def execute_ol_on_many_modes(selected_modes = [2,3,4]):
    
    ff = startup.flattening_startup()
    ff.INT_DELAY = 0
    ff.INT_GAIN = 0
    Nmodes = 200
    j_noll_vector = np.arange(200) + 2
    radial_order = from_noll_to_radial_order(j_noll_vector)
    ampl_vect = 2e-6 /(radial_order)
    counter = 0
    ftag_list = []
    for j in selected_modes:
        counter+=1
        idx = int(j-2)
        print(f"OpenLoop on Z{j} \t step: {counter}/{len(selected_modes)}")
        ftag = time.strftime("%y%m%d_%H%M00")
        ftag_list.append(ftag)
        zc_vect = np.zeros(Nmodes)
        zc_vect[idx] = ampl_vect[idx]
        _execute_ol_on_mode(ff, ftag, zc_vect)
        plt.close('all')
    
    fname = time.strftime("%y%m%d_%H%M00") + "_main250416_ol_on_many_modes"
    file_name = other_folder() / (fname + '.fits')
    hdr = fits.Header()
    hdr['MAIN'] = 'main250416_open_loops'
    fits.writeto(file_name, np.array(selected_modes), hdr)
    fits.append(file_name, ampl_vect)
    ftag_string_array = np.array(ftag_list)
    int_ftag_array = np.array([int(s.replace('_', '')) for s in ftag_string_array])
    fits.append(file_name, np.array(int_ftag_array))
    

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