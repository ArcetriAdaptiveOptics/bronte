import numpy as np 
from bronte import startup
from bronte.telemetry_trash.display_telemetry_data import DisplayTelemetryData
import matplotlib.pyplot as plt

class SlmGeneratedModesAnalyser():
    '''
    This class is aimed to infer the SLM capabilities in reproducing zernike
    modes.
    the data to be analysed are related to the one obtained from 
    250124_bronte_open_loop_on_modal_offset.py
     
    '''
    
    
    def __init__(self, ftag_openloop, ftag_modal_offset):
        
        factory = startup.startup()
        self._modal_offset, _ = DisplayTelemetryData.load_modal_offset(ftag_modal_offset)
        self._dtd = DisplayTelemetryData(ftag_openloop)
        self._delta_cmds = self._dtd._zc_delta_modal_cmds
        self._compute_mean_measured_coeff()
        
    def disply_coeffcient_vs_time(self, j_noll_index = 2, dt_in_sec=0.703):
        k = j_noll_index - 2
        
        amps = - self._delta_cmds[:-2,k] - self._modal_offset[k]
        Npoint = len(amps)
        time = np.arange(Npoint)*dt_in_sec
        
        plt.figure()
        plt.clf()
        plt.plot(time , amps, 'b.-', label='(-dcms) - modal_offset')
        plt.ylabel('Zernike coefficient [m] rms wavefront')
        plt.xlabel('time [s]')
        plt.legend(loc='best')
        plt.title('Z%d'%j_noll_index)
        plt.grid('--', alpha=0.3)
    
    def display_difference_with_modal_offset(self):
        
        self._modes_difference = -self._delta_cmds[-2] - self._modal_offset
        noll_index = np.arange(len(self._modes_difference)) + 2
        plt.figure()
        plt.clf()
        plt.plot(noll_index, self._modes_difference, '.-', label='(-dcmds) - modal_offset')
        plt.ylabel('Zernike coefficient [m] rms wavefront')
        plt.xlabel('j [Noll index]')
        plt.xlim(1, 11)
        plt.grid('--', alpha=0.3)
        
        plt.legend(loc='best')
        
    def display_modal_offset_and_dcmds(self):
        
        self._modes_difference = -self._delta_cmds[-2] - self._modal_offset
        noll_index = np.arange(len(self._modal_offset)) + 2
        plt.figure()
        plt.clf()
        plt.plot(noll_index, self._modal_offset, '.-', label='modal_offset')
        plt.plot(noll_index, -self._delta_cmds[-2], '.-', label='-dcmds')
        plt.ylabel('Zernike coefficient [m] rms wavefront')
        plt.xlabel('j [Noll index]')
        plt.xlim(1, 11)
        plt.grid('--', alpha=0.3)
        plt.legend(loc='best')
        
    def _compute_mean_measured_coeff(self):
        
        mean_amp_per_mode = - self._delta_cmds[:-2] - self._modal_offset
        self._mean_amp_per_mode = mean_amp_per_mode.mean(axis=0)
        err_per_mode = - self._delta_cmds[:-2] - self._modal_offset
        self._err_per_mode = err_per_mode.std(axis=0)
        
    def display_mean_measured_coeff(self):
        Nmodes=len(self._modal_offset)
        noll_index = np.arange(Nmodes) + 2
        
        plt.figure()
        plt.clf()
        plt.semilogy(noll_index, np.abs(self._mean_amp_per_mode),'.-', label=r"$|<c_j>|$")
        plt.ylabel('Mean Zernike coefficient [m] rms wavefront')
        plt.xlabel('j [Noll index]')
       
        plt.grid('--', alpha=0.3)
        plt.legend(loc='best')
        
        plt.figure()
        plt.clf()
        plt.semilogy(noll_index, self._err_per_mode, label =r'$\sigma^{std}_{j}$')
        plt.ylabel('Error on Zernike coefficient [m] rms wavefront')
        plt.xlabel('j [Noll index]')
       
        plt.grid('--', alpha=0.3)
        plt.legend(loc='best')
        