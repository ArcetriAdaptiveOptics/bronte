import numpy as np 
from bronte import startup
from bronte.telemetry.display_telemetry_data import DisplayTelemetryData
import matplotlib.pyplot as plt

class SlmGeneratedModesAnalyser():
    '''
    This class is aimed to infer the SLM capabilities in reproducing zenike
    modes.
    the data to be analysed are related to the one obtained from 
    250124_bronte_open_loop_on_modal_offset.py
     
    '''
    
    
    def __init__(self, ftag_openloop, ftag_modal_offset):
        
        factory = startup.startup()
        self._modal_offset, _ = DisplayTelemetryData.load_modal_offset(ftag_modal_offset)
        self._dtd = DisplayTelemetryData(ftag_openloop)
        self._delta_cmds = self._dtd._zc_delta_modal_cmds
        
        
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
    
    def display_difference_with_modal_offset(self):
        
        modes_difference = self._delta_cmds[-2] - self._modal_offset
        plt.figure()
        plt.clf()
        plt.plot(modes_difference, '.-', label='(-dcmds) - modal_offset')
        plt.ylabel('Zernike coefficient [m] rms wavefront')
        plt.xlabel('index')
        plt.xlim(0, 10)
        plt.legend(loc='best')