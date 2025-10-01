import numpy as np
from bronte.ncpa.sharp_psf_on_camera import SharpPsfOnCamera
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from bronte.utils.raw_strehl_ratio_computer import StrehlRatioComputer

class SharpeningAnalyzer():
    
    def __init__(self, fname):
        
        self._texp, self._tot_dl_flux, self._ncpa, self._zc_offset, \
        self._corrected_z_modes_indexes,self._measured_sr, self._amp_span,\
        self._comp_psf, self._uncomp_psf, self._au_dl_psf, self._mesured_i_roi = SharpPsfOnCamera.load_ncpa(fname)
        self._sr_computer = StrehlRatioComputer()
        self._execute_interpolation()
        
    def _execute_interpolation(self):
        
        damp = 5e-9#self._amp_span.max()*0.01
        self._amps = np.arange(self._amp_span.min(), self._amp_span.max() + damp, damp)
        Nfuncs = len(self._corrected_z_modes_indexes)
        self._sr_func_list = np.zeros((Nfuncs, len(self._amps)))
        self._best_amps = np.zeros(Nfuncs)
        
        for idx, j in enumerate(self._corrected_z_modes_indexes):
            j = int(j) 
            k = j - 2
            sr_interp_functon = CubicSpline(self._amp_span, self._measured_sr[k], bc_type='natural')
            self._sr_func_list[idx] = sr_interp_functon(self._amps)
            max_idx  = np.where(self._sr_func_list[idx] == self._sr_func_list[idx].max())[0][0]
            self._best_amps[idx] = self._amps[max_idx]
            
    def get_ncpa(self):
        
        return self._ncpa
    def get_zc_offset(self):
        
        return self._zc_offset
    
    def display_sr_interpolation(self):
        
        plt.figure()
        #ax = plt.gca()
        for idx, j in enumerate(self._corrected_z_modes_indexes):
           
            
            plt.plot(self._amp_span/1e-9, self._measured_sr[j-2],'o',label='j=%d'%j)
            color = plt.gca().lines[-1].get_color()
            plt.plot(self._amps/1e-9, self._sr_func_list[idx],'-', color = color)
            
        plt.xlabel('$c_j$'+' '+ '[nm] rms')
        plt.ylabel('Strehl Ratio')
        plt.grid(ls='--',alpha = 0.4)
        plt.legend(loc='best')
        
    def display_sharpening_res(self):
        
        sr_before  = self._sr_computer.get_SR_from_image(self._uncomp_psf, True)
        sr_after = self._sr_computer.get_SR_from_image(self._comp_psf, True)
        
        print("Before PSF Sharpening: SR = %f"%sr_before)
        print("After PSF Sharpening: SR = %f"%sr_after)