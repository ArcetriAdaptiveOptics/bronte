import numpy as np
from bronte.package_data import ifs_folder


class ScaoErrorBudgetComputer():
    
    def __init__(self, wl, r0, L0):
        
        self.wl = wl
        self.r0 = r0
        self.L0 = L0
    
    
    def _vk_k_const(self):
        return 2*np.pi*0.0229*(3./5)

    def get_fitting_var_vk_closed(self, fc = None, d_dm = None) :
     
        if fc is None:
            fc = 1.0 / (2.0 * d_dm)
        f0 = 0.0 if np.isinf(self.L0) else 1.0 / self.L0
        K = self._vk_k_const()
        return K * self.r0**(-5./3) * (fc**2 + f0**2)**(-5./6)


    def get_fitting_var_kolm_closed(self, d_dm):
    
        K = self._vk_k_const()
        coeff = K * 2**(5/3)   # sim 0.274
        return coeff * (d_dm / self.r0)**(5./3)
    
    def get_band_width_pure_delay_var(self, delay_in_sec, v_in_ms):
        
        
        tau = delay_in_sec
        tau0 = 0.314 * self.r0/v_in_ms
        
        return (tau/tau0)**(5./3) 
    
    def get_band_width_filtered_var(self, f3dB, v_in_ms):
        '''
        f3dB is typically 1/10 of sampling frequency
        '''
        fg = 0.426 *v_in_ms/self.r0
        
        return (fg/f3dB)**(5./3)
    
    def phase_var2wfe_in_nm(self, phase_var, wl=633e-9) :
        """ converting phase variance to RMS WFE (nm)"""
        return (wl / (2*np.pi)) * np.sqrt(phase_var) * 1e9
    
    def wfe_in_nm2phase_var(self, wfe_in_nm, wl=633e-9):
        return (2*np.pi/wl)*wfe_in_nm**2
        
    def load_singular_values(self, ifs_ftag):
        
        from bronte.startup import set_data_dir
        from astropy.io import fits
        set_data_dir()
        
        fname = ifs_folder() / (ifs_ftag + '_singular_values_.fits')
        hduList = fits.open(fname)
        
        self._s_ifs = hduList[0].data # in rad2
        self._s_turb = hduList[1].data# in rad2
        
    def get_residual_var_from_kl_singular_values(self, j = 200):
        
        res_var_in_rad2 = self._s_turb[j:].sum()
        
        return res_var_in_rad2
    
    def get_sr(self, tot_var_in_rad2):
        
        return np.exp(-tot_var_in_rad2)