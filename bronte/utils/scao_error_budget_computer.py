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
    
    def phase_var2wfe_in_nm(self, phase_var, wl=633e-9) :
        """ converting phase variance to RMS WFE (nm)"""
        return (wl / (2*np.pi)) * np.sqrt(phase_var) * 1e9
    
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