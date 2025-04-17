import numpy as np 
from bronte.scao.phase_screen.phase_screen_generator import PhaseScreenGenerator
from arte.atmo.von_karman_spatial_covariance_calculator import VonKarmanSpatialCovariance
from bronte.types.slm_pupil_mask_generator import SlmPupilMaskGenerator
from bronte.wfs.slm_rasterizer import SlmRasterizer
import matplotlib.pyplot as plt

class PhaseScreenAnalyser():
    
    ARCSEC2RAD = np.pi/(180*60*60)
    
    def __init__(self, ftag):
        
        self._ftag = ftag
        self._hdr, self._modal_coef_cube = \
        PhaseScreenGenerator.load_phase_screen_fits_data(ftag)
        self.wl = 500e-9
        self._read_atmo_params_from_header()
        self._slm_pupil_mask = self._get_slm_pupil_mask()
        self._sr = SlmRasterizer(self._slm_pupil_mask, self.corrected_modes)
        self._noll_index_vector = np.arange(2, self.corrected_modes + 2)
        self._vk_var_in_rad2 = self._get_vk_var_in_rad()
        self._zc_mean = self._modal_coef_cube.mean(axis = 0)
        self._zc_std = self._modal_coef_cube.std(axis = 0)
        
    def _read_atmo_params_from_header(self):
        
        self._atmo_seed = self._hdr.get('ATM_SEED', 'NA')
        self.seeing = self._hdr['SEEING']
        self.L0 = self._hdr['L0_IN_M']
        self.telescope_pupil_diameter = self._hdr.get('D_IN_M', 'NA')
        self.loop_time_step = self._hdr['TSTEP_S'] #time step of the loop in sec
        self.Nstep = self._hdr['N_STEPS']
        self.corrected_modes = self._hdr.get('N_MODES', self._modal_coef_cube.shape[-1])
        
        if self.seeing == 'NA':
            self.r0 = self.wl / (0.3 * self.ARCSEC2RAD)
        else:
            self.r0 = self.wl / (self.seeing * self.ARCSEC2RAD)
            
    def _get_slm_pupil_mask(self):
        
        import ast
        spm = SlmPupilMaskGenerator(
            pupil_radius = self._hdr['SLM_RAD'],
            pupil_center = ast.literal_eval(self._hdr['SLM_YX']))
        
        return spm.circular_pupil_mask()
        
    def _get_vk_var_in_rad(self):
        
        scale = (self.telescope_pupil_diameter/self.r0)**(5./3)
        L0norm = self.L0/self.telescope_pupil_diameter
        vk_var = np.zeros(self.corrected_modes)
        
        for idx, j in enumerate(self._noll_index_vector):
            vk = VonKarmanSpatialCovariance(int(j),int(j), L0norm)
            vk_var[idx] = vk.get_covariance()
        
        vk_var_in_rad2 = 4*np.pi**2 * vk_var * scale
        return vk_var_in_rad2
    
    def display_modal_plot(self):
        
        zc_var = self._zc_std**2
        obs_zc_std = np.sqrt(zc_var)
        exp_zc_std = np.sqrt(self._vk_var_in_rad2)*self.wl*0.5/np.pi
        print(obs_zc_std/exp_zc_std)
        plt.figure()
        plt.clf()
        plt.semilogy(self._noll_index_vector, exp_zc_std, 'g--', label ='Theory')
        plt.semilogy(self._noll_index_vector, obs_zc_std, 'b-', label ='SPECULA')
        plt.grid('--', alpha = 0.3)
        plt.legend(loc = 'best')
        plt.xlabel('Noll index')
        plt.ylabel('std [m] rms wf')
        
    def get_phase_screen_at_step(self, step):
        wfz = self._sr.zernike_coefficients_to_raster(self._modal_coef_cube[step])
        return wfz.toNumpyArray()
    
    def get_wf2raster(self, zc):
        wfz = self._sr.zernike_coefficients_to_raster(zc)
        return wfz.toNumpyArray()