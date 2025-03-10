import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np
from specula.data_objects.slopes import Slopes
from specula.data_objects.intmat import Intmat
from specula.data_objects.recmat import Recmat
from specula.data_objects.subap_data import SubapData
from bronte.package_data import reconstructor_folder, subaperture_set_folder
from bronte.mains.main250206_specula_scao_loop import SpeculaScaoLoop

from functools import cached_property
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class ScaoTelemetryDataAnalyser():
    
    def __init__(self, ftag):
        
        self._telemetry_tag = ftag
        self._hdr, self._slopes_vect, self._delta_cmds, self._integ_cmds = SpeculaScaoLoop.load_telemetry(ftag)
        self._load_data_from_header()
        self._first_idx_mode = 2 # j noll index mode
    
    def get_slope_map_cubes(self):
        return self._compute_slope_maps
    
    @cached_property
    def _compute_slope_maps(self):
        
        slope_maps_x = []
        slope_maps_y = []
        idl_slope_mask = self._subapdata.single_mask()
        slope_mask = np.ones(idl_slope_mask.shape) - idl_slope_mask
        
        for n in range(self.Nstep):
            
            slopes_at_step_n = self._slopes_vect[n]
            slope_map = Slopes(slopes = slopes_at_step_n).get2d(None, self._subapdata)
            slope_maps_x.append(np.ma.array(data = slope_map[0], mask = slope_mask))
            slope_maps_y.append(np.ma.array(data = slope_map[1], mask = slope_mask))
        
        slope_map_cube_y = np.ma.array(slope_maps_y)
        slope_map_cube_x = np.ma.array(slope_maps_x)
        return slope_map_cube_x, slope_map_cube_y
    
    def _compute_rms_slopes(self):
        
        slope_map_cube_x, slope_map_cube_y = self._compute_slope_maps
        self._rms_slopes_x = np.sqrt(np.mean(slope_map_cube_x**2, axis=(1,2)))
        self._rms_slopes_y = np.sqrt(np.mean(slope_map_cube_y**2, axis=(1,2)))
    
    def _get_total_wavefront_error_reconstruction(self):
        
        var = self._integ_cmds[-1]**2
        tot_wf_err_in_m = np.sqrt(var.sum())
        return tot_wf_err_in_m
        
    def _get_residual_wavefront_after_perfect_compensation(self):
        
        J = self.corrected_modes + self._first_idx_mode + 1
        #extend to vonkartmann
        var_J_in_rad2 = 0.2944*J**(-np.sqrt(3)*0.5)*(self.telescope_pupil_diameter/self.r0)**(5./3)
        return var_J_in_rad2
    
    def _get_seeing_limited_total_variance(self):
        
        var_sl_in_rad2 = 1.0299*(self.telescope_pupil_diameter/self.r0)**(5./3)
        return var_sl_in_rad2
    
    def _load_data_from_header(self):
        
        self.r0 = self._hdr['R0_IN_M']
        self.L0 = self._hdr['L0_IN_M']
        self.telescope_pupil_diameter = self._hdr['D_IN_M']
        self.integ_gain = self._hdr['INT_GAIN']
        self.integ_delay = self._hdr['INT_DEL']
        self.Nstep = self._hdr['N_STEPS']
        self.corrected_modes = self._hdr['N_MODES']
        
        self._sh_pix_thr = self._hdr['SHPX_THR']
        self._sh_texp = self._hdr['SH_TEXP']
        self._sh_fps = self._hdr['SH_FPS']
        
        self._psf_cam_texp = self._hdr['PC_TEXP']
        self._psf_cam_fps = self._hdr['PC_FPS']
        
        self._subapdata = self._load_subaperture_set(self._hdr['SUB_TAG'])
        self._intmat = self._load_intmat(self._hdr['REC_TAG'])
        self._recmat = self._load_recmat(self._hdr['REC_TAG'])

    @staticmethod    
    def _load_intmat(intmat_tag):
        
        file_name = reconstructor_folder() / (intmat_tag + '_bronte_im.fits')
        int_mat = Intmat.restore(file_name)
        return int_mat
    
    @staticmethod
    def _load_recmat(recmat_tag):
        
        file_name = reconstructor_folder() / (recmat_tag + '_bronte_rec.fits')
        rec_mat = Recmat.restore(file_name)
        return rec_mat
    
    @staticmethod
    def _load_subaperture_set(subap_tag):
    
        subapdata = SubapData.restore_from_bronte(
            subaperture_set_folder() / (subap_tag + ".fits"))
        return subapdata
    
    def display_delta_modal_commads_at_nth_step(self, step, mode_index_list = None):
        """
        Display the delta modal commands (i.e. Zernike coefficients) at a specific step of the loop.

        Args:
            step (int): Step index of the loop.
            mode_index_list (list, optional): List of mode indices to display. If None, display all.
        """
        if mode_index_list is None:
            delta_modal_command = self._delta_cmds[step]
            mode_index_list = np.arange(self._first_idx_mode,
                                         self.corrected_modes + self._first_idx_mode)
        else:
            mode_index = np.array(mode_index_list) - self._first_idx_mode
            delta_modal_command = self._delta_cmds[step][mode_index]
        
        plt.figure()
        plt.clf()
        plt.title(f"Delta modal command at step {step}")
        plt.plot(mode_index_list, delta_modal_command, 'bo-', label=r'$\Delta c_j$')
        plt.xlabel('j mode index')
        plt.ylabel('Zernike coefficent [m] rms wf')
        plt.grid(alpha=0.3, ls='--')
        plt.legend(loc='best')
        
    def display_integrated_modal_commands_at_nth_step(self, step, mode_index_list=None):
        """
        Display the integrated modal commands (i.e Zernike coefficients) at a specific step of the loop.

        Args:
            step (int): Step index of the loop.
            mode_index_list (list, optional): List of mode indices to display. If None, display all.
        """
        if mode_index_list is None:
            modal_command = self._integ_cmds[step]
            mode_index_list = np.arange(self._first_idx_mode,
                                         self.corrected_modes + self._first_idx_mode)
        else:
            mode_index = np.array(mode_index_list) - self._first_idx_mode
            modal_command = self._integ_cmds[step, mode_index]
        
        plt.figure()
        plt.clf()
        plt.title(f"Integrated modal command at step {step}")
        plt.plot(mode_index_list, modal_command, 'ro-', label=r'$c_j$')
        plt.xlabel('j mode index')
        plt.ylabel('Zernike coefficient [m] rms wf')
        plt.grid(alpha=0.3, ls='--')
        plt.legend(loc='best')
    
    def display_slope_maps_at_nth_step(self, n_th_step):
        
        cube_x, cube_y = self._compute_slope_maps
        slope_map_x = cube_x[n_th_step]
        slope_map_y = cube_y[n_th_step]
        
        z_scale_min = min(slope_map_x.min(), slope_map_y.min())
        z_scale_max = max(slope_map_x.max(), slope_map_y.max())
        
        fig, axs = plt.subplots(1, 2, sharex = True,
                         sharey = True)
        
        axs[0].set_title('Slope Map X')
        im_map_x = axs[0].imshow(slope_map_x, vmin = z_scale_min, vmax = z_scale_max)
        # Use make_axes_locatable to create a colorbar of the same height
        divider_x = make_axes_locatable(axs[0])
        cax_x = divider_x.append_axes("right", size="5%", pad=0.15)  # Adjust size and padding
        fig.colorbar(im_map_x, cax=cax_x, label='a.u.')
        
        axs[1].set_title('Slope Map Y')
        im_map_y = axs[1].imshow(slope_map_y, vmin = z_scale_min, vmax = z_scale_max)
        
        divider_y = make_axes_locatable(axs[1])
        cax_y = divider_y.append_axes("right", size="5%", pad=0.15)
        fig.colorbar(im_map_y, cax=cax_y, label='a.u.')
        fig.subplots_adjust(wspace=0.5)
        fig.suptitle(f"Step {n_th_step}-th")
        fig.tight_layout()
    
    def display_rms_slopes(self):
        
        plt.figure()
        plt.clf()
        plt.plot(self._rms_slopes_x, label='slope-x')
        plt.plot(self._rms_slopes_y, label='slope-y')
        plt.grid(alpha = 0.3, ls='--')
        plt.legend(loc = 'best')
        plt.ylabel('rms slopes [au]')
        plt.xlabel('step')