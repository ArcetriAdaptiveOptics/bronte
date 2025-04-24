import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np
from specula.data_objects.slopes import Slopes
from specula.data_objects.intmat import Intmat
from specula.data_objects.recmat import Recmat
from specula.data_objects.subap_data import SubapData
from bronte.package_data import reconstructor_folder, subaperture_set_folder, modal_offsets_folder, phase_screen_folder
from bronte.scao.specula_scao_runner import SpeculaScaoRunner
from bronte.scao.phase_screen.phase_screen_generator import PhaseScreenGenerator
from bronte.startup import set_data_dir
from functools import cached_property
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits

class ScaoTelemetryDataAnalyser():
    
    ARCSEC2RAD = np.pi/(180*60*60)
    N_PUSH_PULL = 2
    
    def __init__(self, ftag):
        set_data_dir()
        self._telemetry_tag = ftag
        self._hdr, self._slopes_vect, self._delta_cmds, self._integ_cmds = SpeculaScaoRunner.load_telemetry(ftag)
        self._load_data_from_header()
        self._first_idx_mode = 2 # j noll index mode
        self._compute_rms_slopes()
        self._ol_cmds = None
        self._compute_pseudo_open_loop_modal_cmds()
        
        
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
            #slope_map = Slopes(slopes = slopes_at_step_n).get2d()#None, self._subapdata)
            slope_obj = Slopes(slopes = slopes_at_step_n)
            slope_obj.single_mask = self._subapdata.single_mask()
            slope_obj.display_map = self._subapdata.display_map
            slope_map =  slope_obj.get2d()
            slope_maps_x.append(np.ma.array(data = slope_map[0], mask = slope_mask))
            slope_maps_y.append(np.ma.array(data = slope_map[1], mask = slope_mask))
        
        slope_map_cube_y = np.ma.array(slope_maps_y)
        slope_map_cube_x = np.ma.array(slope_maps_x)
        return slope_map_cube_x, slope_map_cube_y
    
    def _compute_rms_slopes(self):
        
        slope_map_cube_x, slope_map_cube_y = self._compute_slope_maps
        self._rms_slopes_x = np.sqrt(np.mean(slope_map_cube_x**2, axis=(1,2)))
        self._rms_slopes_y = np.sqrt(np.mean(slope_map_cube_y**2, axis=(1,2)))
    
    # def _get_total_wavefront_error_reconstruction(self):
    #
    #     var = self._integ_cmds[-1]**2
    #     tot_wf_err_in_m = np.sqrt(var.sum())
    #     return tot_wf_err_in_m
    #
    # def _get_residual_wavefront_after_perfect_compensation(self):
    #
    #     J = self.corrected_modes + self._first_idx_mode + 1
    #     #extend to vonkartmann
    #     var_J_in_rad2 = 0.2944*J**(-np.sqrt(3)*0.5)*(self.telescope_pupil_diameter/self.r0)**(5./3)
    #     return var_J_in_rad2
    #
    # def _get_seeing_limited_total_variance(self):
    #
    #     var_sl_in_rad2 = 1.0299*(self.telescope_pupil_diameter/self.r0)**(5./3)
    #     return var_sl_in_rad2
    def _compute_pseudo_open_loop_modal_cmds(self):
        
        pol_modal_cmd_list = []
        for idx in range(self.Nstep-self.integ_delay):
            pol_modal_cmd = self._integ_cmds[idx] + self._delta_cmds[idx+self.integ_delay]
            pol_modal_cmd_list.append(pol_modal_cmd)
            
        self._pol_modal_cmds = np.array(pol_modal_cmd_list)
        
    def _load_data_from_header(self):
        
        self.seeing = self._hdr.get('SEEING', 'NA')#arcsec
        self.L0 = self._hdr.get('L0_IN_M', 'NA')
        self.telescope_pupil_diameter = self._hdr.get('D_IN_M', 'NA')
        self.integ_gain = self._hdr.get('INT_GAIN','AD_HOC')
        self.integ_delay = self._hdr['INT_DEL']
        self.loop_time_step = self._hdr['TSTEP_S']#time step of the loop in sec
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
        
        if self.seeing == 'NA':
            self.r0 = 0
        else:
            self.r0 = 500e-9/(self.seeing*self.ARCSEC2RAD)

    @staticmethod    
    def _load_intmat(intmat_tag):
        
        file_name = reconstructor_folder() / (intmat_tag + '_bronte_im.fits')
        int_mat = Intmat.restore(file_name)
        Npp = 2
        int_mat._intmat = int_mat._intmat / Npp
        return int_mat
    
    @staticmethod
    def _load_recmat(recmat_tag):
        
        file_name = reconstructor_folder() / (recmat_tag + '_bronte_rec.fits')
        rec_mat = Recmat.restore(file_name)
        Npp = 2
        rec_mat.recmat = Npp * rec_mat.recmat
        return rec_mat
    
    @staticmethod
    def _load_subaperture_set(subap_tag):
    
        subapdata = SubapData.restore_from_bronte(
            subaperture_set_folder() / (subap_tag + ".fits"))
        return subapdata
    
    ###_______________________ DISPLAY_______________________________
    
    def display_delta_cmds_temporal_evolution(self, mode_index_list = [2,3,4], display_ol=None, time_range = None):
        
        plt.figure()
        plt.clf()
        if display_ol is not None:
            N = np.min((self._ol_cmds.shape[0], self._delta_cmds.shape[0]))
            time = np.arange(N)*self.loop_time_step
        else:
            time = np.arange(self.Nstep)*self.loop_time_step
            N = self.Nstep
        for j in mode_index_list:
            k = j-2
            plt.plot(time, self._delta_cmds[:N, k], '-', label = f"c{j} CL")
            if display_ol is not None:
                plt.plot(time, self._ol_cmds[:N, k], '--', label = f"c{j} OL")
            
        plt.legend(loc='best')
        plt.grid('--', alpha = 0.3)
        plt.ylabel('Delta modal command [m] rms wf')
        plt.xlabel('Time [s]')
        if time_range is not None:
            plt.xlim(np.min(time_range), np.max(time_range))
    
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
        
    def show_modal_plot(self):
        '''
        displays the temporal standard deviation of the modal coefficients
        '''

        delta_cmd_std = self._delta_cmds.std(axis=0)
        pseudo_open_loop_cmd_std = self._pol_modal_cmds.std(axis=0)
        mode_index_list = np.arange(self._first_idx_mode,
                                         self.corrected_modes + self._first_idx_mode)
        plt.figure()
        plt.clf()
        plt.loglog(mode_index_list, pseudo_open_loop_cmd_std, 'r-', label = 'P-OL')
        plt.loglog(mode_index_list, delta_cmd_std, label='CL')
        if self._ol_cmds is not None:
            ol_cmd_std = self._ol_cmds.std(axis=0)
            plt.loglog(mode_index_list, ol_cmd_std[:self.corrected_modes],'g--', label='OL')
        plt.xlabel('j Noll index')
        plt.ylabel('modal temporal std '+ r'$\sigma_{std}$' + ' [m rms wf]')
        plt.legend(loc='best')
        plt.grid('--', alpha=0.3)
    
    ### ____________________SAVE/LOAD___________________________________
    
    def save_coeff_as_modal_offset(self, coeff_vector, ftag):
        
        file_name = modal_offsets_folder() / (ftag + '.fits')
        modal_offset = coeff_vector
        hdr = fits.Header()
        hdr['TEL_TAG'] = self._telemetry_tag
        fits.writeto(file_name, modal_offset, hdr)
        
    def load_turbulent_coefficients(self, ftag):
        hdr, self._psg_cmds = PhaseScreenGenerator.load_phase_screen_fits_data(ftag)
        self._check_headers(hdr)
    
    def load_open_loop_modal_commands(self, ftag):
        self._ol_hdr, self._ol_slopes, self._ol_cmds, _ = SpeculaScaoRunner.load_telemetry(ftag)
        
    #TODO: check if the parameters are the same
    def _check_headers(self, hdr):
        '''
        Checks if the configuration parameters of the scao telemetry loop and
        the loaded open loop turbulet coefficients are the same
        '''
        pass
    
    @staticmethod
    def load_modal_offset(ftag):
        """
        Returns modal offsets from a file.

        Args:
            ftag (str): File tag for the modal offset file to be loaded.

        Returns:
            tuple: A tuple containing the modal offset array and the telemetry data file tag, from which are derived.
        """
        file_name = modal_offsets_folder() / (ftag + '.fits')
        
        header = fits.getheader(file_name)
        hduList = fits.open(file_name)
        
        telemetry_data_tag = header['TEL_TAG']
        modal_offset = hduList[0].data
        
        return modal_offset, telemetry_data_tag