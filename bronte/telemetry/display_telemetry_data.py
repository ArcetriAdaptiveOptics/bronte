import numpy as np

from bronte.mains.main240802_ao_test import TestAoLoop
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class DisplayTelemetryData():
    
    def __init__(self, fname):
        
        self._load_telemetry_data_from_file(fname)
        self._n_of_modes = self._zc_integrated_modal_cmds.shape[-1]
        self._first_idx_mode = 2 # j noll index mode
             
    def display_short_exp_psf_at_step(self, step, roi_str = None):
        
        if roi_str is None:
            psf = self._short_exp_psfs[step]
        else:
            psf = self._get_roi(self._short_exp_psfs[step], roi_str)
        plt.figure()
        plt.clf()
        plt.title(f"Short Exposure PSF at step {step}")
        plt.imshow(psf)
        plt.colorbar(label='ADU')
    
    def display_long_exp_psf(self, roi_str = None):
        
        if roi_str is None:
            psf = self._long_exp_psf
        else:
            psf = self._get_roi(self._long_exp_psf, roi_str)
            
        plt.figure()
        plt.clf()
        plt.title(f"Long Exposure PSF after {self._n_of_loop_steps} steps")
        plt.imshow(psf)
        plt.colorbar(label='ADU')
        
    def display_slopes_maps_at_step(self, step, roi_str = None):
        
        if roi_str is None:
            slope_map_x = self._slopes_x_maps[step]
            slope_map_y = self._slopes_y_maps[step]
        else:
            slope_map_x  = self._get_roi(self._slopes_x_maps[step], roi_str)
            slope_map_y  = self._get_roi(self._slopes_y_maps[step], roi_str)
        
        fig, axs = plt.subplots(1, 2, sharex = True,
                                 sharey = True)
        
        axs[0].set_title('Slope Map X')
        im_map_x = axs[0].imshow(slope_map_x)
        # Use make_axes_locatable to create a colorbar of the same height
        divider_x = make_axes_locatable(axs[0])
        cax_x = divider_x.append_axes("right", size="5%", pad=0.15)  # Adjust size and padding
        fig.colorbar(im_map_x, cax=cax_x, label='a.u.')
        
        axs[1].set_title('Slope Map Y')
        im_map_y = axs[1].imshow(slope_map_y)
        
        divider_y = make_axes_locatable(axs[1])
        cax_y = divider_y.append_axes("right", size="5%", pad=0.15)
        fig.colorbar(im_map_y, cax=cax_y, label='a.u.')
        fig.subplots_adjust(wspace=0.5)
        fig.tight_layout()
    
    def display_delta_modal_commads_at_step(self, step, mode_index_list = None):
        
        if mode_index_list is None:
            delta_modal_command = self._zc_delta_modal_cmds[step]
            mode_index_list = np.arange(self._first_idx_mode,
                                         self._n_of_modes + self._first_idx_mode)
        else:
            delta_modal_command = self._zc_delta_modal_cmds[step][mode_index_list]
        
        plt.figure()
        plt.clf()
        plt.title(f"Delta modal command at step {step}")
        plt.plot(mode_index_list, delta_modal_command, 'bo-', label=r'$\Delta c_j$')
        plt.xlabel('j mode index')
        plt.ylabel('Zernike coefficent [m] rms wf')
        plt.grid(alpha=0.3, ls='--')
        plt.legend(loc='best')
    
    def display_integrated_modal_commands_at_step(self, step, mode_index_list=None):
        
        if mode_index_list is None:
            modal_command = self._zc_integrated_modal_cmds[step]
            mode_index_list = np.arange(self._first_idx_mode,
                                         self._n_of_modes + self._first_idx_mode)
        else:
            modal_command = self._zc_integrated_modal_cmds[step ,mode_index_list]
        
        plt.figure()
        plt.clf()
        plt.title(f"Integrated modal command at step {step}")
        plt.plot(mode_index_list, modal_command, 'ro-', label=r'$c_j$')
        plt.xlabel('j mode index')
        plt.ylabel('Zernike coefficient [m] rms wf')
        plt.grid(alpha=0.3, ls='--')
        plt.legend(loc='best')
    
    def get_slopes_x_at_step(self, step):
        return self._slopes_x_maps[step]
    
    def get_slopes_y_at_step(self, step):
        return self._slopes_y_maps[step]
    
    def get_delta_modal_commands_at_step(self, step, mode_index_list = None):
        
        if mode_index_list is None:
            delta_modal_command = self._zc_delta_modal_cmds[step]   
        else:
            delta_modal_command = self._zc_delta_modal_cmds[step][mode_index_list]
        
        return delta_modal_command
    
    def get_integrated_modal_commands_at_step(self, step, mode_index_list = None):
        
        if mode_index_list is None:
            modal_command = self._zc_integrated_modal_cmds[step]
        else:
            modal_command = self._zc_integrated_modal_cmds[step, mode_index_list]
            
        return modal_command
    
    def _get_roi(self, array_map, roi_str):
        '''
        return the roi of a 2d array
        roi_str is a string (for instance '2:10,100:200')
        '''
        rows, cols = roi_str.split(',')
        row_start, row_end = map(lambda x: int(x) if x else None, rows.split(':'))
        col_start, col_end = map(lambda x: int(x) if x else None, cols.split(':'))
        roi = array_map[row_start:row_end, col_start:col_end]
        return roi
    
    def _load_telemetry_data_from_file(self, fname):
        tag_list,\
         atmospheric_param_list,\
          loop_param_list,\
           hardware_param_list,\
            self._long_exp_psf,\
             self._short_exp_psfs,\
              self._slopes_x_maps,\
               self._slopes_y_maps,\
                self._interaction_matrix,\
                self._reconstructor,\
                self._zc_delta_modal_cmds,\
                self._zc_integrated_modal_cmds,\
                self._zc_modal_offset = TestAoLoop.load_telemetry(fname)
    
        self._subaps_tag, self._phase_screen_tag,\
            self._modal_dec_tag = tag_list[:]
        self._r0, self._wind_speed = atmospheric_param_list[:]
        
        self._integ_type, self._int_gain,\
         self._n_of_loop_steps = loop_param_list[:]
         
        self._texp_psf_cam, self._fps_psf_cam,\
            self._texp_sh_cam, self._fps_sh_cam,\
             self._tresp_slm = hardware_param_list[:]