import numpy as np

from bronte.mains.main240802_ao_test import TestAoLoop
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from bronte.package_data import modal_offsets_folder
from astropy.io import fits


class DisplayTelemetryData():
    """
    A class for visualizing and managing telemetry data collected from main240802_ao_test.py.

    This class provides methods to display point spread functions (PSFs), slope maps, and Zernike modal coefficients
    from telemetry data, as well as utility functions for saving and loading modal offsets.

    Attributes:
        ftag (str): File tag associated with the telemetry data file, saved with
        main240802_ao_test.py.

    """
    def __init__(self, ftag):
        
        self._telemetry_data_ftag = ftag
        self._load_telemetry_data_from_file(ftag)
        self._n_of_modes = self._zc_integrated_modal_cmds.shape[-1]
        self._first_idx_mode = 2 # j noll index mode
             
    def display_short_exp_psf_at_step(self, step, roi_str = None):
        """
        Display the short-exposure PSF at a specific step of the loop.

        Args:
            step (int): Step index of the PSF frame.
            roi_str (str, optional): Region of interest (ROI) of the PSF to be displayed as a string, e.g., '2:10,100:200'.
        """
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
        """
        Display the long-exposure PSF.

        Args:
            roi_str (str, optional): Region of interest (ROI)  of the PSF to be displayed as a string, e.g., '2:10,100:200'.
        """
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
        """
        Display the slope maps (X and Y) at a specific step of the loop.

        Args:
            step (int): Step index of the loop.
            roi_str (str, optional): Region of interest (ROI) of the slope maps as a string, e.g., '2:10,100:200'.
        """
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
        fig.colorbar(im_map_x, cax=cax_x, label='rad')
        
        axs[1].set_title('Slope Map Y')
        im_map_y = axs[1].imshow(slope_map_y)
        
        divider_y = make_axes_locatable(axs[1])
        cax_y = divider_y.append_axes("right", size="5%", pad=0.15)
        fig.colorbar(im_map_y, cax=cax_y, label='rad')
        fig.subplots_adjust(wspace=0.5)
        fig.tight_layout()
        
    def display_rms_slopes(self):
        
        plt.figure()
        plt.clf()
        plt.plot(self._rms_slopes_x, label='slope-x')
        plt.plot(self._rms_slopes_y, label='slope-y')
        plt.grid(alpha=0.3, ls='--')
        plt.legend(loc='best')
        plt.ylabel('rms slopes [rad]')
        plt.xlabel('step')
    
    def display_delta_modal_commads_at_step(self, step, mode_index_list = None):
        """
        Display the delta modal commands (i.e. Zernike coefficients) at a specific step of the loop.

        Args:
            step (int): Step index of the loop.
            mode_index_list (list, optional): List of mode indices to display. If None, display all.
        """
        if mode_index_list is None:
            delta_modal_command = self._zc_delta_modal_cmds[step]
            mode_index_list = np.arange(self._first_idx_mode,
                                         self._n_of_modes + self._first_idx_mode)
        else:
            mode_index = np.array(mode_index_list) - self._first_idx_mode
            delta_modal_command = self._zc_delta_modal_cmds[step][mode_index]
        
        plt.figure()
        plt.clf()
        plt.title(f"Delta modal command at step {step}")
        plt.plot(mode_index_list, delta_modal_command, 'bo-', label=r'$\Delta c_j$')
        plt.xlabel('j mode index')
        plt.ylabel('Zernike coefficent [m] rms wf')
        plt.grid(alpha=0.3, ls='--')
        plt.legend(loc='best')
    
    def display_integrated_modal_commands_at_step(self, step, mode_index_list=None):
        """
        Display the integrated modal commands (i.e Zernike coefficients) at a specific step of the loop.

        Args:
            step (int): Step index of the loop.
            mode_index_list (list, optional): List of mode indices to display. If None, display all.
        """
        if mode_index_list is None:
            modal_command = self._zc_integrated_modal_cmds[step]
            mode_index_list = np.arange(self._first_idx_mode,
                                         self._n_of_modes + self._first_idx_mode)
        else:
            mode_index = np.array(mode_index_list) - self._first_idx_mode
            modal_command = self._zc_integrated_modal_cmds[step, mode_index]
        
        plt.figure()
        plt.clf()
        plt.title(f"Integrated modal command at step {step}")
        plt.plot(mode_index_list, modal_command, 'ro-', label=r'$c_j$')
        plt.xlabel('j mode index')
        plt.ylabel('Zernike coefficient [m] rms wf')
        plt.grid(alpha=0.3, ls='--')
        plt.legend(loc='best')
    
    def show_psd_of_residual_slopes(self, loop_time_step_in_sec=0.703):
        
        fft_x = np.fft.fft(self._rms_slopes_x - np.mean(self._rms_slopes_x))
        fft_y = np.fft.fft(self._rms_slopes_y - np.mean(self._rms_slopes_y))
        freqs = np.fft.fftfreq(len(self._rms_slopes_x), d = loop_time_step_in_sec)
        psd_x = np.abs(fft_x)**2
        psd_y = np.abs(fft_y)**2
        
        plt.figure()
        plt.loglog(freqs[:len(freqs)//2], psd_x[:len(freqs)//2], label='PSD Slope-X')
        plt.loglog(freqs[:len(freqs)//2], psd_y[:len(freqs)//2], label='PSD Slope-Y')
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Power Spectral Density " +r"$[rad^2/Hz]$")
        plt.legend()
        plt.grid()
        
        
        
    def get_rms_slopes_x(self):
        
        return self._rms_slopes_x

    def get_rms_slopes_y(self):
        
        return self._rms_slopes_y
        
    def get_slopes_x_at_step(self, step):
        """
        Returns the slope map (2D) in the X axis at a specific step of the loop.

        Args:
            step (int): Step index of the loop.

        Returns:
            np.ndarray: slope map in the X direction for the specified step of the loop.
        """
        return self._slopes_x_maps[step]
    
    def get_slopes_y_at_step(self, step):
        """
        Returns the slope map (2D) in the Y axis at a specific step of the loop.

        Args:
            step (int): Step index of the loop.

        Returns:
            np.ndarray: slope map in the Y direction for the specified step of the loop.
        """
        return self._slopes_y_maps[step]
    
    def get_delta_modal_commands_at_step(self, step, mode_index_list = None):
        """
        Returns the delta modal commands at a specific step of the loop.

        Args:
            step (int): Step index of the loop.
            mode_index_list (list, optional): List of mode indices to return. If None, returns.

        Returns:
            np.ndarray: Delta modal commands for the specified step of the loop.
        """
        if mode_index_list is None:
            delta_modal_command = self._zc_delta_modal_cmds[step]   
        else:
            delta_modal_command = self._zc_delta_modal_cmds[step][mode_index_list]
        
        return delta_modal_command
    
    def get_integrated_modal_commands_at_step(self, step, mode_index_list = None):
        """
        Returns the integrated modal commands at a specific step of the loop.

        Args:
            step (int): Step index of the loop.
            mode_index_list (list, optional): List of mode indices to return. If None, returns.

        Returns:
            np.ndarray: Integrated modal commands for the specified step of the loop.
        """
        if mode_index_list is None:
            modal_command = self._zc_integrated_modal_cmds[step]
        else:
            modal_command = self._zc_integrated_modal_cmds[step, mode_index_list]
            
        return modal_command
    
    def _get_roi(self, array_map, roi_str):
        """
        Returns the region of interest (ROI) from a 2D array.

        Args:
            array_map (np.ndarray): 2D array.
            roi_str (str): ROI as a string, e.g., '2:10,100:200'.

        Returns:
            np.ndarray: Extracted ROI from the input array.
        """
        
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
        
        self._convert_slope_maps_to_masked_array()
        self._compute_rms_slopes()
        
        self._subaps_tag, self._phase_screen_tag,\
            self._modal_dec_tag = tag_list[:]
        self._r0, self._wind_speed = atmospheric_param_list[:]
        
        self._integ_type, self._int_gain,\
         self._n_of_loop_steps = loop_param_list[:]
         
        self._texp_psf_cam, self._fps_psf_cam,\
            self._texp_sh_cam, self._fps_sh_cam,\
             self._tresp_slm = hardware_param_list[:]
             
    def _convert_slope_maps_to_masked_array(self):
        frame_shape = self._slopes_x_maps[0].shape
        Nframes = self._slopes_x_maps.shape[0]
        mask = np.zeros(frame_shape)
        mask[self._slopes_x_maps[-1] == 0.0] = 1
        slope_map_x_ma = np.ma.zeros(self._slopes_x_maps.shape)
        slope_map_y_ma = np.ma.zeros(self._slopes_y_maps.shape)
        for idx in np.arange(Nframes):
            slope_map_x_ma[idx] = np.ma.array(self._slopes_x_maps[idx], mask = mask)
            slope_map_y_ma[idx] = np.ma.array(self._slopes_y_maps[idx], mask = mask)
        self._slopes_x_maps = slope_map_x_ma
        self._slopes_y_maps = slope_map_y_ma
        
    def _compute_rms_slopes(self):
        self._rms_slopes_x = np.sqrt(np.mean(self._slopes_x_maps**2, axis=(1,2)))
        self._rms_slopes_y = np.sqrt(np.mean(self._slopes_y_maps**2, axis=(1,2)))
             
    def save_integrated_coefficients_as_modal_offset(self, ftag):
        """
        Saves the integrated modal coefficients as modal offsets in a fits file.

        Args:
            ftag (str): File tag for the modal offset file.
        """
        file_name = modal_offsets_folder() / (ftag + '.fits')
        modal_offset = self.get_integrated_modal_commands_at_step(-1)
        hdr = fits.Header()
        hdr['TEL_TAG'] = self._telemetry_data_ftag
        fits.writeto(file_name, modal_offset, hdr)
       
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
    
    def _display_slopes_maps(self, sl_x, sl_y, roi_str = None):
      
        if roi_str is None:
            slope_map_x = sl_x
            slope_map_y = sl_y
        else:
            slope_map_x  = self._get_roi(sl_x, roi_str)
            slope_map_y  = self._get_roi(sl_y, roi_str)
        
        fig, axs = plt.subplots(1, 2, sharex = True,
                                 sharey = True)
        
        axs[0].set_title('Slope Map X')
        im_map_x = axs[0].imshow(slope_map_x)
        # Use make_axes_locatable to create a colorbar of the same height
        divider_x = make_axes_locatable(axs[0])
        cax_x = divider_x.append_axes("right", size="5%", pad=0.15)  # Adjust size and padding
        fig.colorbar(im_map_x, cax=cax_x, label='rad')
        
        axs[1].set_title('Slope Map Y')
        im_map_y = axs[1].imshow(slope_map_y)
        
        divider_y = make_axes_locatable(axs[1])
        cax_y = divider_y.append_axes("right", size="5%", pad=0.15)
        fig.colorbar(im_map_y, cax=cax_y, label='rad')
        fig.subplots_adjust(wspace=0.5)
        fig.tight_layout()