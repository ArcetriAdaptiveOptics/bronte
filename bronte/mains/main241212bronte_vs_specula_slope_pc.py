import numpy as np
import copy
import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np

from specula.data_objects.subap_data import SubapData
from specula.data_objects.pixels import Pixels
from specula.processing_objects.sh_slopec import ShSlopec
from specula.data_objects.slopes import Slopes
from bronte.mains import main241211_pixel_output_from_specula
from bronte.startup import startup
from bronte.package_data import subaperture_set_folder
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

def main():
    
    subap_tag = '241202_172000'
    modal_offset_tag = '241211_160500_modal_offset'
    n_modes = 3
    
    factory = startup()
    
    if factory.SUBAPS_TAG != subap_tag and \
        factory.MODAL_OFFSET_TAG != modal_offset_tag and \
            n_modes != factory.N_MODES_TO_CORRECT :
        err_message = 'Error: configure the factory with the same Nmodes, subap and modal offset tag file!'
        raise ValueError(err_message)
    
    #uncomment if needed
    #modal_offset,_ = DisplayTelemetryData.load_modal_offset(modal_offset_tag)
    
    #ftag of the calibration with 3 modes tiptilt and focus with a pp of 1um rms wf
    ftag_sh_frame = '241212_105500'
    #ftag for shframe with the flat + offset command on the slm
    # the last 2 frames are null push-pulls
    ftag_sh_frame_ref = '241212_104300' 
    
    ppsh_frames = main241211_pixel_output_from_specula.main(ftag_sh_frame)
    sh_ref_frames = main241211_pixel_output_from_specula.main(ftag_sh_frame_ref)
    ppsh_frames.append(sh_ref_frames[-1])
    out_pixel_list =  ppsh_frames
    bvs = BronteVsSpeculaSlopePC(out_pixel_list, factory)
    
    return bvs

class BronteVsSpeculaSlopePC():
    '''
    This class is meant to compare the slopes maps obtained using Bronte/Argos
    and specula slope computers to debug test_atmo.py and main240802_ao_test.
    Note that test_atmo.py (specula) relies on measured int and rec matrices
    while main240802_ao_test.py (fully bronte) on synthetic ones
    
    
    out_pixel list is a list of sh frames.
    the last elemtent of the list is taken as a reference (sh frame acquired
    when the flat+offset cmd is applied on the slm)
    '''
    
    def __init__(self, out_pixel_list, factory):
        
        self._pp_sh_frames = out_pixel_list[:-1]
        self._sh_ref_frame = out_pixel_list[-1]
        self._frame_shape = self._sh_ref_frame.shape
        self._factory = factory
        
        self._specula_subap = SubapData.restore_from_bronte(
            subaperture_set_folder() / (self._factory.SUBAPS_TAG + ".fits"))
        
        # if bkg is None:
        #     bkg = np.zeros(self._frame_shape)
        
    def _compute_specula_slopes(self, idx):
        pix = Pixels(dimx = self._frame_shape[1], dimy = self._frame_shape[0])
        pix.pixels = self._pp_sh_frames[idx]
        slopec = ShSlopec(subapdata=self._specula_subap)
        slopec.inputs['in_pixels'].set(pix)
        slopec.calc_slopes_nofor()
        
        s = copy.copy(slopec.slopes.slopes)
        
        pix.pixels = self._sh_ref_frame
        slopec.inputs['in_pixels'].set(pix)
        slopec.calc_slopes_nofor()
        sref = copy.copy(slopec.slopes.slopes)
        
        s1_2d =  Slopes(slopes = s).get2d(None, pupdata=self._specula_subap)
        sref_2d = Slopes(slopes = sref).get2d(None, pupdata=self._specula_subap)
        
        return s1_2d, sref_2d
    
    def display_specula_slope_maps(self, j_mode_index):
        
        idx = 2*j_mode_index-4
        push_s2d, s2d_ref = self._compute_specula_slopes(idx)
        
        slope_map_x = push_s2d[0] - s2d_ref[0]
        slope_map_y = push_s2d[1] - s2d_ref[1]
        
        fig, axs = plt.subplots(1, 2, sharex = True,
                                 sharey = True)
        
        axs[0].set_title('Slope Map X')
        im_map_x = axs[0].imshow(slope_map_x, cmap='jet')
        # Use make_axes_locatable to create a colorbar of the same height
        divider_x = make_axes_locatable(axs[0])
        cax_x = divider_x.append_axes("right", size="5%", pad=0.15)  # Adjust size and padding
        fig.colorbar(im_map_x, cax=cax_x, label='a.u.')
        
        axs[1].set_title('Slope Map Y')
        im_map_y = axs[1].imshow(slope_map_y, cmap='jet')
        
        divider_y = make_axes_locatable(axs[1])
        cax_y = divider_y.append_axes("right", size="5%", pad=0.15)
        fig.colorbar(im_map_y, cax=cax_y, label='a.u.')
        fig.subplots_adjust(wspace=0.5)
        
        fig.suptitle(f"SPECULA slopes: Mode index {j_mode_index}")
        fig.tight_layout()
        
        
    def _compute_bronte_slopes(self, idx):
        
        self._factory.slope_computer.set_frame(self._pp_sh_frames[idx])
        sx = self._factory.slope_computer.slopes_x_map()
        sy = self._factory.slope_computer.slopes_y_map()
        
        self._factory.slope_computer.set_frame(self._sh_ref_frame)
        sx_ref = self._factory.slope_computer.slopes_x_map()
        sy_ref = self._factory.slope_computer.slopes_y_map()
        
        s1_2d  = np.array([sx,sy])
        sref_2d = np.array([sx_ref, sy_ref])
        
        return s1_2d, sref_2d
    
    def display_bronte_slope_maps(self, j_mode_index):
        
        idx = 2*j_mode_index-4
        push_s2d, s2d_ref = self._compute_bronte_slopes(idx)
        
        slope_map_x = push_s2d[0] - s2d_ref[0]
        slope_map_y = push_s2d[1] - s2d_ref[1]
        
        fig, axs = plt.subplots(1, 2, sharex = True,
                                 sharey = True)
        
        axs[0].set_title('Slope Map X')
        im_map_x = axs[0].imshow(slope_map_x, cmap='jet')
        # Use make_axes_locatable to create a colorbar of the same height
        divider_x = make_axes_locatable(axs[0])
        cax_x = divider_x.append_axes("right", size="5%", pad=0.15)  # Adjust size and padding
        fig.colorbar(im_map_x, cax=cax_x, label='a.u.')
        
        axs[1].set_title('Slope Map Y')
        im_map_y = axs[1].imshow(slope_map_y, cmap='jet')
        
        divider_y = make_axes_locatable(axs[1])
        cax_y = divider_y.append_axes("right", size="5%", pad=0.15)
        fig.colorbar(im_map_y, cax=cax_y, label='a.u.')
        fig.subplots_adjust(wspace=0.5)
        
        fig.suptitle(f"Bronte slopes: Mode index {j_mode_index}")
        fig.tight_layout()
