import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np
#from specula.data_objects.subap_data import SubapData
from specula.data_objects.slopes import Slopes
from bronte.mains.main250212_check_eigenvalues_from_specula_calib import load_intmat
from bronte.startup import specula_startup
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def main():
    
    subap_tag = '250120_122000'
    intmat_tag_200modespp2 = '250218_125600' # thresholded in subaps pix
    intmat_tag_200modespp5 = '250218_085300' # thresholded in subaps pix
    intmat_tag_200modespp8 = '250211_154500' # not thresholded in subaps pix
    
    dsm = DisplaySlopeMapsFromIfs(intmat_tag_200modespp2, subap_tag)
    dsm.display_slope_maps(0)
    
class DisplaySlopeMapsFromIfs():
    
    def __init__(self, intmat_tag, subaps_tag):
        
        sf = specula_startup()
        sf.SUBAPS_TAG = subaps_tag
        self._subapdata = sf.subapertures_set
        self._intmat = load_intmat(intmat_tag)
        self._load_slopes_from_ifs()
        
    def _load_slopes_from_ifs(self):
    
        #n_slopes = int_mat.shape[-1]
        n_modes = self._intmat._intmat.shape[0]
        
        slope_maps_x = []
        slope_maps_y = []
        idl_slope_mask = self._subapdata.single_mask()
        slope_mask = np.ones(idl_slope_mask.shape) - idl_slope_mask
        
        for idx in range(n_modes):
            ifs = self._intmat._intmat[idx]
            slope_obj = Slopes(slopes = ifs)
            slope_obj.single_mask = self._subapdata.single_mask()
            slope_obj.display_map = self._subapdata.display_map
            slope_map =  slope_obj.get2d()
            slope_maps_x.append(np.ma.array(data = slope_map[0], mask = slope_mask))
            slope_maps_y.append(np.ma.array(data = slope_map[1], mask = slope_mask))
        
        self._slope_maps_y = np.ma.array(slope_maps_y)
        self._slope_maps_x = np.ma.array(slope_maps_x)
        
        
    def get_slope_maps(self):
        return self._slope_maps_x, self._slope_maps_y

    def display_slope_maps(self, mode_index):
            
            slope_map_x = self._slope_maps_x[mode_index]
            slope_map_y = self._slope_maps_y[mode_index]
            
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
            fig.suptitle(f"Mode index {mode_index}")
            fig.tight_layout()
            