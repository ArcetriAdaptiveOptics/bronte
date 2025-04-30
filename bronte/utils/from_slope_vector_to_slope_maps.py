import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np
from specula.data_objects.slopes import Slopes
from specula.data_objects.subap_data import SubapData
from bronte.package_data import subaperture_set_folder
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class SlopeVectorTo2DMap():
    
    def __init__(self, subap_tag):
        
        self._subapdata = self._load_subaperture_set(subap_tag)
        self._slope_mask = self._get_slope_mask()
        
    @staticmethod
    def _load_subaperture_set(subap_tag):
    
        subapdata = SubapData.restore_from_bronte(
            subaperture_set_folder() / (subap_tag + ".fits"))
        return subapdata
    
    def _get_slope_mask(self):
    
        idl_slope_mask = self._subapdata.single_mask()
        slope_mask = np.ones(idl_slope_mask.shape) - idl_slope_mask
        return slope_mask 
    
    def get2Dmaps_from_slopes_vector(self, slope_vector):
        
        slope_obj = Slopes(slopes = slope_vector)
        slope_obj.single_mask = self._subapdata.single_mask()
        slope_obj.display_map = self._subapdata.display_map
        slope_map =  slope_obj.get2d()
        slope_map_x = np.ma.array(data = slope_map[0], mask = self._slope_mask)
        slope_map_y = np.ma.array(data = slope_map[1], mask = self._slope_mask)
        
        return slope_map_x, slope_map_y
    
    def display_2Dmap_from_slope_vector(self, slope_vector):
            
        slope_map_x, slope_map_y = self.get2Dmaps_from_slopes_vector(slope_vector)
        
        fig, axs = plt.subplots(1, 2, sharex = True,
                                 sharey = True)
        
        axs[0].set_title('Slope Map X')
        im_map_x = axs[0].imshow(slope_map_x)
        divider_x = make_axes_locatable(axs[0])
        cax_x = divider_x.append_axes("right", size = "5%", pad = 0.15)  
        fig.colorbar(im_map_x, cax=cax_x, label = 'a.u.')
        
        axs[1].set_title('Slope Map Y')
        im_map_y = axs[1].imshow(slope_map_y)
        divider_y = make_axes_locatable(axs[1])
        cax_y = divider_y.append_axes("right", size = "5%", pad = 0.15)
        fig.colorbar(im_map_y, cax=cax_y, label = 'a.u.')
        fig.subplots_adjust(wspace=0.5)
        fig.tight_layout()