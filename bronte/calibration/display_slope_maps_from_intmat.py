import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np
from specula.data_objects.slopes import Slopes
from specula.data_objects.intmat import Intmat
from specula.data_objects.subap_data import SubapData
from bronte.package_data import reconstructor_folder, subaperture_set_folder
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class DisplaySlopeMapsFromInteractionMatrix():
    
    def __init__(self, intmat_tag, subap_tag, pp_vect_in_nm = None):
        
        self._subapdata = self._load_subaperture_set(subap_tag)
        self._intmat = self._load_intmat(intmat_tag)
        self._pp_vec_in_nm = pp_vect_in_nm
        self._load_slopes_from_intmat()
        
        
    @staticmethod    
    def _load_intmat(intmat_tag):
        
        file_name = reconstructor_folder() / (intmat_tag + '_bronte_im.fits')
        int_mat = Intmat.restore(file_name)
        Npp = 2
        int_mat._intmat = int_mat._intmat / Npp
        return int_mat
    
    @staticmethod
    def _load_subaperture_set(subap_tag):
    
        subapdata = SubapData.restore_from_bronte(
            subaperture_set_folder() / (subap_tag + ".fits"))
        return subapdata
    
    def _load_slopes_from_intmat(self):
    
        n_modes = self._intmat._intmat.shape[0]
        slope_maps_x = []
        slope_maps_y = []
        idl_slope_mask = self._subapdata.single_mask()
        self._slope_mask = np.ones(idl_slope_mask.shape) - idl_slope_mask
        ifs_list = []
        for idx in range(n_modes):
            if self._pp_vec_in_nm is not None:
                ifs = self._intmat._intmat[idx]*self._pp_vec_in_nm[idx]
            else:
                ifs = self._intmat._intmat[idx]
            ifs_list.append(ifs)
            slope_obj = Slopes(slopes = ifs)
            slope_obj.single_mask = self._subapdata.single_mask()
            slope_obj.display_map = self._subapdata.display_map
            slope_map =  slope_obj.get2d()
            slope_maps_x.append(np.ma.array(data = slope_map[0], mask = self._slope_mask))
            slope_maps_y.append(np.ma.array(data = slope_map[1], mask = self._slope_mask))
        
        self._slope_maps_y = np.ma.array(slope_maps_y)
        self._slope_maps_x = np.ma.array(slope_maps_x)
        self._ifs = np.array(ifs_list)
        
    def get_slope_maps(self):
        return self._slope_maps_x, self._slope_maps_y
    
    def display_slope_maps(self, mode_index):
            
        slope_map_x = self._slope_maps_x[mode_index]
        slope_map_y = self._slope_maps_y[mode_index]
        
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
        fig.suptitle(f"Mode index {mode_index}")
        fig.tight_layout()
    
    def _compute_full_slopes_map(self, size = 45, ncols = 3, nrows = 2):
        #Nmodes = self._intmat._intmat.shape[0] 
        Nx = 2 * size * ncols
        Ny = size * nrows
        full_map = np.zeros((Ny,Nx))
        full_map_mask = np.zeros((Ny,Nx))
        
        idx = 0
        # Nrows = nrows
        # Ncols = ncols*2
        for row in range(nrows):
            for col in range(ncols):

                full_map[row*size:(row+1)*size, (col*2)*size:(2*col+1)*size] = self._slope_maps_x[idx, 8:8+size, 14:14+size]
                full_map[row*size:(row+1)*size, (2*col+1)*size:(2*col+2)*size] = self._slope_maps_y[idx, 8:8+size, 14:14+size]
                full_map_mask[row*size:(row+1)*size, (col*2)*size:(2*col+1)*size] = self._slope_mask[8:8+size, 14:14+size]
                full_map_mask[row*size:(row+1)*size, (2*col+1)*size:(2*col+2)*size] = self._slope_mask[8:8+size, 14:14+size]
                idx+=1
                
        self._full_slopes_map = np.ma.array(data=full_map, mask=full_map_mask)
        
    def display_all_slope_maps(self, size = 45, ncols = 3, nrows = 2, title = None):
               
        self._compute_full_slopes_map(size, ncols, nrows)
        plt.figure()
        plt.clf()
        plt.imshow(self._full_slopes_map)
        plt.colorbar()
        if title is not None:
            plt.title(title)