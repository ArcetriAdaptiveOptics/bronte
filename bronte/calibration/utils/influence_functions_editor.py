import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np
from specula.data_objects.ifunc import IFunc
from bronte.calibration.utils.display_ifs_map import DisplayInfluenceFunctionsMap
from bronte.calibration.utils.kl_modal_base_generator import KarhunenLoeveGenerator
from bronte.startup import set_data_dir
from bronte.package_data import ifs_folder
from scipy.interpolate import RegularGridInterpolator

class InfluenceFucntionEditor():
    
    def __init__(self, ifs_tag):
        
        self._ifs_tag = ifs_tag
        self._ifunc = KarhunenLoeveGenerator.load_modal_ifs(self._ifs_tag)
        self._dispIFs = DisplayInfluenceFunctionsMap(None, self._ifunc)
        self._pupil_mask_idl = self._ifunc.mask_inf_func
        
        self._edited_pupil_mask_idl = None
        self._edited_np_ifs = None
        
    def load_sigular_values(self):
        
        self._s1, self._s2 = KarhunenLoeveGenerator.load_singular_values(self._ifs_tag)
    
    def display_singular_values(self):
        import matplotlib.pyplot as plt
        
        plt.semilogy(self._s1, '.-', label='IF Covariance')
        plt.semilogy(self._s2, '.-', label='Turbulence Covariance')
        plt.xlabel('Mode number')
        plt.ylabel('Singular value')
        plt.title('Singular values of covariance matrices')
        plt.legend(loc='best')
        plt.grid('--', alpha=0.3)
        
    
    def remove_modes(self, Nmodes = 10):

        self._Nmodes = Nmodes
        full_np_ifs = self._ifunc.influence_function.T.copy()
        self._filtered_np_ifs = full_np_ifs[:Nmodes,:]
        
    def rescale_ifs(self, new_frame_size = 545*2):
        
        self._edited_pupil_mask_idl = self._rescale_pupil_mask(
            self._pupil_mask_idl,
            new_frame_size,
            int_method = 'linear')
        
        valid_pup_points = self._edited_pupil_mask_idl[self._edited_pupil_mask_idl==1].sum()
        Nmodes = self._filtered_np_ifs.shape[0]
        self._edited_np_ifs = np.zeros((Nmodes, valid_pup_points))
        original_pup_size = self._pupil_mask_idl.shape[0]
        
        for idx in range(Nmodes):
            print("Rescaling mode %d" %idx)
            ifs_vector = self._filtered_np_ifs[idx]
            masked_array_ifs_map = self._get_2D_masked_array(ifs_vector, original_pup_size)
            rescaled_ifs_map = self._interpolate_2d_array(
                masked_array_ifs_map,
                new_frame_size,
                self._edited_pupil_mask_idl,
                int_method='slinear')
            
            self._edited_np_ifs[idx] = rescaled_ifs_map[self._edited_pupil_mask_idl == 1]
            
            
    def _rescale_pupil_mask(self, original_pupil_mask, new_size, int_method = 'linear'):
        
        interpolated_pupil_mask =  self._interpolate_square_pupil_mask(original_pupil_mask, new_size, int_method)
        rescaled_pupil = (interpolated_pupil_mask > 0.5).astype(int)
        return rescaled_pupil
    
    def _interpolate_square_pupil_mask(self, original_pupil_mask, new_size, int_method='linear'):
        '''
        original_pupil_mask: is a 2D array of 0 and 1,
        out and in pupil rispectively
        new_size: scalar, size of the interpolated frame
        '''
        x_original = np.linspace(0, 1, original_pupil_mask.shape[1])
        y_original = np.linspace(0, 1, original_pupil_mask.shape[0])
        original_grid = (y_original, x_original)
        
        x_new = np.linspace(0, 1, new_size)
        y_new = np.linspace(0, 1, new_size)
        x_new_grid, y_new_grid = np.meshgrid(x_new, y_new)
        new_points = np.column_stack([y_new_grid.ravel(), x_new_grid.ravel()])
        
        interpolator = RegularGridInterpolator(original_grid, original_pupil_mask, method=int_method)
        interpolated_pupil = interpolator(new_points).reshape(new_size, new_size)
        
        return interpolated_pupil
    
    def _interpolate_2d_array(self, original_2Darray, new_size, int_mask=None, int_method = 'slinear'):
        '''
        original_2Darray: 2d array  
        new_size: scalar, size of the interpolated frame
        int_mask: 2d array of 0 and 1 like pupil_mask(idl) that identify where thr
        interpolation is performed, if None the interpolation is computed in all
        the points of the rescaled array
        '''
        x_original = np.linspace(0, 1, original_2Darray.shape[1])
        y_original = np.linspace(0, 1, original_2Darray.shape[0])
        original_grid = (y_original, x_original)
        
        x_new = np.linspace(0, 1, new_size)
        y_new = np.linspace(0, 1, new_size)
        x_new_grid, y_new_grid = np.meshgrid(x_new, y_new)
        
        if int_mask.shape != x_new_grid.shape:
            raise ValueError("La pupil mask non ha la stessa shape della nuova griglia.")
        
        if int_mask is not None:
            valid_mask = int_mask.astype(bool)
        else:
            valid_mask = np.ones((new_size, new_size))
            
        valid_coords = np.column_stack([
        y_new_grid[valid_mask],  # y
        x_new_grid[valid_mask],  # x
        ])
        
        interpolator = RegularGridInterpolator(original_grid, original_2Darray, method=int_method)
        # Interpola solo nei punti validi
        interpolated_values = interpolator(valid_coords)
        
        interpolated_full = np.zeros((new_size, new_size))#_like(self._edited_pupil_mask_idl, dtype=float)
        interpolated_full[valid_mask] = interpolated_values

        return interpolated_full
        

    def _get_2D_masked_array(self, ifs_vector, frame_size):
        
        ifs_1d = ifs_vector
        masked_array_ifs_map = np.ma.array(
            data = np.zeros((frame_size, frame_size)),
            mask = 1 - self._pupil_mask_idl)
        masked_array_ifs_map[masked_array_ifs_map.mask == False] = ifs_1d
        
        return masked_array_ifs_map
    
    
    def save_filtered_ifs(self, ftag):

        pupil_mask_idl = self._edited_pupil_mask_idl
        np_ifs = self._edited_np_ifs
        
        if self._edited_pupil_mask_idl is None:
            pupil_mask_idl = self._pupil_mask_idl
        if self._edited_np_ifs is None:
            np_ifs = self._filtered_np_ifs
        
        set_data_dir()
        edited_pupil_mask_idl = pupil_mask_idl 
        ifunc_obj = IFunc(
            ifunc = np_ifs,
            mask = edited_pupil_mask_idl)
        fname  = ifs_folder() / (ftag + '.fits')
        ifunc_obj.save(fname)
        