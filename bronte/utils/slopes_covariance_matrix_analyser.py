from bronte.utils.slopes_vector_analyser import SlopesVectorAnalyser
import numpy as np 
import matplotlib.pyplot as plt


class SlopesCovariaceMatrixAnalyser():
    
    def __init__(self, subap_tag):
        
        self._subap_tag = subap_tag
        self._frame_cube = None
        self._slopes_cube = None
        self._slope_covariance_matrix = None
        self._sva = SlopesVectorAnalyser(subap_tag)
        self.Nsubap = self._sva._subapertures_set.n_subaps
        self.NpixperSub = self._sva._subapertures_set.np_sub    
    
    def set_slopes_from_slopes_cube(self, slopes_cube, pix_thr_ratio = 0, abs_pix_thr = 200):
        '''
        slope_cube: ndarray with shape (Nsteps, Nsubaps*2)
        '''
        self._slopes_cube = slopes_cube
        self._sva.reload_slope_pc(pix_thr_ratio, abs_pix_thr)
        self._rms_slopes_x_cube, self._rms_slopes_x_cube = self.get_rms_slopes_cube()
        self._compute_slopes_covariance_matrix()
        
    def set_slopes_from_frame_cube(self, frame_cube, pix_thr_ratio = 0, abs_pix_thr = 200):
        '''
        frame_cube: ndarray with shape (Nframes, frame_y_size, frame_x_size)
        '''
        self._frame_cube = frame_cube
        Nframes = frame_cube.shape[0]
        self._sva.reload_slope_pc(pix_thr_ratio, abs_pix_thr)
        slopes_list = []
        flux_per_sub_list = []
        for idx in np.arange(Nframes):
            slopes, flux_per_sub = self._sva.get_slopes_from_frame(frame_cube[idx], True)
            slopes_list.append(slopes)
            flux_per_sub_list.append(flux_per_sub)
        self._slopes_cube = np.array(slopes_list)
        self._flux_per_sub_cube = np.array(flux_per_sub_list)
        self._rms_slopes_x_cube, self._rms_slopes_y_cube = self.get_rms_slopes_cube()
        self._compute_slopes_covariance_matrix()
        
    def get_rms_slopes_cube(self):
        Nsteps = self._slopes_cube.shape[0]
        rms_slope_x_list = []
        rms_slope_y_list = []
        
        for idx in np.arange(Nsteps):
            rms_slope_x, rms_slope_y = self._sva.get_rms_slopes(self._slopes_cube[idx])
            rms_slope_x_list.append(rms_slope_x)
            rms_slope_y_list.append(rms_slope_y)
            
        return np.array(rms_slope_x_list), np.array(rms_slope_y_list)
    
    def _compute_slopes_covariance_matrix(self):
        self._slope_covariance_matrix = self._slopes_cube.T @ self._slopes_cube
    
    def get_slopes_covariace_matrix(self):
        return self._slope_covariance_matrix
    
    def display_rms_slopes(self):
        
        norm2pixel = 0.5*self.NpixperSub
        plt.figure()
        plt.clf()
        plt.plot(self._rms_slopes_x_cube * norm2pixel, '.-', label ='x')
        plt.plot(self._rms_slopes_y_cube * norm2pixel, '.-', label ='y')
        plt.grid('--', alpha = 0.3)
        plt.legend(loc = 'best')
        plt.xlabel('Steps')
        plt.ylabel('RMS Slopes [pixels]')
    
    def display_slope_covariance_matrix(self):
        
        plt.figure()
        plt.clf()
        plt.imshow(self._slope_covariance_matrix)
        plt.colorbar()