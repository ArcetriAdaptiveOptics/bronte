import numpy as np
from numpy import arange
from bronte.utils.modal_base_generators.orthonormal_base_computer \
    import OrthonormalBaseComputer 
from functools import cache 

class CovarianceMatrixComputer():
    '''
    Computes the covariance matrix Ca as ft{F}^T psd ft{F}*
    with F the orthonormal base, where each column is a mode of the base
    psd is the pawer spectrum of the turbulence
    '''
    def __init__(self, orthonormal_base, frame_shape =(1152, 1920), psd = None):
        
        OrthonormalBaseComputer._check_if_is_orthonormal(orthonormal_base)
        self._base = orthonormal_base
        self._frame_shape = frame_shape
        if psd is not None:
            self._psd  = psd
    
    def _compute_base_fft(self, Npad = 2):
        
        nModes = self._base.shape[1]  # num of columns
        #Npoints = self._base.shape[0] # num of rows
        
        padded_frame_size = np.max(self._frame_shape) * Npad
        padded_points = padded_frame_size * padded_frame_size
        base_in_freq_domain = np.zeros((padded_points, nModes), dtype=complex)
        
        for idx in np.arange(nModes):
            
            mode = self._base[:, idx].reshape(self._frame_shape)
            
            padded_frame_size = np.max(self._frame_shape) * Npad
            self._padded_frame_shape = (padded_frame_size, padded_frame_size)
            padded_mode = np.zeros(self._padded_frame_shape, dtype=complex)
            padded_mode[0 : mode.shape[0], 0 : mode.shape[1]] = mode
        
            fft_mode = np.fft.fftshift(np.fft.fft2(padded_mode))
            
            base_in_freq_domain[:,idx] = fft_mode.flatten()
        
        return base_in_freq_domain
    
    def _compute_covariance_matrix(self):
        return 0
    
    @cache
    def get_covariance_matrix(self):
        self._cov_mat = self._compute_covariance_matrix()
        return self._cov_mat