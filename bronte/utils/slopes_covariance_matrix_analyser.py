import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np
from specula.processing_objects.modalrec import Modalrec
from specula.data_objects.recmat import Recmat
from bronte.utils.slopes_vector_analyser import SlopesVectorAnalyser

import matplotlib.pyplot as plt
from bronte.startup import set_data_dir
from bronte.package_data import slope_offset_folder, reconstructor_folder
from astropy.io import fits


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
        self._pix_thr_ratio = pix_thr_ratio
        self._abs_pix_thr = abs_pix_thr
        self._sva.reload_slope_pc(pix_thr_ratio, abs_pix_thr)
        slopes_list = []
        flux_per_sub_list = []
        Neff_per_sub_list = []
        for idx in np.arange(Nframes):
            slopes, flux_per_sub = self._sva.get_slopes_from_frame(frame_cube[idx], True)
            slopes_list.append(slopes)
            flux_per_sub_list.append(flux_per_sub)
            #Neff_per_sub_list.append(Neff_per_sub)
        self._slopes_cube = np.array(slopes_list)
        self._flux_per_sub_cube = np.array(flux_per_sub_list)
        self._Neff_per_sub_cube = np.array(Neff_per_sub_list)
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
        St = self._slopes_cube - self._slopes_cube.mean(axis=0)
        self._slope_covariance_matrix = St.T @ St
    
    def get_slopes_covariace_matrix(self):
        return self._slope_covariance_matrix
    
    def _compute_average_slopes(self):
        self._average_slopes = self._slopes_cube.mean(axis = 0)
    
    def _compute_std_slopes(self):
        self._std_slopes = self._slopes_cube.std(axis = 0)
        
    def get_average_slopes(self):
        return self._average_slopes
    
    def get_std_slopes(self):
        return self._std_slopes
    
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
        
    def save_average_slopes_as_slope_offset(self, ftag):
        
        set_data_dir()
        file_name = slope_offset_folder() / (ftag + '.fits')
        hdr = fits.Header()
        hdr['SUB_TAG'] = self._subap_tag
        hdr['ABS_THR'] = self._abs_pix_thr
        hdr['REL_THR'] = self._pix_thr_ratio
        slope_offset = self._average_slopes
        fits.writeto(file_name, slope_offset, hdr)
        
    def load_reconstructor(self, ftag):
        
        recmat = Recmat.restore(reconstructor_folder() / (ftag + "_bronte_rec.fits"))
        #added factor 2 missed on IFs normalization
        N_pp = 2
        recmat.recmat = N_pp*recmat.recmat  
        Nmodes = recmat.recmat.shape[0]
        self._modalrec =  Modalrec(Nmodes, recmat=recmat)
        
    def compute_delta_modal_command(self):
        Nmodes = self._modalrec.recmat.recmat.shape[0]
        Nstep = self._slopes_cube.shape[0]
        self._delta_modal_cmd_in_nm = np.zeros((Nstep, Nmodes))
        
        for step in np.arange(Nstep):
            self._delta_modal_cmd_in_nm[step, :] = np.dot(self._modalrec.recmat.recmat,self._slopes_cube[step,:])
    
    def get_delta_modal_command(self):
        return self._delta_modal_cmd_in_nm        
    
    @staticmethod
    def load_slope_offset(ftag):
        set_data_dir()
        file_name = slope_offset_folder() / (ftag + '.fits')
        hduList = fits.open(file_name)
        hdr = fits.getheader(file_name)
        abs_thr = hdr['ABS_THR']
        rel_thr = hdr['REL_THR']
        slope_offset = hduList[0].data
        return slope_offset, abs_thr, rel_thr