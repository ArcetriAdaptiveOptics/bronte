import numpy as np 
from bronte.startup import startup
from bronte.utils.raw_strehl_ratio_computer import StrehlRatioComputer
from bronte.utils.data_cube_cleaner import DataCubeCleaner
import time
from astropy.io import fits
from scipy.interpolate import CubicSpline


class SharpPsfOnCamera():
    
    RESCALING_INDEX2AVOID_PISTON = 1
    RESCALING_INDEX2START_FROM_Z2 = 2 # so that arr[0] corresponds to Z2
    SLM_RESPONSE_TIME_SEC = 0.005
    
    def __init__(self, noll_index_list2correct=[4,5,7,11]):
        
        self._factory = startup()
        self._slm = self._factory.deformable_mirror
        self._cam = self._factory.psf_camera
        self._sr = self._factory.slm_rasterizer
        
        self._z_modes_indexes_to_correct = np.array(noll_index_list2correct)
        self._j_noll_max = self._z_modes_indexes_to_correct.max()
        
        self._N_zernike_modes =  self._j_noll_max - self.RESCALING_INDEX2AVOID_PISTON
        self._zc_offset = None
        self._master_dark = None
        self._texp = None
        self._yc_roi = None
        self._xc_roi = None
        self._size = None 
        self._sr_interp_func  = []
        self._sr_computer = StrehlRatioComputer()
        self._cleaner = DataCubeCleaner()
        
    def acquire_master_dark(self, texp_in_ms = 7, Nframe2average = 20):
        
        self._texp  = texp_in_ms
        self._cam.setExposureTime(texp_in_ms)
        
        raw_dark_dataCube = self._cam.getFutureFrames(Nframe2average).toNumpyArray()
        
        self._master_dark = np.median(raw_dark_dataCube, axis = -1)
    
    def load_master_dark(self, master_dark):
        self._master_dark = master_dark
        
    def load_zc_offset(self, zernike_coeff_np_array):
        """
        start from tip (Z2), thus arr[0]=c2, arr[1]=c3, arr[2]=c4, ...
        """
        self._zc_offset = self._sr.get_zernike_coefficients_from_numpy_array(zernike_coeff_np_array)
    
    def reset_zc_offset(self):
        self._zc_offset = None
        
    def get_master_dark(self):
        return self._master_dark
    
    def get_zc_offset(self):
        return self._zc_offset
    
    def define_roi(self, yc, xc, size = 60):
        
        self._yc_roi = yc
        self._xc_roi = xc
        self._size = size
    
    def sharp(self, amp_span = np.linspace(-2e-6, 2e-6, 5), texp_in_ms = 7, Nframe2average = 20):
        
        self._compute_au_dl_psf()
        
        if self._master_dark is None:
            self._master_dark = np.zeros(self._cam.shape())
        
        if self._zc_offset is None:
            self._zc_offset = self._sr.get_zernike_coefficients_from_numpy_array(np.zeros(3))
        
        self._texp = texp_in_ms
        self._cam.setExposureTime(texp_in_ms)
        
        self._amp_span = amp_span
        
        self._measured_sr = np.zeros((self._N_zernike_modes, len(amp_span)))
        
        zc2explore = self._sr.get_zernike_coefficients_from_numpy_array(
            np.zeros(self._N_zernike_modes))
        
        zc2apply = zc2explore + self._zc_offset
        
        wfz  = self._sr.zernike_coefficients_to_raster(zc2apply)
        command = self._sr.reshape_map2vector(wfz.toNumpyArray())
        self._slm.set_shape(command)
        time.sleep(self.SLM_RESPONSE_TIME_SEC)
        
        self._uncompensated_psf = self.get_psf_in_roi(Nframe2average)
        
        # for loop for each mode
        for j in self._z_modes_indexes_to_correct:
            
            idx_n = j - self.RESCALING_INDEX2START_FROM_Z2
            print("noll index %d (array index_n: %d)\n"%(j,idx_n))
            zc_np_array = zc2apply.toNumpyArray().copy()
            amp = zc_np_array[idx_n]
            #print("zc_np_array:")
            #print(zc_np_array)
            zc_array_temp = zc_np_array.copy()
            
            # for loop to inject a different z_coeff for the j mode 
            for idx_m, delta_amp in enumerate(self._amp_span):
                
                zc_array_temp[idx_n] = amp + delta_amp
                zc2apply_temp = self._sr.get_zernike_coefficients_from_numpy_array(zc_array_temp)
                print("\t index_m : %d ; app_coeff : %f" %(idx_m,zc_array_temp[idx_n]))
                #print("\t zc2appay_temp:",zc2apply_temp.toNumpyArray())
                
                wfz  = self._sr.zernike_coefficients_to_raster(zc2apply_temp)
                command = self._sr.reshape_map2vector(wfz.toNumpyArray())
                self._slm.set_shape(command)
                time.sleep(self.SLM_RESPONSE_TIME_SEC)
                roi_master = self.get_psf_in_roi(Nframe2average)
                self._measured_sr[idx_n, idx_m]  = self._get_sr(roi_master)
                
            best_amplitude = self._get_best_amplitude(self._measured_sr[idx_n, :])
            
            zc2apply.toNumpyArray()[idx_n] = best_amplitude
        
        self._ncpa_zc = zc2apply
        ncpa_wfz = self._sr.zernike_coefficients_to_raster(self._ncpa_zc)
        print("NCPA\n")
        print(self._ncpa_zc)
        
        command = self._sr.reshape_map2vector(ncpa_wfz.toNumpyArray())
        self._slm.set_shape(command)
        time.sleep(self.SLM_RESPONSE_TIME_SEC)
        self._compensated_psf = self.get_psf_in_roi(Nframe2average)
    
    def get_ncpa(self):
        return self._ncpa_zc
    
    def _get_sr(self, image):
        return self._sr_computer.get_SR_from_image(image, enable_display=False)
    
    def get_psf_in_roi(self, Nframes):
        
        raw_dataCube = self._cam.getFutureFrames(Nframes).toNumpyArray()
        master_image = self._cleaner.get_master_from_rawCube(raw_dataCube, self._master_dark)
                
        hsize = int(np.round(self._size*0.5))
        roi_master = master_image[self._yc_roi-hsize:self._yc_roi+hsize,
                                          self._xc_roi-hsize:self._xc_roi+hsize]
        return roi_master
    
    def _compute_au_dl_psf(self):
        hsize = int(np.round(self._size * 0.5))
        self._au_dl_psf= self._sr_computer._dl_psf[self._yc_roi-hsize:self._yc_roi+hsize,
                                  self._xc_roi-hsize:self._xc_roi+hsize]
        
    
    def _get_best_amplitude(self, sr_vector):
      
        damp = 5e-9#self._amp_span.max()*0.01
        amps = np.arange(self._amp_span.min(), self._amp_span.max() + damp, damp)
        sr_interp_functon = CubicSpline(self._amp_span, sr_vector, bc_type='natural')
        sr_func  = sr_interp_functon(amps)
        max_idx  = np.where(sr_func == sr_func.max())[0][0]
        best_amp = amps[max_idx]
        return best_amp
    
    def save_ncpa(self, fname):
        
        
        hdr = fits.Header()
        hdr['T_EX_MS'] = self._texp
        #hdr['ROI_DIM'] = self._size
        hdr['DL_FLUX'] = self._sr_computer._total_dl_flux
        
        fits.writeto(fname, self._ncpa_zc.toNumpyArray(), hdr)
        fits.append(fname, self._zc_offset.toNumpyArray())
        fits.append(fname, self._z_modes_indexes_to_correct)
        fits.append(fname, self._measured_sr)
        fits.append(fname, self._amp_span)
        fits.append(fname, self._compensated_psf)
        fits.append(fname, self._uncompensated_psf)
        fits.append(fname, self._au_dl_psf)
        
    @staticmethod
    def load_ncpa(fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        texp = header['T_EX_MS']
        tot_dl_flux = header['DL_FLUX']
        ncpa = hduList[0].data
        zc_offset = hduList[1].data
        corr_z_modes_index = hduList[2].data
        measured_sr = hduList[3].data
        amp_span = hduList[4].data
        comp_psf = hduList[5].data
        uncomp_psf = hduList[6].data
        au_dl_psf = hduList[7].data
        return texp, tot_dl_flux, ncpa, zc_offset, corr_z_modes_index, measured_sr, amp_span, comp_psf, uncomp_psf,au_dl_psf 
        