import numpy as np 
from bronte.startup import startup
from bronte.utils.raw_strehl_ratio_computer import StrehlRatioComputer
from bronte.utils.data_cube_cleaner import DataCubeCleaner
import time
from astropy.io import fits
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit


class SharpPsfOnCamera():
    
    RESCALING_INDEX2AVOID_PISTON = 1
    RESCALING_INDEX2START_FROM_Z2 = 2 # so that arr[0] corresponds to Z2
    SLM_RESPONSE_TIME_SEC = 0.010
    
    def __init__(self, noll_index_list2correct=[4,5,6,7,8,9,10,11]):
        
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
    
    def sharp(self, amp_span = np.linspace(-2e-6, 2e-6, 5), texp_in_ms = 7, Nframe2average = 20, useGaussFit = False):
        
        self._compute_au_dl_psf()
        
        if self._master_dark is None:
            self._master_dark = np.zeros(self._cam.shape())
        
        if self._zc_offset is None:
            self._zc_offset = self._sr.get_zernike_coefficients_from_numpy_array(np.zeros(3))
        
        self._texp = texp_in_ms
        self._cam.setExposureTime(texp_in_ms)
        
        self._amp_span = amp_span
        
        self._measured_sr = np.zeros((self._N_zernike_modes, len(amp_span)))
        self._measured_i_in_roi = np.zeros((self._N_zernike_modes, len(amp_span)))
        self._best_coeff = np.zeros(self._N_zernike_modes)
        self._best_coeff_err = np.zeros(self._N_zernike_modes)
        
        zc2explore = self._sr.get_zernike_coefficients_from_numpy_array(
            np.zeros(self._N_zernike_modes))
        
        zc2apply = zc2explore + self._zc_offset
        
        command  = self._sr.m2c(zc2apply, applyTiltUnderMask=True)
        #command = self._sr.reshape_map2vector(wfz.toNumpyArray())
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
                self._measured_i_in_roi[idx_n, idx_m] = roi_master.sum()
            if useGaussFit == False:
                best_amplitude = self._get_best_amplitude(self._measured_sr[idx_n, :])
                best_err = 5e-9
            else:
                best_amplitude, best_err = self._get_best_amplitude_gaussfit(
                    self._amp_span,
                    self._measured_sr[idx_n, :],
                    self._measured_i_in_roi[idx_n, :],
                    ron_adu=2.4,
                    gain_e_per_adu=3.5409
                )
            print(f"\t -> best coeff (fit gauss): {best_amplitude:.3e} ± {best_err if np.isfinite(best_err) else np.nan:.3e} [m RMS]")
            
            self._best_coeff[idx_n] = best_amplitude
            self._best_coeff_err[idx_n] = best_err
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
        hdr['ROI_DIM'] = self._size
        hdr['DL_FLUX'] = self._sr_computer._total_dl_flux
        
        fits.writeto(fname, self._ncpa_zc.toNumpyArray(), hdr)
        fits.append(fname, self._zc_offset.toNumpyArray())
        fits.append(fname, self._z_modes_indexes_to_correct)
        fits.append(fname, self._measured_sr)
        fits.append(fname, self._amp_span)
        fits.append(fname, self._compensated_psf)
        fits.append(fname, self._uncompensated_psf)
        fits.append(fname, self._au_dl_psf)
        fits.append(fname,  self._measured_i_in_roi)
        fits.append(fname, self._best_coeff)
        fits.append(fname, self._best_coeff_err)
        
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
        measured_i_in_roi = hduList[8].data
        return texp, tot_dl_flux, ncpa, zc_offset, corr_z_modes_index, measured_sr, amp_span, comp_psf, uncomp_psf,au_dl_psf,measured_i_in_roi 
    
    @staticmethod
    def load_ncpa_gauss_fit(fname):
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
        measured_i_in_roi = hduList[8].data
        best_coeff = hduList[9].data
        err_best_coeff = hduList[10].data
        return texp, tot_dl_flux, ncpa, zc_offset, corr_z_modes_index, measured_sr, amp_span, comp_psf, uncomp_psf,au_dl_psf,measured_i_in_roi,best_coeff,err_best_coeff 
    
    
    
    def _get_best_amplitude_gaussfit(
        self,
        amp_span,                # array 1D delle ampiezze comandate (m RMS)
        sr_vector,               # SR misurati corrispondenti (stessa lunghezza)
        i_roi_vector,            # somma ADU nella ROI per ciascun punto (stessa lunghezza)
        ron_adu=2.4,             # RON in ADU (dato: 2.4 ADU)
        gain_e_per_adu=3.5409,   # gain in e-/ADU (dato)
        return_details=False
    ):
        """
        Stima il best coeff con fit gaussiano su -ln(SR) e calcola l'errore 1-sigma.
        Richiede che self._au_dl_psf e self._size siano definiti (sono settati in sharp()).
        Ritorna: best_amp [m RMS], best_amp_err [m RMS], (opz.) dizionario con dettagli del fit.
        """

        amp_span   = np.asarray(amp_span, float)
        SR         = np.asarray(sr_vector, float)
        A_ROI      = np.asarray(i_roi_vector, float)

        # --- controllo minimi requisiti
        ok = np.isfinite(amp_span) & np.isfinite(SR) & np.isfinite(A_ROI) & (SR > 0) & (A_ROI > 0)
        if ok.sum() < 3:
            # fallback: usa massimo discreto (senza errore)
            best_amp = amp_span[np.nanargmax(SR)]
            return (best_amp, np.nan, {}) if return_details else (best_amp, np.nan)

        amp_span = amp_span[ok]
        SR       = SR[ok]
        A_ROI    = A_ROI[ok]

        # --- costanti/parametri da classe: rapporto picco/somma della PSF DL nella ROI
        #     peak_DL = r * A_ROI, con r costante per geometria della ROI e pupil
        if not hasattr(self, "_au_dl_psf"):
            # in caso non fosse stato chiamato _compute_au_dl_psf:
            self._compute_au_dl_psf()
        dl_roi = self._au_dl_psf
        r = float(dl_roi.max() / dl_roi.sum())  # rapporto picco/somma del riferimento DL nella ROI

        # --- grandezze utili
        g = float(gain_e_per_adu)
        ron = float(ron_adu)
        Nroi = int(self._size) * int(self._size)

        # --- incertezze: derivazione vedi messaggio precedente
        # peak_DL = r * A_ROI ;  sigma_Aroi in ADU
        sigma_Aroi = np.sqrt(A_ROI / g + Nroi * (ron / g)**2)
        peak_DL = r * A_ROI
        sigma_peakDL = r * sigma_Aroi

        # A_p = SR * peak_DL
        A_p = SR * peak_DL
        sigma_Ap = np.sqrt(A_p / g + (ron / g)**2)

        # sigma_SR via propagazione: SR * sqrt( (σAp/Ap)^2 + (σDl/Dl)^2 )
        # salvaguardia per divisioni:
        eps = np.finfo(float).eps
        sigma_SR = SR * np.sqrt(
            (sigma_Ap / (A_p + eps))**2 +
            (sigma_peakDL / (peak_DL + eps))**2
        )

        # trasformazione log: y = -ln(SR), σ_y = σ_SR / SR
        y = -np.log(SR)
        sigma_y = sigma_SR / (SR + eps)

        # filtra eventuali outlier numerici
        ok2 = np.isfinite(y) & np.isfinite(sigma_y) & (sigma_y > 0)
        if ok2.sum() < 3:
            best_amp = amp_span[np.nanargmax(SR)]
            return (best_amp, np.nan, {}) if return_details else (best_amp, np.nan)

        c = amp_span[ok2]
        y = y[ok2]
        sy = sigma_y[ok2]

        # --- inizializzazione: usa uno spline per un guess di mu (come la tua funzione)
        try:
            from scipy.interpolate import CubicSpline
            damp = max((c.max()-c.min())/1000.0, 1e-10)
            c_dense = np.arange(c.min(), c.max()+damp, damp)
            # per spline su y servono SR (o y); preferiamo SR per cercare il massimo
            cs_SR = CubicSpline(c, np.exp(-y), bc_type='natural')
            SR_dense = cs_SR(c_dense)
            mu0 = c_dense[np.argmax(SR_dense)]
        except Exception:
            mu0 = c[np.argmax(np.exp(-y))]

        # curvatura iniziale a0 > 0 (stima dalla parabola sui 3 punti migliori)
        try:
            idx_top = np.argsort(-np.exp(-y))[:3]
            cc = c[idx_top]
            ssr = np.exp(-y[idx_top])
            # fit locale parabola: s(c) ~ A c^2 + B c + C -> traduci in y = -ln s
            # per semplicità imposta a0 da ampiezza media
            width = max(np.std(cc), (c.max()-c.min())/10.0)
            a0 = 1.0 / (width**2 + 1e-30)
        except Exception:
            a0 = 1.0 / ( (c.max()-c.min())**2 + 1e-30 )

        b0 = float(np.min(y))

        def model(c_, a, mu, b):
            return a*(c_-mu)**2 + b

        # bounds: a>0 per garantire convessità di -ln SR
        bounds = ([1e-16, c.min()-10*( np.ptp(c)), -np.inf],
                  [np.inf,  c.max()+10*(np.ptp(c)),  np.inf])

        p0 = [a0, mu0, b0]
        

        try:
            popt, pcov = curve_fit(
                model, c, y, p0=p0, sigma=sy, absolute_sigma=True, bounds=bounds, maxfev=10000
            )
            a_hat, mu_hat, b_hat = popt
            mu_err = float(np.sqrt(max(pcov[1,1], 0.0)))
            # sanity: se a troppo piccolo (piatto), l'errore esplode -> fallback su massimo discreto
            if not np.isfinite(mu_err) or mu_err > 10.0 * np.ptp(c):
                best_amp = c[np.nanargmax(np.exp(-y))]
                return (best_amp, np.nan, {}) if return_details else (best_amp, np.nan)
            details = {
                "a": a_hat, "mu": mu_hat, "b": b_hat,
                "cov": pcov, "c_used": c, "y_used": y, "sy_used": sy,
                "r_peak_over_sum": r
            }
            return (float(mu_hat), float(mu_err), details) if return_details else (float(mu_hat), float(mu_err))
        except Exception:
            # fallback robusto: massimo dello spline su SR (senza errore)
            best_amp = c[np.nanargmax(np.exp(-y))]
            return (best_amp, np.nan, {}) if return_details else (best_amp, np.nan)