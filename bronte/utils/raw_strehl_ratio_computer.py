import numpy as np 
from arte.types.mask import CircularMask
from astropy.modeling import models, fitting

class StrehlRatioComputer():
    
    RAD2ARCSEC = 180/np.pi*3600
    PUPIL_RADIUS = 545
    
    def __init__(self):
        
        self._pupil_diameter = 2*self.PUPIL_RADIUS*9.2e-6
        self._wl = 633e-9
        self._telescope_focal_length = 250e-3
        self._ccd_pixel_size = 4.65e-6
        self._pixel_scale_in_arcsec = self._ccd_pixel_size/self._telescope_focal_length * self.RAD2ARCSEC
        self._dl_size_in_arcsec = self._wl / self._pupil_diameter * self.RAD2ARCSEC 
        self._dl_size_in_pixels = self._dl_size_in_arcsec / self._pixel_scale_in_arcsec
        self._compute_dl_psf()
        
    def _compute_dl_psf(self):
        
        Npix = 200
        pupil = np.zeros((Npix,Npix))
        pupil_radius_in_pix = Npix/2
        
    
        pupil_mask_obj = CircularMask(
            frameShape=pupil.shape,
            maskRadius=pupil_radius_in_pix,
            )
    
        pupil_mask = pupil_mask_obj.mask()
        phase = np.ma.array(pupil, mask = pupil_mask)
    
        # computing transmitted electric field
        Ut = 1 * np.exp(1j * phase)
        Ut.fill_value = 0
        Ut.data[Ut.mask == True] = 0
    
       
        # padding transmitted electric field to match resulting px scale with the instrument pixel scale
        Npad = self._dl_size_in_pixels #(self._wl / self._pupil_diameter  * self.RAD2ARCSEC) /  self._pixel_scale_in_arcsec
        #print("Pupil padding %g" % Npad)
        padded_frame_size = int(np.round(Npix * Npad))
        padded_Ut = np.zeros((padded_frame_size, padded_frame_size), dtype=complex)
        padded_Ut[0 : Ut.shape[0], 0 : Ut.shape[1]] = Ut   
    
        #computing psf
        self._dl_psf = np.abs(np.fft.fftshift(np.fft.fft2(padded_Ut)))**2
        self._dl_psf_scale_in_arcsec = self._wl / self._pupil_diameter / Npad * self.RAD2ARCSEC
        self._total_dl_flux = self._dl_psf.sum()
        
        #estimating psf parameters from Airy fitting
        amp = self._dl_psf.max()
        yc = np.where(self._dl_psf == amp)[0][0]
        xc = np.where(self._dl_psf == amp)[1][0]
        airy_radius_in_pixel =  (1.22*self._wl/self._pupil_diameter*self._telescope_focal_length)/self._ccd_pixel_size
        
        model_airy = models.AiryDisk2D(amp, xc, yc, airy_radius_in_pixel)
        fitter = fitting.LevMarLSQFitter()
        
        roi_shape = self._dl_psf.shape
        y, x = np.mgrid[:roi_shape[0], :roi_shape[1]]
        
        best_fit_airy = fitter(model_airy, x, y, z = self._dl_psf)
        self._fitted_dl_max_au = best_fit_airy.parameters[0]
        self._fitted_dl_fwhm_in_pixel  = 1.028 * best_fit_airy.parameters[-1]/(1.22)
    
    def get_SR_from_image(self, image, enable_display = False):
        
        hsize = int(np.round(image.shape[0]*0.5))
        total_measured_flux = image.sum()
        normalized_dl_psf = self._dl_psf * total_measured_flux/self._total_dl_flux
        center=int(np.round(normalized_dl_psf.shape[0] * 0.5))
        normalized_dl_psf_roi = normalized_dl_psf[center-hsize:center+hsize,
                                                  center-hsize:center+hsize]
        #TODO: estimate the measured psf maximum in a better way, moffat maybe 
        meas_max = image.max()
        ymc = np.where(image == meas_max)[0][0]
        xmc = np.where(image == meas_max)[1][0]
        meas_max = self.quadfit_peak_3x3(image, ymc, xmc) 
        sr = meas_max/(self._fitted_dl_max_au * total_measured_flux/self._total_dl_flux)
        
        if enable_display is True:
            
            import matplotlib.pyplot as plt
            
            v_min = 0
            v_max = np.log10(normalized_dl_psf_roi.max())
            fig ,axs = plt.subplots(1,2,sharex=True,sharey=True)
            im0 = axs[0].imshow(np.log10(np.clip(normalized_dl_psf_roi, 0, None)+1),vmin=v_min, vmax=v_max)
            im1 = axs[1].imshow(np.log10(np.clip(image, 0, None)+1),vmin=v_min, vmax=v_max)
            fig.colorbar(im1, ax = axs[1])
            axs[0].title.set_text('DL PSF')
            axs[1].title.set_text('Measured PSF')
            
        return sr
    
    def quadfit_peak_3x3(self, img, yc, xc):

        """
        Stima sub-pixel del picco con un fit quadratico 2D su patch 3x3
        centrato su (yc, xc). Ritorna SOLO I_peak_fit (float).
        Se il fit è degenere o siamo al bordo, usa il valore del pixel centrale.
        """
        h, w = img.shape
        y0, x0 = int(yc), int(xc)
    
        # bordo: se non posso estrarre 3x3, ritorno il pixel
        if y0-1 < 0 or y0+1 >= h or x0-1 < 0 or x0+1 >= w:
            return float(img[y0, x0])
    
        patch = img[y0-1:y0+2, x0-1:x0+2].astype(float)
        Y, X = np.mgrid[-1:2, -1:2]
    
        # modello: a + bX + cY + dX^2 + eXY + fY^2
        A = np.column_stack([
            np.ones(X.size), X.ravel(), Y.ravel(),
            (X**2).ravel(), (X*Y).ravel(), (Y**2).ravel()
        ])
        try:
            coeff, *_ = np.linalg.lstsq(A, patch.ravel(), rcond=None)
            a, b, c, d, e, f = coeff
            denom = (4*d*f - e**2)
            if denom >= 0:  # paraboloide non concavo -> fallback
                return float(patch[1, 1])
            x_peak = (e*c - 2*f*b) / denom
            y_peak = (e*b - 2*d*c) / denom
            # se il vertice è troppo lontano dal centro, fallback (fit fuori patch)
            if abs(x_peak) > 1.5 or abs(y_peak) > 1.5:
                return float(patch[1, 1])
            I_peak = a + b*x_peak + c*y_peak + d*x_peak**2 + e*x_peak*y_peak + f*y_peak**2
            return float(I_peak)
        except Exception:
            return float(patch[1, 1])