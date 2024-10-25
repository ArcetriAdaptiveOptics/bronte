import numpy as np 
from arte.types.mask import CircularMask

class StrehlRatioComputer():
    
    RAD2ARCSEC = 180/np.pi*3600
    
    def __init__(self):
        
        self._pupil_diameter = 2*571*9.2e-6
        self._wl = 633e-9
        self._telescope_focal_length = 250e-3
        self._ccd_pixel_size = 4.65e-6
        self._pixel_scale_in_arcsec = self._ccd_pixel_size/self._telescope_focal_length
        self._dl_size_in_arcsec = self._el / self._pupil_diameter * self.RAD2ARCSEC 
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
        Npad = self._wl / self._pupil_diameter  * self.RAD2ARCSEC /  self._pixel_scale_in_arcsec
        print("Pupil padding %g" % Npad)
        padded_frame_size = int(np.round(Npix * Npad))
        padded_Ut = np.zeros((padded_frame_size, padded_frame_size), dtype=complex)
        padded_Ut[0 : Ut.shape[0], 0 : Ut.shape[1]] = Ut   
    
        #computing psf
        self._dl_psf = np.abs(np.fft.fftshift(np.fft.fft2(padded_Ut)))**2
        self._dl_psf_scale_in_arcsec = self._wl / self._pupil_diameter / Npad * self.RAD2ARCSEC
        self._total_dl_flux = self._dl_psf.sum()
    
    def get_SR_from_image(self, image, enable_display = False):
        
        hsize = int(np.round(image.shape()[0]*0.5))
        total_measured_flux = image.sum()
        normalized_dl_psf = self._dl_psf * total_measured_flux/self._total_dl_flux
        center=(normalized_dl_psf.shape[0]) // 2
        normalized_dl_psf_roi = normalized_dl_psf[center-hsize:center+hsize,
                                                  center-hsize:center+hsize]
        
        sr = image.max()/normalized_dl_psf_roi.max()
        
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
    