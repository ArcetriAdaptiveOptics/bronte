import numpy as np 
from bronte.package_data import other_folder
from bronte.startup import specula_startup, set_data_dir
from astropy.io import fits
from bronte.utils.raw_strehl_ratio_computer import StrehlRatioComputer
from bronte.utils.data_cube_cleaner import DataCubeCleaner
from bronte.plots.main251001_psf_sharpening import main_low_order, main_high_order
import matplotlib.pyplot as plt

def main(ftag, Nframes  = 10, zc2apply = np.zeros(3)):
    
    sf = specula_startup()

    texp = sf.psf_camera.exposureTime()
    fps = sf.psf_camera.getFrameRate()
    
    #zc2apply = np.zeros(3)
    
    command  = sf.slm_rasterizer.m2c(zc2apply, applyTiltUnderMask=True)
    
    sf.deformable_mirror.set_shape(command)
    
    yc = 554
    xc = 653
    size = 40
    
    #measure_flux_in_roi = np.zeros(Nframes)
    psf_cube = np.zeros((Nframes, size, size))
    
    for idx in np.arange(Nframes):
        
        psf_in_roi = get_psf_in_roi(sf, yc, xc, size)
        psf_cube[idx] = psf_in_roi
        #measure_flux_in_roi[idx] = psf_in_roi.sum()
    
    
    fname = other_folder() / (ftag + '.fits')
    hdr = fits.Header()
    hdr['TEXP_MS'] = texp
    hdr['FPS'] = fps
    
    fits.writeto(fname, psf_cube, hdr)
    
    
def main2(ftag, Nframes  = 10, zc2apply = np.zeros(3), Nframe2avarage = 10):
    
    sf = specula_startup()

    texp = sf.psf_camera.exposureTime()
    fps = sf.psf_camera.getFrameRate()
    
    #zc2apply = np.zeros(3)
    
    command  = sf.slm_rasterizer.m2c(zc2apply, applyTiltUnderMask=True)
    
    sf.deformable_mirror.set_shape(command)
    
    yc = 554
    xc = 653
    size = 40
    
    #measure_flux_in_roi = np.zeros(Nframes)
    psf_cube = np.zeros((Nframes, size, size))
    
    for idx in np.arange(Nframes):
        
        psf_in_roi = get_psf_in_roi_as_avarage(sf, yc, xc, size, Nframe2avarage)
        psf_cube[idx] = psf_in_roi
        #measure_flux_in_roi[idx] = psf_in_roi.sum()
    
    
    fname = other_folder() / (ftag + '.fits')
    hdr = fits.Header()
    hdr['TEXP_MS'] = texp
    hdr['FPS'] = fps
    
    fits.writeto(fname, psf_cube, hdr)
        
def get_psf_in_roi(sf, yc, xc, size):
    
    yc_roi = yc
    xc_roi = xc
    size = size
    bkg = sf.psf_camera_master_bkg
    
    raw_data = sf.psf_camera.getFutureFrames(1).toNumpyArray()
    master_image = raw_data - bkg
    master_image[master_image < 0] = 0
            
    hsize = int(np.round(size*0.5))
    roi_master = master_image[yc_roi-hsize:yc_roi+hsize, xc_roi-hsize:xc_roi+hsize]
    return roi_master

def get_psf_in_roi_as_avarage(sf, yc, xc, size, Nframe2avarage = 10):
    
    yc_roi = yc
    xc_roi = xc
    size = size
    bkg = sf.psf_camera_master_bkg
    
    
    raw_dataCube = sf.psf_camera.getFutureFrames(Nframe2avarage).toNumpyArray()
    master_image = DataCubeCleaner.get_master_from_rawCube(raw_dataCube, bkg)
            
    hsize = int(np.round(size*0.5))
    roi_master = master_image[yc_roi-hsize:yc_roi+hsize, xc_roi-hsize:xc_roi+hsize]
    return roi_master

# --- data acquisition ---

def main251006_115700():
    
    ftag = '251006_115700'
    Nframes = 200
    zc2apply = np.zeros(3)
    main(ftag, Nframes, zc2apply)

def main251006_121100():
    
    ftag = '251006_121100'
    Nframes = 200
    zc2apply = np.zeros(3)
    main(ftag, Nframes, zc2apply)
    
def main251006_121500():
    
    ftag = '251006_121500'
    Nframes = 200
    zc2apply = np.zeros(3)
    main(ftag, Nframes, zc2apply)

def main251006_122500():
    
    ftag = '251006_122500'
    Nframes = 1000
    zc2apply = np.zeros(3)
    main(ftag, Nframes, zc2apply)

def main251006_120300():
    
    ftag = '251006_120300'
    Nframes = 200
    
    zc2apply =  np.array([ 0.0e+00,  0.0e+00, -2.0e-08,  4.0e-08,  1.5e-08,  1.0e-08,
        1.0e-08,  5.0e-09, -5.0e-09, -5.0e-09])
  
    main(ftag, Nframes, zc2apply)
    
def main251006_121300():
    
    ftag = '251006_121300'
    Nframes = 200
    
    zc2apply =  np.array([ 0.0e+00,  0.0e+00, -2.0e-08,  4.0e-08,  1.5e-08,  1.0e-08,
        1.0e-08,  5.0e-09, -5.0e-09, -5.0e-09])
  
    main(ftag, Nframes, zc2apply)

def main251006_121600():
    
    ftag = '251006_121600'
    Nframes = 200
    
    zc2apply =  np.array([ 0.0e+00,  0.0e+00, -2.0e-08,  4.0e-08,  1.5e-08,  1.0e-08,
        1.0e-08,  5.0e-09, -5.0e-09, -5.0e-09])
  
    main(ftag, Nframes, zc2apply)
    
def main251006_123100():
    
    ftag = '251006_123100'
    Nframes = 1000
    
    zc2apply =  np.array([ 0.0e+00,  0.0e+00, -2.0e-08,  4.0e-08,  1.5e-08,  1.0e-08,
        1.0e-08,  5.0e-09, -5.0e-09, -5.0e-09])
  
    main(ftag, Nframes, zc2apply)
    
def main251006_150600():
    
    ftag = '251006_150600'
    Nframes = 200
    Nframes2average = 10
    
    zc2apply =  np.array([ 0.0e+00,  0.0e+00, -2.0e-08,  4.0e-08,  1.5e-08,  1.0e-08,
        1.0e-08,  5.0e-09, -5.0e-09, -5.0e-09])
  
    main2(ftag, Nframes, zc2apply, Nframes2average)

def main251006_154000():
    ftag = '251006_154000'
    Nframes = 1000
    
    c_hat_low, _ =  main_low_order()
    c_hat_hi, _ = main_high_order() 
    plt.close('all')
    
    zc2apply = np.zeros(30)
    
    zc2apply[0:2] = 0
    zc2apply[2:10] = c_hat_low*1e-9
    zc2apply[10:] = c_hat_hi*1e-9
    
    main(ftag, Nframes, zc2apply)
    
# --- analysis --- 
# class PSFDynamicsAnalyser():
#
#     def __init__(self, ftag):
#
#         self._ftag = ftag
#         self._psf_cube, self._texp, self._fps = self._load()
#         self._Nframes = self._psf_cube.shape[0]
#         self._sr_pc  = StrehlRatioComputer()
#         self._flux_in_roi_vector, self._measured_sr_vector = self._compute_flux_and_sr_in_roi()
#
#         self._gain_e_per_adu=3.5409 
#         self._ron_adu = 2.4
#
#     def _compute_flux_and_sr_in_roi(self):
#
#         flux_in_roi_vector = np.zeros(self._Nframes)
#         sr_vector = np.zeros(self._Nframes)
#         for idx in np.arange(self._Nframes):
#             flux_in_roi_vector[idx] = self._psf_cube.sum()
#             sr_vector[idx] = self._sr_pc.get_SR_from_image(self._psf_cube[idx], enable_display=False)
#         return flux_in_roi_vector, sr_vector
#
#     def _load(self):
#         set_data_dir()
#         fname = other_folder() / (self._ftag + '.fits')
#         hdr = fits.getheader(fname)
#         texp =  hdr['TEXP_MS'] 
#         fps = hdr['FPS']
#
#         hdulist = fits.open(fname)
#         psf_cube = hdulist[0].data
#         return psf_cube, texp, fps


    
# def main_analysis():
#
#     ftag = '251006_122500' #1000frames flat
#     #ftag = '251006_123100' #1000frames low order
#     #ftag = '251006_154000' #1000 frames higher orders
#
#     pda = PSFDynamicsAnalyser(ftag)
#
#     sr_mean = pda._measured_sr_vector.mean()
#     sr_err = pda._measured_sr_vector.std()

from astropy.modeling import models, fitting

class PSFDynamicsAnalyser():

    def __init__(self, ftag):

        self._ftag = ftag
        self._psf_cube, self._texp, self._fps = self._load()
        self._Nframes = self._psf_cube.shape[0]
        self._sr_pc  = StrehlRatioComputer()
        self._flux_in_roi_vector, self._measured_sr_vector = self._compute_flux_and_sr_in_roi()

        self._gain_e_per_adu=3.5409 
        self._ron_adu = 2.4

    def _compute_flux_and_sr_in_roi(self):

        flux_in_roi_vector = np.zeros(self._Nframes)
        sr_vector = np.zeros(self._Nframes)
        for idx in np.arange(self._Nframes):
            flux_in_roi_vector[idx] = self._psf_cube.sum()
            sr_vector[idx] = self._sr_pc.get_SR_from_image(self._psf_cube[idx], enable_display=False)
        return flux_in_roi_vector, sr_vector

    def _load(self):
        set_data_dir()
        fname = other_folder() / (self._ftag + '.fits')
        hdr = fits.getheader(fname)
        texp =  hdr['TEXP_MS'] 
        fps = hdr['FPS']

        hdulist = fits.open(fname)
        psf_cube = hdulist[0].data
        return psf_cube, texp, fps

    # ---------- PARAMETRI OTTICI PER LA SCALA ----------
    @property
    def _phys(self):
        # dimensione pupilla fisica (m): 1090 px * 9.2 um/px
        D_m   = 1090 * 9.2e-6
        f_m   = 250e-3
        lam_m = 633e-9
        p_m   = 4.65e-6  # pixel pitch camera
        return {"D": D_m, "f": f_m, "lam": lam_m, "p": p_m}

    def fwhm_theoretical_pixels(self):
        """FWHM DL teorica (Airy) in pixel: 1.028 * λ f / (D * p)."""
        P = self._phys
        return 1.028 * P["lam"] * P["f"] / (P["D"] * P["p"])
    
    def fwhm_theoretical_pixels_with_unc(self,
                                         sigma_f_rel=0.01,
                                         sigma_lambda_abs=0.0,
                                         sigma_D_abs=9.2e-6/np.sqrt(12.0),
                                         sigma_p_rel=0.0):
        """
        Restituisce (FWHM_pix, sigma_FWHM_pix) con propagazione degli errori.

        Parametri di default:
        - sigma_f_rel:    incertezza relativa su f (default 1% ~ tolleranza EFL ±1% per AC254-250-B).
        - sigma_lambda_abs: incertezza assoluta su λ in metri (default 0 se trascurabile).
        - sigma_D_abs:    incertezza assoluta su D in metri (default quantizzazione ~ 9.2µm/√12).
        - sigma_p_rel:    incertezza relativa su p (se nota dal datasheet camera, altrimenti 0).

        Nota: D = 1090 * 9.2 µm, f = 250 mm, λ = 633 nm, p = 4.65 µm.
        """
        # valori nominali
        P = self._phys
        D_m   = P["D"]
        f_m   = P["f"]
        lam_m = P["lam"]
        p_m   = P["p"]

        FWHM_pix = 1.028 * lam_m * f_m / (D_m * p_m)

        # incertezze relative
        rel_lam = (sigma_lambda_abs / lam_m) if sigma_lambda_abs is not None else 0.0
        rel_f   = sigma_f_rel if sigma_f_rel is not None else 0.0
        rel_D   = (sigma_D_abs / D_m) if sigma_D_abs is not None else 0.0
        rel_p   = sigma_p_rel if sigma_p_rel is not None else 0.0

        rel_tot = np.sqrt(rel_lam**2 + rel_f**2 + rel_D**2 + rel_p**2)
        sigma_FWHM_pix = FWHM_pix * rel_tot
        return float(FWHM_pix), float(sigma_FWHM_pix)

    # ---------- UTILITIES: background & crop ----------
    def _estimate_background_ring(self, img, inner_frac=0.7, outer_frac=0.95):
        h, w = img.shape
        yc, xc = (h - 1) / 2.0, (w - 1) / 2.0
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt((yy - yc) ** 2 + (xx - xc) ** 2)
        R = min(yc, xc)
        mask = (r >= inner_frac * R) & (r <= outer_frac * R)
        return float(np.median(img[mask]))

    def _subtract_background(self, img, inner_frac=0.7, outer_frac=0.95):
        bkg = self._estimate_background_ring(img, inner_frac, outer_frac)
        out = img.astype(float) - bkg
        out[out < 0] = 0.0
        return out, bkg

    def _crop_around_peak(self, img, half_size=20):
        """Croppa una finestra centrata sul massimo."""
        y0, x0 = np.unravel_index(np.argmax(img), img.shape)
        y0 = int(y0); x0 = int(x0)
        y1 = max(0, y0 - half_size); y2 = min(img.shape[0], y0 + half_size + 1)
        x1 = max(0, x0 - half_size); x2 = min(img.shape[1], x0 + half_size + 1)
        return img[y1:y2, x1:x2], (y1, x1)

    # ---------- GAUSSIAN 2D FIT ----------
    def _fit_gaussian2d_frame(self, img, crop_half=20, subtract_bkg=True):
        """
        Fit Gaussiano 2D ellittico/ruotato su un singolo frame.
        Ritorna: FWHM_x, FWHM_y (in pixel, lungo gli assi immagine), dict info.
        """
        work = img.copy().astype(float)
        if subtract_bkg:
            work, _ = self._subtract_background(work)

        crop, (y1, x1) = self._crop_around_peak(work, half_size=crop_half)
        h, w = crop.shape
        yy, xx = np.mgrid[0:h, 0:w]

        # stime iniziali
        amp0 = float(crop.max())
        yc0, xc0 = np.unravel_index(np.argmax(crop), crop.shape)
        # larghezze iniziali via momenti semplici
        I = crop
        I_sum = I.sum() + 1e-12
        mx = (xx * I).sum() / I_sum
        my = (yy * I).sum() / I_sum
        var_x = ((xx - mx) ** 2 * I).sum() / I_sum + 1e-6
        var_y = ((yy - my) ** 2 * I).sum() / I_sum + 1e-6
        sx0 = np.sqrt(var_x)
        sy0 = np.sqrt(var_y)

        g2d = models.Gaussian2D(amplitude=amp0, x_mean=xc0, y_mean=yc0,
                                x_stddev=max(sx0, 1.0), y_stddev=max(sy0, 1.0),
                                theta=0.0)
        fitter = fitting.LevMarLSQFitter()
        try:
            best = fitter(g2d, xx, yy, crop)
        except Exception:
            # fallback: usa momenti
            fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))
            fwhm_x = fwhm_factor * sx0
            fwhm_y = fwhm_factor * sy0
            return float(fwhm_x), float(fwhm_y), {"ok": False, "model": None}

        # parametri finali
        sx = float(abs(best.x_stddev.value))
        sy = float(abs(best.y_stddev.value))
        th = float(best.theta.value)

        # varianze effettive lungo assi immagine (rotazione)
        # σ_x'² = σ_x² cos²θ + σ_y² sin²θ ; σ_y'² = σ_x² sin²θ + σ_y² cos²θ
        cos_t, sin_t = np.cos(th), np.sin(th)
        sig_x_eff = np.sqrt((sx**2) * (cos_t**2) + (sy**2) * (sin_t**2))
        sig_y_eff = np.sqrt((sx**2) * (sin_t**2) + (sy**2) * (cos_t**2))

        fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))
        fwhm_x = fwhm_factor * sig_x_eff
        fwhm_y = fwhm_factor * sig_y_eff

        info = {
            "ok": True,
            "model": best,
            "sigma_x": sx, "sigma_y": sy, "theta": th,
            "center_xy_img": (y1 + best.y_mean.value, x1 + best.x_mean.value),
        }
        return float(fwhm_x), float(fwhm_y), info

    # ---------- AIRY 2D FIT ----------
    def _fit_airy2d_frame(self, img, crop_half=25, subtract_bkg=True):
        """
        Fit AiryDisk2D su un singolo frame. Ritorna: FWHM_airy (px), dict info.
        Nota: il modello è circolare; FWHM_x = FWHM_y = FWHM_airy.
        """
        work = img.copy().astype(float)
        if subtract_bkg:
            work, _ = self._subtract_background(work)

        crop, (y1, x1) = self._crop_around_peak(work, half_size=crop_half)
        h, w = crop.shape
        yy, xx = np.mgrid[0:h, 0:w]

        # stime iniziali
        amp0 = float(crop.max())
        yc0, xc0 = np.unravel_index(np.argmax(crop), crop.shape)

        # raggio primo zero (stima teorica in pixel)
        P = self._phys
        r1_th_pix = 1.22 * P["lam"] * P["f"] / (P["D"] * P["p"])
        r0 = max(3.0, r1_th_pix)  # limite inferiore per stabilità

        airy = models.AiryDisk2D(amplitude=amp0, x_0=xc0, y_0=yc0, radius=r0)
        fitter = fitting.LevMarLSQFitter()
        try:
            best = fitter(airy, xx, yy, crop)
        except Exception:
            # fallback: usa stima teorica
            fwhm_airy = (1.028 / 1.22) * r1_th_pix
            return float(fwhm_airy), {"ok": False, "model": None}

        r_fit = float(abs(best.radius.value))
        # FWHM(Airy) = 1.028 λf/D = (1.028/1.22) * r1  (in pixel)
        fwhm_airy = (1.028 / 1.22) * r_fit

        info = {
            "ok": True,
            "model": best,
            "radius_first_zero_px": r_fit,
            "center_xy_img": (y1 + best.y_0.value, x1 + best.x_0.value),
        }
        return float(fwhm_airy), info

    # ---------- SERIE TEMPORALE DELLE FWHM ----------
    def compute_fwhm_timeseries(self, method="gaussian", crop_half=20, subtract_bkg=True):
        """
        Calcola FWHM per ogni frame del cubo.
        method: "gaussian" -> ritorna (fwhm_x, fwhm_y);
                "airy"     -> ritorna (fwhm, fwhm) con stessa FWHM nei due assi.
        """
        N = self._Nframes
        fwhm_x = np.zeros(N, dtype=float)
        fwhm_y = np.zeros(N, dtype=float)

        for k in range(N):
            frame = self._psf_cube[k]
            if method.lower() == "gaussian":
                fx, fy, _ = self._fit_gaussian2d_frame(frame, crop_half=crop_half, subtract_bkg=subtract_bkg)
                fwhm_x[k], fwhm_y[k] = fx, fy
            elif method.lower() == "airy":
                fa, _ = self._fit_airy2d_frame(frame, crop_half=max(crop_half, 25), subtract_bkg=subtract_bkg)
                fwhm_x[k] = fwhm_y[k] = fa
            else:
                raise ValueError("method must be 'gaussian' or 'airy'")

        return fwhm_x, fwhm_y
    
def main_analysis(ftag):
    #ftag = '251006_122500'  # esempio: 1000 frames
    pda = PSFDynamicsAnalyser(ftag)

    # SR medio e std (già nel tuo codice)
    sr_mean = pda._measured_sr_vector.mean()
    sr_std  = pda._measured_sr_vector.std()
    print(ftag)
    print(f"Time average SR = {sr_mean} +/- {sr_std}")

    # FWHM con fit Gaussiano 2D (ellittico/ruotato)
    fwhm_x_g, fwhm_y_g = pda.compute_fwhm_timeseries(method="gaussian", crop_half=20, subtract_bkg=True)
    print("Gaussian2D FWHM [px]:",
          f"mean_x={np.mean(fwhm_x_g):.2f}±{np.std(fwhm_x_g):.2f},",
          f"mean_y={np.mean(fwhm_y_g):.2f}±{np.std(fwhm_y_g):.2f}")

    # FWHM con fit Airy 2D (circolare)
    fwhm_x_a, fwhm_y_a = pda.compute_fwhm_timeseries(method="airy", crop_half=25, subtract_bkg=True)
    print("Airy2D FWHM [px]:",
          f"mean={np.mean(fwhm_x_a):.2f}±{np.std(fwhm_x_a):.2f}")

    # FWHM teorica (DL) in pixel
    
    fwhm_pix, sigma_fwhm_pix = pda.fwhm_theoretical_pixels_with_unc(
                                         sigma_f_rel=0.01, #2.5mm
                                         sigma_lambda_abs=0.2e-9,
                                         sigma_D_abs=9.2e-6/np.sqrt(12.0)
    )
    print(f"FWHM_DL_theoretical [px] ≈ {pda.fwhm_theoretical_pixels():.2f}")
    print(f"Con λ±0.2 nm, f±1%, D quant err: {fwhm_pix:.2f} px ± {sigma_fwhm_pix:.2f} px")
    
    

def main_analysis_flat():
    ftag = '251006_122500' #1000frames flat
    main_analysis(ftag)
    
def main_analysis_low():
    ftag = '251006_123100' # 1000 frames low order
    main_analysis(ftag)
    
def main_analysis_high():
    ftag = '251006_154000' #1000 frames higher orders
    main_analysis(ftag)
    
    