import numpy as np 
# from tesi_slm.utils.my_tools import cut_image_around_coord, \
#     get_index_from_image, execute_gaussian_fit_on_image, get_index_from_array
from scipy.optimize import curve_fit
from astropy.io import fits 
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats.funcs import gaussian_fwhm_to_sigma
from astropy.modeling.functional_models import Gaussian2D, Const2D
from matplotlib.patches import FancyArrowPatch, Arc


# --- new functions --- 

def estimate_ron_from_annulus(image, yc, xc, r_in, r_out):
    """Stima RON in ADU dalla corona [r_in, r_out] attorno al centro (yc, xc)."""
    H, W = image.shape
    yy, xx = np.mgrid[:H, :W]
    rr = np.hypot(yy - yc, xx - xc)
    ann = (rr >= r_in) & (rr <= r_out)
    vals = image[ann]
    # usa MAD robusta per RON (senza shot, essendo lontano dal core)
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    sigma_ron_adu = 1.4826 * mad
    bkg_adu = med
    return bkg_adu, sigma_ron_adu

def build_sigma_map_ADU(roi, bkg_adu, ron_adu, g_e_per_adu):
    """
    Var_ADU = shot_ADU + RON_ADU^2  con shot_ADU = max(roi - bkg, 0)/g
    """
    signal_adu = np.clip(roi - bkg_adu, 0, None)
    var_adu = signal_adu / max(g_e_per_adu, 1e-12) + ron_adu**2
    sigma_adu = np.sqrt(var_adu)
    # evita zeri patologici
    sigma_adu[sigma_adu == 0] = np.nanmedian(sigma_adu[sigma_adu > 0]) if np.any(sigma_adu > 0) else 1.0
    return sigma_adu


def _gaussian_fit_core_weighted(image, x0, y0, fwhm_x, fwhm_y, amplitude,
                                g_e_per_adu=3.54, core_halfsize_pix=10,
                                annulus_rin_pix=15, annulus_rout_pix=25):
    """
    1) definisce una ROI core centrata su (y0, x0) di lato 2*core_halfsize+1
    2) stima bkg e RON in ADU da un'annulus fuori dalla ROI
    3) costruisce mappa sigma (ADU) = sqrt( shot(ADU) + RON(ADU)^2 )
    4) fit Gauss + offset costante pesato
    """
    H, W = image.shape
    # ROI del core per il fit
    yc = int(np.clip(y0, core_halfsize_pix, H-core_halfsize_pix-1))
    xc = int(np.clip(x0, core_halfsize_pix, W-core_halfsize_pix-1))
    roi = image[yc-core_halfsize_pix:yc+core_halfsize_pix+1,
                xc-core_halfsize_pix:xc+core_halfsize_pix+1]

    # Stima bkg e RON da annulus nella full image (non dalla ROI)
    bkg_adu, ron_adu = estimate_ron_from_annulus(
        image, yc, xc, annulus_rin_pix, annulus_rout_pix
    )

    # Mappa sigma in ADU
    sigma = build_sigma_map_ADU(roi, bkg_adu, ron_adu, g_e_per_adu)
    weights = 1.0 / (sigma**2)

    # Fit Gauss + offset
    Hc, Wc = roi.shape
    yy, xx = np.mgrid[:Hc, :Wc]
    g = Gaussian2D(
        amplitude=max(amplitude - bkg_adu, 1.0),
        x_mean=core_halfsize_pix, y_mean=core_halfsize_pix,
        x_stddev=fwhm_x * gaussian_fwhm_to_sigma,
        y_stddev=fwhm_y * gaussian_fwhm_to_sigma,
        theta=0.0
    )
    c = Const2D(amplitude=bkg_adu)
    model = c + g

    fitter = LevMarLSQFitter(calc_uncertainties=True)
    fit = fitter(model, xx, yy, roi, weights=weights)

    # parametri gaussiani
    names = fit.param_names
    idx = {n: i for i, n in enumerate(names)}
    amp_g   = fit.parameters[idx['amplitude_1']]
    xmu_g   = fit.parameters[idx['x_mean_1']]
    ymu_g   = fit.parameters[idx['y_mean_1']]
    xsig_g  = fit.parameters[idx['x_stddev_1']]
    ysig_g  = fit.parameters[idx['y_stddev_1']]
    theta_g = fit.parameters[idx['theta_1']]
    fwhm_x_g = xsig_g / gaussian_fwhm_to_sigma
    fwhm_y_g = ysig_g / gaussian_fwhm_to_sigma

    pars = np.array([
        amp_g + fit.parameters[idx['amplitude_0']],
        xmu_g + (xc - core_halfsize_pix),  # riporta in coords immagine
        ymu_g + (yc - core_halfsize_pix),
        fwhm_x_g, fwhm_y_g, theta_g
    ])

    # incertezze (sui soli param. gaussiani)
    try:
        cov = fit.cov_matrix.cov_matrix
        sel = [idx['amplitude_1'], idx['x_mean_1'], idx['y_mean_1'],
               idx['x_stddev_1'], idx['y_stddev_1'], idx['theta_1']]
        cov_g = cov[np.ix_(sel, sel)]
        # scala per chi2_red locale (robusto)
        dof = roi.size - len(names)
        resid = roi - fit(xx, yy)
        chi2 = np.sum(weights * resid**2)
        chi2r = chi2 / max(dof, 1)
        if chi2r > 1:
            cov_g = cov_g * chi2r
        errs = np.sqrt(np.diag(cov_g))
        errs[3] /= gaussian_fwhm_to_sigma
        errs[4] /= gaussian_fwhm_to_sigma
    except Exception:
        errs = np.full(6, np.nan)

    return pars, errs#, (bkg_adu, ron_adu, chi2r)


def _sanitize_err(yerr, min_jitter_pix=0.03):
    yerr = np.asarray(yerr, float)
    bad = ~np.isfinite(yerr) | (yerr <= 0)
    yerr[bad] = np.nan  # marca i brutti
    # se ci sono NaN, rimpiazza col mediano dei buoni o con min_jitter
    if np.any(bad):
        good = np.isfinite(yerr) & (yerr > 0)
        fallback = np.nanmedian(yerr[good]) if np.any(good) else min_jitter_pix
        yerr[bad] = max(fallback, min_jitter_pix)
    # aggiungi un jitter minimo per evitare pesi infiniti
    yerr = np.hypot(yerr, min_jitter_pix)
    return yerr


def estimate_background_and_sigma(roi):
    # fondo robusto
    cam_gain=3.54
    bkg = np.percentile(roi, 10)
    # stima noise (MAD)
    diff = roi - np.median(roi)
    mad = np.median(np.abs(diff))
    sigma_read = 1.4826 * mad
    # shot + read (se ADU ~ fotoni; altrimenti resta una proxy robusta)
    var = np.clip(roi - bkg, 0, None)/cam_gain + sigma_read**2
    sigma = np.sqrt(var)
    # evita zeri
    sigma[sigma == 0] = np.median(sigma[sigma > 0]) if np.any(sigma > 0) else 1.0
    return bkg, sigma

def _gaussian_fit(image, x_mean, y_mean, fwhm_x, fwhm_y, amplitude):
    H, W = image.shape
    yy, xx = np.mgrid[:H, :W]

    # stima fondo e pesi
    bkg, sigma = estimate_background_and_sigma(image)
    weights = 1.0 / (sigma**2)

    # modello: offset costante + gaussiana
    g = Gaussian2D(
        amplitude=max(amplitude - bkg, 1.0),
        x_mean=x_mean, y_mean=y_mean,
        x_stddev=fwhm_x * gaussian_fwhm_to_sigma,
        y_stddev=fwhm_y * gaussian_fwhm_to_sigma,
        theta=0.0
    )
    c = Const2D(amplitude=bkg)
    model = c + g

    fitter = LevMarLSQFitter(calc_uncertainties=True)
    fit = fitter(model, xx, yy, image, weights=weights)

    # -------- estrazione parametri gaussiani (6) --------
    # nomi dei parametri nel fit composito
    names = fit.param_names  # es.: ['amplitude_0','amplitude_1','x_mean_1',...]
    # mappa nome -> indice in vettore parametri
    idx = {name: i for i, name in enumerate(names)}

    # prendi SOLO i sei gaussiani (con suffisso _1)
    amp_g   = fit.parameters[idx['amplitude_1']]
    xmu_g   = fit.parameters[idx['x_mean_1']]
    ymu_g   = fit.parameters[idx['y_mean_1']]
    xsig_g  = fit.parameters[idx['x_stddev_1']]
    ysig_g  = fit.parameters[idx['y_stddev_1']]
    theta_g = fit.parameters[idx['theta_1']]

    # converti stddev -> FWHM
    fwhm_x_g = xsig_g / gaussian_fwhm_to_sigma
    fwhm_y_g = ysig_g / gaussian_fwhm_to_sigma

    pars = np.array([amp_g + fit.parameters[idx['amplitude_0']],  # amp ~ picco+offset
                     xmu_g, ymu_g, fwhm_x_g, fwhm_y_g, theta_g])

    # -------- incertezze (6) coerenti con i parametri gaussiani --------
    try:
        cov = fit.cov_matrix.cov_matrix  # matrice (7x7)
        # costruisci un indice per i soli gaussiani
        sel = [idx['amplitude_1'], idx['x_mean_1'], idx['y_mean_1'],
               idx['x_stddev_1'], idx['y_stddev_1'], idx['theta_1']]
        cov_g = cov[np.ix_(sel, sel)]

        # scala la covarianza se il chi2_red locale > 1 (evita sottostima)
        dof = image.size - len(names)
        resid = image - fit(xx, yy)
        chi2_local = np.sum(weights * resid**2)
        chi2r_local = chi2_local / max(dof, 1)
        if chi2r_local > 1:
            cov_g = cov_g * chi2r_local

        errs = np.sqrt(np.diag(cov_g))
        # converti anche le incertezze su stddev -> FWHM
        # attenzione: il mapping degli elementi della diagonale:
        # [amp1, xmean1, ymean1, xstd1, ystd1, theta1]
        errs[3] /= gaussian_fwhm_to_sigma
        errs[4] /= gaussian_fwhm_to_sigma
    except Exception:
        errs = np.full(6, np.nan)

    return pars, errs

def execute_gaussian_fit_on_image(cut_image, FWHMx, FWHMy, print_par=True):
    
    ymax, xmax = get_index_from_image(cut_image)
    imax = cut_image.max()
    par, err = _gaussian_fit(cut_image, xmax, ymax, FWHMx, FWHMy, imax)
    if print_par:
        print('best fit results: amp, x_mean, y_mean, fwhm_x, fwhm_y, theta')
        print(par); print(err)
        

    return par, err

def execute_gaussian_fit_on_image2(cut_image, FWHMx, FWHMy, print_par=True):
    
    hsize_cut_ima = cut_image.shape[0] //2
    core_hsize = 5
    ymax, xmax = get_index_from_image(cut_image)
    imax = cut_image.max()
    par, err = _gaussian_fit_core_weighted(cut_image, xmax, ymax, FWHMx, FWHMy, imax, g_e_per_adu=3.54, core_halfsize_pix=core_hsize,
                                annulus_rin_pix=hsize_cut_ima - core_hsize, annulus_rout_pix=hsize_cut_ima)
    
                                
    if print_par:
        print('best fit results: amp, x_mean, y_mean, fwhm_x, fwhm_y, theta')
        print(par); print(err)
        

    return par, err

def cut_image_around_coord(image2D, yc, xc, halfside=25):
    yc = int(np.clip(yc, halfside, image2D.shape[0]-halfside-1))
    xc = int(np.clip(xc, halfside, image2D.shape[1]-halfside-1))
    return image2D[yc-halfside:yc+halfside+1, xc-halfside:xc+halfside+1]

def get_index_from_image(image2D):
    # usa un piccolo smoothing per evitare scegliere un pixel anomalo
    from scipy.ndimage import gaussian_filter
    sm = gaussian_filter(image2D, 1.0)
    idx = np.unravel_index(np.argmax(sm), sm.shape)
    return idx[0], idx[1]

# --- Old functions ---




def load_measures(fname):
        header = fits.getheader(fname)
        hduList = fits.open(fname)
        images_3d = hduList[0].data
        c_span = hduList[1].data
        init_coeff = hduList[2].data
            
        Nframes = header['N_AV_FR']
        texp = header['T_EX_MS']
        j_noll = header['Z_J']
        return images_3d, c_span, Nframes, texp, j_noll, init_coeff

# --- Analyzer class ---
class TiltedPsfAnalyzer():
    
    def __init__(self, fname):
        self._tilts_cube, self._c_span, self._Nframes, self._texp,\
         self._j_noll, self._init_coeff = load_measures(fname)
         
    def compute_tilted_psf_desplacement(self, diffraction_limit_fwhm = 3.3, half_side_roi=20):
        
        num_of_tilts = self._tilts_cube.shape[0]
        
        self._pos_x = np.zeros(num_of_tilts)
        self._pos_y = np.zeros(num_of_tilts)
        self._amplitude = np.zeros(num_of_tilts)
        self._fwhm_x = np.zeros(num_of_tilts)
        self._fwhm_y = np.zeros(num_of_tilts)
        self._theta = np.zeros(num_of_tilts)
        self._pos_x_err = np.zeros(num_of_tilts)
        self._pos_y_err = np.zeros(num_of_tilts)
        self._amplitude_err = np.zeros(num_of_tilts)
        self._fwhm_x_err = np.zeros(num_of_tilts)
        self._fwhm_y_err = np.zeros(num_of_tilts)
        self._theta_err = np.zeros(num_of_tilts)
        
        #self._tilts_cube[self._tilts_cube < 0] = 0
        
        for idx in range(num_of_tilts):
            
            tilt_ima = self._tilts_cube[idx].astype(float)
            # bkg = np.percentile(tilt_ima, 5)
            # tilt_ima -= bkg
            y,x = get_index_from_image(tilt_ima)
            half_side = half_side_roi
            roi_tilt = cut_image_around_coord(tilt_ima, y, x, half_side)
            
            fit_par, fit_err = execute_gaussian_fit_on_image(
                roi_tilt, diffraction_limit_fwhm, diffraction_limit_fwhm, False)
            
            self._amplitude[idx], self._pos_x[idx],  self._pos_y[idx], \
                self._fwhm_x[idx], self._fwhm_y[idx],  self._theta[idx] = fit_par
                
            self._amplitude_err[idx], self._pos_x_err[idx], \
                self._pos_y_err[idx], self._fwhm_x_err[idx],\
                    self._fwhm_y_err[idx], self._theta_err[idx] = fit_err
            
            self._pos_x[idx] = x - half_side + self._pos_x[idx]
            self._pos_y[idx] = y - half_side + self._pos_y[idx]
            
    def show_fitted_parametes_as_a_function_of_applyed_tilts(self, diffraction_limit_fwhm = None):
        import matplotlib.pyplot as plt
        
        plt.figure()
        plt.plot(self._c_span, self._fwhm_x, 'bo-', label='FWHM-X')
        plt.errorbar(self._c_span, self._fwhm_x, self._fwhm_x_err, fmt='.', ecolor ='b')
        plt.plot(self._c_span, self._fwhm_y, 'ro-', label='FWHM-Y')
        plt.errorbar(self._c_span, self._fwhm_y, self._fwhm_y_err, fmt='.', ecolor ='r')
        if diffraction_limit_fwhm is not None:
            plt.hlines(diffraction_limit_fwhm, self._c_span.min(),\
                        self._c_span.max(), 'k', '--', label = 'Diffraction limit')
            
        plt.legend(loc='best')
        plt.grid('--', alpha=0.3)
        plt.xlabel('$c_{%d} [m]$'%self._j_noll)
        plt.ylabel('FWHM [pixel]')
        
        plt.figure()
        plt.plot(self._c_span, self._amplitude, 'ro-', )
        plt.errorbar(self._c_span, self._amplitude, self._amplitude_err, fmt='.', ecolor ='r')
        plt.grid('--', alpha=0.3)
        plt.xlabel('$c_{%d} [m]$'%self._j_noll)
        plt.ylabel('Amplitude [ADU]')
    
    def show_tilt_desplacement(self):
        import matplotlib.pyplot as plt
        
        plt.figure()
        plt.plot(self._c_span, self._pos_x,'bo-')
        plt.errorbar(self._c_span, self._pos_x, self._pos_x_err, fmt='.', ecolor='b')
        plt.grid('--', alpha=0.3)
        plt.xlabel('$c_{%d} [m]$'%self._j_noll)
        plt.ylabel('X position [pixel]')
        
        plt.figure()
        plt.plot(self._c_span, self._pos_y,'ro-')
        plt.errorbar(self._c_span, self._pos_y, self._pos_y_err, fmt='.', ecolor='r')
        plt.grid('--', alpha=0.3)
        plt.xlabel('$c_{%d} [m]$'%self._j_noll)
        plt.ylabel('Y position [pixel]')
        
        parX, covX = np.polyfit(
            self._normalize(self._c_span), self._normalize(self._pos_x), 1, cov=True)  
        errX = np.sqrt(covX.diagonal())
        
        parY, covY = np.polyfit(
            self._normalize(self._c_span), self._normalize(self._pos_y), 1, cov=True)  
        errY = np.sqrt(covY.diagonal())
        
        plt.figure()
        plt.plot(self._normalize(self._c_span),self._normalize(self._pos_x),'bo-',label='along-x')
        plt.plot(self._normalize(self._c_span),self._normalize(self._pos_y),'ro-',label='along-y')
        plt.grid('--', alpha=0.3)
        plt.xlabel('$c_{%d} [normalized]$'%self._j_noll)
        plt.ylabel('Normalized position')
        plt.legend()
        
        
        return parX,errX,parY,errY
    
    def _normalize(self, linear_vector):
        a = linear_vector.min()
        b = linear_vector.max()
        x = (linear_vector - a)/(b-a)
        return x
    
    def execute_curve_fit(self):
        y0, x0 = self._pos_y[self._c_span == 0], self._pos_x[self._c_span == 0]
        d = np.sqrt((self._pos_x-x0)**2+(self._pos_y-y0)**2) 
        popt, pcov = curve_fit(
            self._func, 
            self._c_span + self._init_coeff[0],
            d,
            p0=[250e-3, 10.7e-3, 4.65e-6, 0],
            bounds=([200e-3, 10.2e-3, 0, 0], [250e-3, 10.7e-3, 4.65e-6, d.max()]),
            )
        return popt,pcov
    
    def _func(self, x , f, Dpe, pixel_pitch, offset):
        return 4*f*x/(Dpe*pixel_pitch) + offset
    
    def execite_linfit_along1axis(self):
        y0, x0 = self._pos_y[self._c_span == 0], self._pos_x[self._c_span == 0]
        c_span = self._c_span #np.delete(self._c_span, np.where(self._c_span == 0)[0][0])
        #print(c_span)
        if self._j_noll == 2 :
            pos = self._pos_x #np.delete(self._pos_x, np.where(self._c_span == 0)[0][0])
            err_pos = self._pos_x_err #np.delete(self._pos_x_err, np.where(self._c_span == 0)[0][0])
            ref_pos = x0
            
        if self._j_noll == 3 :
            pos = self._pos_y #np.delete(self._pos_y, np.where(self._c_span == 0)[0][0])
            err_pos = self._pos_y_err #np.delete(self._pos_y_err, np.where(self._c_span == 0)[0][0])
            ref_pos = y0
            
        obs_displacement = pos - ref_pos    
        coeff, cov= np.polyfit(c_span, obs_displacement, 1, cov=True, full=False)
        a,b = coeff
        err_coeff = np.sqrt(cov.diagonal())
        
        sigma_a = err_coeff[0]
        fit_displacement = a * c_span + b
        
        exp_displacement = 4*c_span *250e-3/(10.7e-3*4.65e-6)
        a_exp, b_exp = np.polyfit(c_span, -exp_displacement, 1)
        #linearity goodness estimations
        #slope ratio
        slope_ratio = a/a_exp
        err_slope = np.sqrt((sigma_a/a_exp)**2)
        #print(np.sqrt(((a_exp/a**2)*sigma_a)**2))
        #R-squared
        R2 = np.sum((fit_displacement + exp_displacement)**2)/np.sum((obs_displacement+exp_displacement)**2)
        #print(1-(np.sum((obs_displacement-fit_displacement)**2)/np.sum((obs_displacement+exp_displacement)**2)))
        print('R**2 = %g '%R2)
        
        chisq = np.sum(((obs_displacement - fit_displacement)**2/(err_pos)**2))
        redchi = chisq/(len(self._c_span)-1)
        rms_diff_obs_minus_fit = (obs_displacement - fit_displacement).std()
        
        import matplotlib.pyplot as plt
        pix_um = 4.65
        scale = 1e-6 
        plt.subplots(2, 1, sharex=True)
        plt.subplot(2,1,1)
        plt.plot(c_span/scale, obs_displacement*pix_um, 'x',markersize=7, label='measured')
        plt.plot(c_span/scale, fit_displacement*pix_um, 'r-', label='fit')
        plt.plot(c_span/scale, - exp_displacement*pix_um, 'k--',label='expected')
        plt.xlabel('$c_{%d} [\mu m]$ rms' %self._j_noll)
        plt.ylabel('Displacement $[\mu m]$')
        plt.legend(loc = 'best')
        plt.grid('--', alpha = 0.3)
        plt.subplot(2,1,2)
        plt.plot(c_span/scale, (obs_displacement + exp_displacement)*pix_um,'xb', label='meas - exp')
        plt.plot(c_span/scale, (fit_displacement + exp_displacement)*pix_um,'xr' ,label='fit - exp')
        plt.plot(c_span/scale, (obs_displacement - fit_displacement)*pix_um,'xk' ,label='meas - fit')
        plt.legend(loc='best')
        plt.ylabel('Difference $[\mu m]$')
        plt.xlabel('$c_{%d} [\mu m]$ rms' %self._j_noll)
        plt.grid('--', alpha = 0.3)
        
        rel_err_obs_minus_exp = (obs_displacement + exp_displacement)/obs_displacement
        rel_err_fit_obs_minus_exp = (fit_displacement + exp_displacement)/obs_displacement
        rel_err_obs_minus_fit = (obs_displacement - fit_displacement)/obs_displacement
        print(rel_err_obs_minus_exp)
        print(rel_err_fit_obs_minus_exp)
        print(rel_err_obs_minus_fit)
                
        return coeff,err_coeff, redchi, rms_diff_obs_minus_fit, slope_ratio, err_slope, R2
        
        
        
# ---- main ----

def main():
    
    
    fname_z2 = "D:\\phd_slm_edo\\old_data\\230414tpm_red_z2_v4.fits"
    fname_z3 = "D:\phd_slm_edo\old_data\\230414tpm_red_z3_v4.fits"
    
    tpa2 = TiltedPsfAnalyzer(fname_z2)
    tpa2.compute_tilted_psf_desplacement(diffraction_limit_fwhm = 3.3, half_side_roi=20)
    tpa2.execite_linfit_along1axis()
    tpa3 = TiltedPsfAnalyzer(fname_z3)
    tpa3.compute_tilted_psf_desplacement(diffraction_limit_fwhm = 3.3, half_side_roi=16)
    tpa3.execite_linfit_along1axis()
    return tpa2, tpa3 
#
# def main22():
#
#     import matplotlib.pyplot as plt
#     tpa2, tpa3  = main()
#     plt.close('all')
#
#     D = 10.5e-3
#     f = 250e-3
#     pp_cam = 4.65e-6
#
#
#     z2_centroid_pos_x_in_px = tpa2._pos_x
#     z2_centroid_pos_y_in_px = tpa2._pos_y
#     z2_centroid_err_pos_x_in_px = tpa2._pos_x_err
#     z2_centroid_err_pos_y_in_px = tpa2._pos_y_err
#     c2_span = tpa2._c_span
#     z3_centroid_pos_x_in_px = tpa3._pos_x
#     z3_centroid_pos_y_in_px = tpa3._pos_y
#     z3_centroid_err_pos_x_in_px = tpa3._pos_x_err
#     z3_centroid_err_pos_y_in_px = tpa3._pos_y_err
#     c3_span = tpa2._c_span
    
# ---

def _weighted_baseline_at_zero(c_span, y, yerr, tol=1e-15):
    """Baseline (flat) come media pesata dei punti con c≈0."""
    c_span = np.asarray(c_span)
    y = np.asarray(y)
    yerr = np.asarray(yerr)
    m = np.isfinite(c_span) & np.isfinite(y) & np.isfinite(yerr)
    m &= (np.abs(c_span) <= tol)
    if not np.any(m):
        raise ValueError("Nessun punto con c_span=0 trovato per definire il riferimento (flat).")
    w = 1.0 / np.clip(yerr[m], 1e-30, np.inf)**2
    y0 = np.sum(w * y[m]) / np.sum(w)
    # incertezza della media pesata
    y0_err = np.sqrt(1.0 / np.sum(w))
    return y0, y0_err

def _linear_fit_weighted(x, y, yerr):
    """
    Fit lineare pesato: y = m x + q.
    Ritorna m, dm, q, dq, R2, chi2_red.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    yerr = np.asarray(yerr)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    x, y, yerr = x[m], y[m], yerr[m]
    if x.size < 3:
        raise ValueError("Troppi pochi punti utili per il fit.")

    # polyfit con pesi = 1/sigma
    coeffs, cov = np.polyfit(x, y, deg=1, w=1.0/np.clip(yerr, 1e-30, np.inf), cov=True)
    m_fit, q_fit = coeffs[0], coeffs[1]
    dm, dq = np.sqrt(np.diag(cov))

    yhat = m_fit * x + q_fit
    # R^2 pesato (usiamo la definizione standard con media pesata)
    w = 1.0 / yerr**2
    ybar_w = np.sum(w * y) / np.sum(w)
    ss_res = np.sum(w * (y - yhat)**2)
    ss_tot = np.sum(w * (y - ybar_w)**2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    chi2 = np.sum(((y - yhat) / yerr)**2)
    dof = max(len(x) - 2, 1)
    chi2_red = chi2 / dof

    return m_fit, dm, q_fit, dq, R2, chi2_red, (x, y, yerr, yhat)

def _nice_ci(val, err, k=1.96):
    return f"{val:.6g} ± {k*err:.2g}"

def main2():
    # --- INPUT / DATI ---
    import matplotlib.pyplot as plt
    tpa2, tpa3 = main()  # <- deve restituire gli oggetti con gli attributi usati sotto
    plt.close('all')

    # Geometria/scala
    D = 10.5e-3      # pupil diameter [m]
    f = 250e-3       # focal length f2 [m]
    pp_cam = 4.65e-6 # pixel pitch camera [m/pix]

    # Dataset TIP (c2-only)
    z2_x_px = np.asarray(tpa2._pos_x)      # [pix]
    z2_y_px = np.asarray(tpa2._pos_y)      # [pix]
    z2_xe_px = np.asarray(tpa2._pos_x_err) # [pix]
    z2_ye_px = np.asarray(tpa2._pos_y_err) # [pix]
    c2_span  = np.asarray(tpa2._c_span)    # [m of OPD RMS]

    # Dataset TILT (c3-only)
    z3_x_px = np.asarray(tpa3._pos_x)      # [pix]
    z3_y_px = np.asarray(tpa3._pos_y)      # [pix]
    z3_xe_px = np.asarray(tpa3._pos_x_err) # [pix]
    z3_ye_px = np.asarray(tpa3._pos_y_err) # [pix]
    c3_span  = np.asarray(tpa3._c_span)    # [m of OPD RMS]  # <-- fix del tuo snippet

    # --- RIFERIMENTO (flat) e DELTE in PIXEL ---
    # baseline al flat (c≈0) per ciascun asse/dataset
    z2_x0, z2_x0e = _weighted_baseline_at_zero(c2_span, z2_x_px, z2_xe_px)
    z2_y0, z2_y0e = _weighted_baseline_at_zero(c2_span, z2_y_px, z2_ye_px)
    z3_x0, z3_x0e = _weighted_baseline_at_zero(c3_span, z3_x_px, z3_xe_px)
    z3_y0, z3_y0e = _weighted_baseline_at_zero(c3_span, z3_y_px, z3_ye_px)

    # Δ = pos - baseline; errore combinato
    dx2_px  = z2_x_px - z2_x0
    dy2_px  = z2_y_px - z2_y0
    dx2e_px = np.hypot(z2_xe_px, z2_x0e)
    dy2e_px = np.hypot(z2_ye_px, z2_y0e)

    dx3_px  = z3_x_px - z3_x0
    dy3_px  = z3_y_px - z3_y0
    dx3e_px = np.hypot(z3_xe_px, z3_x0e)
    dy3e_px = np.hypot(z3_ye_px, z3_y0e)
    
    dx2e_px = _sanitize_err(dx2e_px)
    dy2e_px = _sanitize_err(dy2e_px)
    dx3e_px = _sanitize_err(dx3e_px)
    dy3e_px = _sanitize_err(dy3e_px)


    # --- FIT LINEARE (in PIXEL vs c) ---
    # TIP: Δx vs c2 ; Δy vs c2
    sx2_pix, dsx2_pix, bx2_pix, dbx2_pix, R2_x2, chi2r_x2, pack_x2 = _linear_fit_weighted(c2_span, dx2_px, dx2e_px)
    sy2_pix, dsy2_pix, by2_pix, dby2_pix, R2_y2, chi2r_y2, pack_y2 = _linear_fit_weighted(c2_span, dy2_px, dy2e_px)

    # TILT: Δx vs c3 ; Δy vs c3
    sx3_pix, dsx3_pix, bx3_pix, dbx3_pix, R2_x3, chi2r_x3, pack_x3 = _linear_fit_weighted(c3_span, dx3_px, dx3e_px)
    sy3_pix, dsy3_pix, by3_pix, dby3_pix, R2_y3, chi2r_y3, pack_y3 = _linear_fit_weighted(c3_span, dy3_px, dy3e_px)

    # --- CONVERSIONE A METRI (per K e ψ) ---
    # slope in [pix / mOPD] -> [m_image / mOPD]
    sx2 = sx2_pix * pp_cam
    sy2 = sy2_pix * pp_cam
    sx3 = sx3_pix * pp_cam
    sy3 = sy3_pix * pp_cam
    dsx2 = dsx2_pix * pp_cam
    dsy2 = dsy2_pix * pp_cam
    dsx3 = dsx3_pix * pp_cam
    dsy3 = dsy3_pix * pp_cam

    # Gain atteso/Stimato
    K_th = 4.0 * f / D  # [m_image / mOPD]
    K2 = np.hypot(sx2, sy2)
    K3 = np.hypot(sx3, sy3)
    K_hat = 0.5 * (K2 + K3)

    # errore su K (propagazione semplice, indipendenza approssimata)
    dK2 = (1.0 / K2) * np.sqrt((sx2*dsx2)**2 + (sy2*dsy2)**2) if K2 > 0 else np.nan
    dK3 = (1.0 / K3) * np.sqrt((sx3*dsx3)**2 + (sy3*dsy3)**2) if K3 > 0 else np.nan
    dKhat = 0.5 * np.hypot(dK2, dK3)

    # Rotazioni (in radianti -> gradi)
    psi2 = np.degrees(np.arctan2(sy2, sx2))
    psi3 = np.degrees(np.arctan2(-sx3, sy3))
    psi_hat = (psi2 + psi3) / 2.0
    dpsi = 0.5 * np.hypot(  # stima grezza dell'incertezza media
        np.degrees(np.hypot(dsy2/abs(sx2), dsx2*abs(sy2)/(sx2**2 + 1e-30))),
        np.degrees(np.hypot(dsx3/abs(sy3 + 1e-30), dsy3*abs(sx3)/(sy3**2 + 1e-30)))
    )
    
    # phi1 = atan2(sy|2, sx|2)
    phi1_deg = np.degrees(np.arctan2(sy2_pix, sx2_pix))
    var_phi1 = ((sy2_pix**2)*(dsx2_pix**2) + (sx2_pix**2)*(dsy2_pix**2)) / ((sx2_pix**2 + sy2_pix**2)**2)
    dphi1_deg = np.degrees(np.sqrt(var_phi1))
    
    # phi2 = atan2(-sx|3, sy|3)  -> y = -sx3, x = sy3
    phi2_deg = np.degrees(np.arctan2(-sx3_pix, sy3_pix))
    var_phi2 = (((-sx3_pix)**2)*(dsy3_pix**2) + (sy3_pix**2)*(dsx3_pix**2)) / ((sy3_pix**2 + (-sx3_pix)**2)**2)
    dphi2_deg = np.degrees(np.sqrt(var_phi2))
    

    
    # --- REPORT TESTUALE ---
    print("\n=== TIP-only (c2) fits (pixels per meter OPD) ===")
    print(f"sx|2 = {sx2_pix:.6g} ± {dsx2_pix:.2g}  [pix/m],  R^2={R2_x2:.5f},  χ²_red={chi2r_x2:.3f},  intercept={bx2_pix:.3g}±{dbx2_pix:.1g} pix")
    print(f"sy|2 = {sy2_pix:.6g} ± {dsy2_pix:.2g}  [pix/m],  R^2={R2_y2:.5f},  χ²_red={chi2r_y2:.3f},  intercept={by2_pix:.3g}±{dby2_pix:.1g} pix")

    print("\n=== TILT-only (c3) fits (pixels per meter OPD) ===")
    print(f"sx|3 = {sx3_pix:.6g} ± {dsx3_pix:.2g}  [pix/m],  R^2={R2_x3:.5f},  χ²_red={chi2r_x3:.3f},  intercept={bx3_pix:.3g}±{dbx3_pix:.1g} pix")
    print(f"sy|3 = {sy3_pix:.6g} ± {dsy3_pix:.2g}  [pix/m],  R^2={R2_y3:.5f},  χ²_red={chi2r_y3:.3f},  intercept={by3_pix:.3g}±{dby3_pix:.1g} pix")

    print("\n=== Mapping & geometry (meters per meter OPD) ===")
    print(f"K_th  = {K_th:.6g}  [m/m]")
    print(f"K2    = {K2:.6g} ± {dK2:.2g}  [m/m]   (from c2-only)")
    print(f"K3    = {K3:.6g} ± {dK3:.2g}  [m/m]   (from c3-only)")
    print(f"K_hat = {K_hat:.6g} ± {dKhat:.2g}  [m/m]   (average)")

    print("\n=== Rotation estimates ===")
    print(f"psi2 (from c2)  = {psi2:.3f} deg")
    print(f"psi3 (from c3)  = {psi3:.3f} deg")
    print(f"psi_hat (avg)   = {psi_hat:.3f} ± {dpsi:.3f} deg")
    print(f"\nAngle uncertainties:")
    print(f"phi1 = {phi1_deg:.3f} ± {dphi1_deg:.3f} deg   (from c2 column)")
    print(f"phi2 = {phi2_deg:.3f} ± {dphi2_deg:.3f} deg   (from c3 column)")

    # Coerenza incrociata ideale: sx|2 ≈ sy|3 ; sy|2 ≈ -sx|3
    print("\n=== Cross-checks (ideal equalities) ===")
    print(f"sx|2  vs  sy|3  : {sx2_pix:.6g}  vs  {sy3_pix:.6g}  [pix/m]  -> Δ={sx2_pix - sy3_pix:.3g}")
    print(f"sy|2  vs -sx|3  : {sy2_pix:.6g}  vs {-sx3_pix:.6g}  [pix/m]  -> Δ={sy2_pix + sx3_pix:.3g}")

    # --- PLOT ---
    # Helper per i quattro pannelli + residui
    def _plot_one(ax_main, ax_res, pack, title):
        x, y, ye, yhat = pack
        
        ax_main.errorbar(x/1e-6, y, ye, fmt='o', ms=4, capsize=2, label='data')
        xx = np.linspace(np.min(x), np.max(x), 200)
        # ricostruisco la retta dal fit corrente
        mloc, qloc = np.polyfit(x, y, 1, w=1.0/np.clip(ye, 1e-30, np.inf))
        ax_main.plot(xx/1e-6, mloc*xx + qloc, '-', label='fit')
        ax_main.set_title(title)
        ax_main.set_xlabel('Zernike coefficient [m WF RMS]')
        ax_main.set_ylabel('Centroid shift [pix]')
        ax_main.grid(True, alpha=0.3)
        ax_main.legend()
        # residui
        ax_res.axhline(0, color='k', ls='--', lw=1, alpha=0.4)
        ax_res.errorbar(x/1e-6, y - yhat, ye, fmt='o', ms=3)
        ax_res.set_xlabel('Zernike coefficient [um WF RMS]')
        ax_res.set_ylabel('residual [pix]')
        ax_res.grid(True, alpha=0.3)

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(4, 2, height_ratios=[3,1,3,1], hspace=0.6)

    ax11 = fig.add_subplot(gs[0,0]); ax12 = fig.add_subplot(gs[1,0])
    _plot_one(ax11, ax12, pack_x2, 'TIP-only (c2): Δx vs c2')

    ax21 = fig.add_subplot(gs[0,1]); ax22 = fig.add_subplot(gs[1,1])
    _plot_one(ax21, ax22, pack_y2, 'TIP-only (c2): Δy vs c2')

    ax31 = fig.add_subplot(gs[2,0]); ax32 = fig.add_subplot(gs[3,0])
    _plot_one(ax31, ax32, pack_x3, 'TILT-only (c3): Δx vs c3')

    ax41 = fig.add_subplot(gs[2,1]); ax42 = fig.add_subplot(gs[3,1])
    _plot_one(ax41, ax42, pack_y3, 'TILT-only (c3): Δy vs c3')

    fig.suptitle('SLM Tip–Tilt Linearity: centroid shifts and residuals', fontsize=14)
    plt.show()

    # # Diagramma vettoriale degli slope (intuizione su rotazione)
    # fig2, ax = plt.subplots(figsize=(5,5))
    # ax.quiver([0,0], [0,0], [sx2_pix, sx3_pix], [sy2_pix, sy3_pix],
    #           angles='xy', scale_units='xy', scale=1, width=0.006)
    # ax.set_aspect('equal', 'box')
    # ax.set_xlabel('Δx slope [pix/m]')
    # ax.set_ylabel('Δy slope [pix/m]')
    # ax.grid(True, alpha=0.3)
    # ax.set_title(f"Slope vectors (pix/m)   |   ψ≈{psi_hat:.2f}°  |  K_hat={K_hat/pp_cam:.3g} pix/m")
    # ax.legend([r'$\vec s_{\cdot|2}$', r'$\vec s_{\cdot|3}$'], loc='upper left')
    # plt.show()
    # v2 = np.array([sx2_pix, sy2_pix])   # s_{·|2}
    # v3 = np.array([sx3_pix, sy3_pix])   # s_{·|3}
    #
  # --- Diagramma vettoriale normalizzato (solo geometria/orientazione) ---


    # vettori in pixel/m
    v2 = np.array([sx2_pix, sy2_pix])   # s_{·|2}
    v3 = np.array([sx3_pix, sy3_pix])   # s_{·|3}
    
    # normalizzazione a norma 1
    n2 = np.hypot(*v2); n3 = np.hypot(*v3)
    u2 = v2 / n2
    u3 = v3 / n3
    
    # angolo tra i vettori (gradi)
    dot = float(np.dot(u2, u3))
    ang_deg = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    
    # componenti (interpretazione: cosψ e sinψ, con segni)
    ux2, uy2 = u2
    ux3, uy3 = u3
    
    # cross-coupling in percento (modulo delle componenti "trasversali")
    cc_2  = 100.0 * abs(uy2)   # quanto c2 "entra" in y
    cc_3  = 100.0 * abs(ux3)   # quanto c3 "entra" in x
    
    # figura
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    
    # cerchio unitario
    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), ls='--', lw=1.2, alpha=0.6, color='0.5')
    
    # assi
    ax.axhline(0, lw=1.0, color='0.6', alpha=0.6)
    ax.axvline(0, lw=1.0, color='0.6', alpha=0.6)
    
    def arrow_unit(vec, color, label):
        a = FancyArrowPatch((0, 0), (vec[0], vec[1]),
                            arrowstyle='-|>', mutation_scale=16,
                            lw=2.2, color=color, alpha=0.95)
        ax.add_patch(a)
        ax.text(vec[0]*1.06, vec[1]*1.06, label,
                ha='center', va='center', fontsize=10, fontweight='bold', color=color)
    
    # frecce normalizzate
    arrow_unit(u2, '#1f77b4', r'$\hat s_{\cdot|2}$')
    arrow_unit(u3, '#d62728', r'$\hat s_{\cdot|3}$')
    
    # arco indicante l'angolo tra i due vettori
    phi2 = np.degrees(np.arctan2(u2[1], u2[0]))
    phi3 = np.degrees(np.arctan2(u3[1], u3[0]))
    # sweep minimo in modulo
    sweep = phi3 - phi2
    while sweep <= -180: sweep += 360
    while sweep >   180: sweep -= 360
    arc_r = 0.35
    arc = Arc((0,0), 2*arc_r, 2*arc_r, angle=0,
              theta1=phi2, theta2=phi2+sweep, lw=2, ls='--', alpha=0.8, color='0.3')
    ax.add_patch(arc)
    ax.text(arc_r*1.15*np.cos(np.radians(phi2 + sweep/2)),
            arc_r*1.15*np.sin(np.radians(phi2 + sweep/2)),
            fr'{abs(ang_deg):.2f}$^\circ$', ha='center', va='center',
            fontsize=10, bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.8'))
    
    # box con cos/sin e cross-coupling
    ax.text(0.02, 0.98,
            fr'$\hat s_{{\cdot|2}} = (\cos\psi,\ \sin\psi)\approx({ux2:+.3f},\,{uy2:+.3f})$'
            '\n'
            fr'$\hat s_{{\cdot|3}} = (-\sin\psi,\ \cos\psi)\approx({ux3:+.3f},\,{uy3:+.3f})$'
            '\n'
            fr'cross-talk: $|y|$@c2 $\approx$ {cc_2:.2f}\%, $|x|$@c3 $\approx$ {cc_3:.2f}\%',
            transform=ax.transAxes, ha='left', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='0.8'))
    
    # box riassuntivo con ψ e K (inserisco K in pix/m per riferimento)
    ax.text(0.98, 0.02,
            fr'$\psi \approx {psi_hat:.2f}^\circ$'
            '\n'
            fr'$K_\mathrm{{hat}} \approx {K_hat/pp_cam:.3g}\ \mathrm{{pix/m}}$',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.45', fc='#f7f7f7', ec='0.8'))
    
    # finiture
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel('normalized Δx slope', fontsize=11)
    ax.set_ylabel('normalized Δy slope', fontsize=11)
    ax.set_title('Normalized slope directions — orientation & cross-coupling', fontsize=12)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()
    # Box riassuntivo rapido
    print("\n=== Summary ===")
    rel_err_K = 100.0 * (K_hat - K_th) / K_th
    print(f"K_hat / K_th = {K_hat:.6g} / {K_th:.6g}  ->  {rel_err_K:+.2f}%")
    print(f"Rotation ψ   = {psi_hat:.3f}°   (ψ2={psi2:.3f}°, ψ3={psi3:.3f}°)")
    print(
        f"R^2          = "
        f"x|c2: {R2_x2:.5f}, y|c2: {R2_y2:.5f}, x|c3: {R2_x3:.5f}, y|c3: {R2_y3:.5f}"
    )
    
    print(
        f"χ²_red       = "
        f"x|c2: {chi2r_x2:.3f}, y|c2: {chi2r_y2:.3f}, x|c3: {chi2r_x3:.3f}, y|c3: {chi2r_y3:.3f}"
    )


