from astropy.io import fits
from bronte.startup import set_data_dir
from bronte.package_data import shframes_folder, modal_offsets_folder
import numpy as np 
from bronte.utils.slopes_vector_analyser import SlopesVectorAnalyser
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def z2_scan_analysis():
    '''
    this main is meant to check the wfs linearity
    analysing tip (Z2) scanned slopes for a fixed threshold and to
    estimate sandrine's non linearity factor beta
    '''
    set_data_dir()
    # loading TT scan data
    ftag ='250430_144800' #'250512_102100'#'250430_144800' # Z2 data
    file_name = shframes_folder() / (ftag + '.fits')
    hdr = fits.getheader(file_name)
    jnoll_index = hdr['NOLL_J']
    hdl = fits.open(file_name)
    frame_cube = hdl[0].data 
    Nframes = frame_cube.shape[0]
    ref_frame = hdl[1].data
    ref_frame[ref_frame<0] = 0
    c_vector = hdl[2].data
    
    subap_tag = '250120_122000'
    sva = SlopesVectorAnalyser(subap_tag)
    sva.reload_slope_pc(pix_thr_ratio = 0.18, abs_pix_thr = 0)
    Nsubap = sva._subapertures_set.n_subaps
    NpixperSub = sva._subapertures_set.np_sub
    
    
    s_ref = sva.get_slopes_from_frame(ref_frame)
    slope_cube = np.zeros((Nframes, len(s_ref)))
    
    for idx in np.arange(Nframes):
        
        slope_cube[idx] = sva.get_slopes_from_frame(frame_cube[idx]) - s_ref
    
    sx_cube = slope_cube[:, :Nsubap]
    sy_cube = slope_cube[:, Nsubap:]
    
    sx = sx_cube.mean(axis=-1)
    sy = sy_cube.mean(axis=-1)
    err_sx = sx_cube.std(axis=-1)
    err_sy = sy_cube.std(axis=-1)
    
    plt.figure()
    plt.clf()

    plt.errorbar(c_vector/1e-6, sx*0.5*NpixperSub, err_sx*0.5*NpixperSub, fmt='.-', label='$S_x$')
    plt.errorbar(c_vector/1e-6, sy*0.5*NpixperSub, err_sy*0.5*NpixperSub, fmt='.-', label='$S_y$')
    plt.grid('--', alpha = 0.3)
    plt.ylabel('Slopes [Pixels]')
    plt.xlabel('Zernike coefficient [um] rms wf')
    plt.title(f"$Z{jnoll_index}:\  Npps = 26 pixels \  (pixel \ size=5.5\\mu m)$")
    
    Dslm = 568*2*9.2e-6
    Deff = 10.2e-3
    f_la = 8.31477e-3
    R = Deff*0.5
    f2 = 250e-3
    f3 = 150e-3
    pixel_size = 5.5e-6
    d_la = NpixperSub * pixel_size
    
    c = c_vector[26:41]
    s_exp = ((f2/f3)*(4*c/Dslm)*f_la)/(0.5*NpixperSub*pixel_size)
    plt.plot(c/1e-6, s_exp*0.5*NpixperSub, 'r-', label='$S_{exp}$')

    sexp_in_pixel = s_exp*0.5*NpixperSub
    smeas_in_pixel = sx[26:41]*0.5*NpixperSub
    err_meas_in_pixel = err_sx[26:41]*0.5*NpixperSub
    
    par, cov = curve_fit(_sandrine,  sexp_in_pixel , smeas_in_pixel,p0=[0.8, 1, 1], sigma=err_meas_in_pixel, absolute_sigma=True)
    alpha_r, beta, bias = par
    alpha_r_err, beta_err, bias_err = np.sqrt(np.diag(cov))
    
    print(f"alpha_r = {alpha_r} +/- {alpha_r_err}")
    print(f"beta = {beta} +/- {beta_err}")
    print(f"bias = {bias} +/- {bias_err}")
    
    xfit = np.linspace(sexp_in_pixel[0], sexp_in_pixel[-1], 100)
    sfit = _sandrine(xfit, alpha_r, beta, bias)
    plt.figure()
    plt.clf()
    plt.errorbar(sexp_in_pixel, smeas_in_pixel, err_meas_in_pixel, fmt='.-', label='Data')
    plt.plot(xfit, sfit, 'r-', label = 'fit')
    plt.xlabel(r"$x_0 \ [Pixels]$")
    plt.ylabel(r"$\hat{x} \ [Pixels]$")
    plt.title(r"$\hat{x} = \alpha_r x_0 \ + \beta \ x_0^3 \ + bias$")
    plt.legend(loc = 'best')
    plt.grid('--', alpha = 0.3)
    
    
    return par, cov

def _sandrine(x, alpha_r, beta, bias):
    return alpha_r*x + beta*x**3 + bias
