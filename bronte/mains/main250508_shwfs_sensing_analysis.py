from astropy.io import fits
from bronte.startup import set_data_dir
from bronte.package_data import shframes_folder
import numpy as np 
from bronte.utils.slopes_vector_analyser import SlopesVectorAnalyser
import matplotlib.pyplot as plt

def z2_scan_analysis():
    '''
    this main is meant to plot check the wfs linearity
    analysing tip (Z2) scanned slopes for a fixed threshold
    '''
    set_data_dir()
    # loading TT scan data
    ftag = '250509_163100'#'250509_112800'#'250430_144800' # Z2 data
    file_name = shframes_folder() / (ftag + '.fits')
    hdr = fits.getheader(file_name)
    jnoll_index = hdr['NOLL_J']
    hdl = fits.open(file_name)
    frame_cube = hdl[0].data 
    Nframes = frame_cube.shape[0]
    ref_frame = hdl[1].data
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
    
    plt.figure()
    plt.clf()
    plt.plot(c_vector/1e-6, sx, '.-', label='$S_x$')
    plt.plot(c_vector/1e-6, sy, '.-', label='$S_y$')
    plt.grid('--', alpha = 0.3)
    plt.ylabel('Slopes [Normalized]')
    plt.xlabel('Zernike coefficient [um] rms wf')
    
    plt.title(f"Z{jnoll_index}")
    
    Dslm = 568*2*9.2e-6
    Deff = 10.2e-3
    xdata = c_vector[26:37]*Deff/Dslm
    ydata = sx[26:37]
    param = np.polyfit(xdata, ydata, 1)
    
    m = param[0] # in au/meters
    f_la = 8.31477e-3
    R = Deff*0.5
    f2=250e-3
    f3=150e-3
    d_la = NpixperSub*5.5e-6
    m_exp = (f2/f3)*(4/Deff)*f_la/(2*d_la)
    
    print(f"expeceted m = {m_exp}")
    print(f"measured m = {m}")

    xfit = np.linspace(-7e-6,13e-6,200)
    yfit = param[0]*xfit + param[1]
    
    plt.plot(xfit/1e-6,yfit, 'r-', label=r"$fit$") 
    plt.legend(loc='best')
    
    return param


def z3_scan_analysis():
    '''
    this main is meant to plot check the wfs linearity
    analysing tilt (Z3) scanned slopes for a fixed threshold
    '''
    set_data_dir()
    # loading TT scan data
    ftag = '250509_163700'#'250509_113400'#'250430_145200' # Z3 data
    file_name = shframes_folder() / (ftag + '.fits')
    hdr = fits.getheader(file_name)
    jnoll_index = hdr['NOLL_J']
    hdl = fits.open(file_name)
    frame_cube = hdl[0].data 
    Nframes = frame_cube.shape[0]
    ref_frame = hdl[1].data
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
    
    plt.figure()
    plt.clf()
    plt.plot(c_vector/1e-6, sx, '.-', label='$S_x$')
    plt.plot(c_vector/1e-6, sy, '.-', label='$S_y$')
    plt.grid('--', alpha = 0.3)
    plt.ylabel('Slopes [Normalized]')
    plt.xlabel('Zernike coefficient [um] rms wf')
    
    plt.title(f"Z{jnoll_index}")
    
    Dslm = 568*2*9.2e-6
    Deff = 10.2e-3
    xdata = c_vector[23:36]*Deff/Dslm
    ydata = sy[23:36]
    param = np.polyfit(xdata, ydata, 1)
    
    m = param[0] # in au/meters
    f_la = 8.31477e-3
    R = Deff*0.5
    f2=250e-3
    f3=150e-3
    d_la = NpixperSub*5.5e-6
    m_exp = (f2/f3)*(4/Deff)*f_la/(2*d_la)
    
    print(f"expeceted m = {m_exp}")
    print(f"measured m = {m}")
    xfit = np.linspace(-10e-6,8e-6,200)
    yfit = param[0]*xfit + param[1]
    
    plt.plot(xfit/1e-6,yfit, 'r-', label=r"$fit$") 
    plt.legend(loc='best')
    return param
    