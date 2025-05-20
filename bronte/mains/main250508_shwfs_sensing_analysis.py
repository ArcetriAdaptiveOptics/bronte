from astropy.io import fits
from bronte.startup import set_data_dir
from bronte.package_data import shframes_folder, modal_offsets_folder
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
    ftag ='250512_102100' #'250512_102100'#'250430_144800' # Z2 data
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
    #plt.plot(c_vector/1e-6, sx, '.-', label='$S_x$')
    #plt.plot(c_vector/1e-6, sy, '.-', label='$S_y$')
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
    
    c = c_vector[25:42]
    s_exp = ((f2/f3)*(4*c/Dslm)*f_la)/(0.5*NpixperSub*pixel_size)
    plt.plot(c/1e-6, s_exp*0.5*NpixperSub, 'r-', label='$S_{exp}$')
    plt.legend(loc='best')
    dd = sx[25:42] - s_exp
    
    
    sexp_in_pixel = s_exp*0.5*NpixperSub
    smeas_in_pixel = sx[25:42]*0.5*NpixperSub
    err_meas_in_pixel = err_sx[25:42]*0.5*NpixperSub
    
    a = c/1e-6
    b = sexp_in_pixel

    # Formatta la colonna "c ± d" come stringhe
    c_pm_d = [f"{c_val:.2f} ± {d_val:.2f}" for c_val, d_val in zip(smeas_in_pixel, err_meas_in_pixel)]
    
    # Stampa intestazione
    print(f"{'c2[um]rms wf':>10} {'Sexp [pixels]':>10} {'Sx_meas [pixels]':>15}")
    print("-" * 40)
    
    # Stampa i dati riga per riga
    for a_val, b_val, c_str in zip(a, b, c_pm_d):
        print(f"{a_val:10.3f} {b_val:10.3f} {c_str:>15}")
    
    plt.figure();
    plt.clf()
    plt.imshow(ref_frame + 1000*sva._subaperture_grid_map)
    plt.colorbar()
    
    plt.figure()
    plt.clf()
    plt.plot(s_ref*0.5*NpixperSub)
    plt.xlabel('2Nsubap')
    plt.ylabel('Slopes [Pixels]')
    plt.grid('--', alpha = 0.3)
    
    sva.display2Dslope_maps_from_slope_vector(s_ref)
    
    hdl = fits.open(modal_offsets_folder()/ ('250509_161700.fits'))  
    modes = -hdl[0].data
    plt.figure()
    plt.clf()
    j_noll = np.arange(2, len(modes)+2)
    plt.plot(j_noll, modes/1e-6, '.-')
    plt.grid('--', alpha = 0.3)
    plt.xlabel('Zernike index j')
    plt.ylabel('$c_j \ [\mu m \ rms \ wf]$')
    plt.title('Modal Offset')
    
    
    
    return frame_cube[30], c_vector, subap_tag


def z3_scan_analysis():
    '''
    this main is meant to plot check the wfs linearity
    analysing tilt (Z3) scanned slopes for a fixed threshold
    '''
    set_data_dir()
    # loading TT scan data
    ftag = '250512_102900'#'250512_102900'#'250430_145200' # Z3 data
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
    err_sx = sx_cube.std(axis=-1)
    err_sy = sy_cube.std(axis=-1)
    
    plt.figure()
    plt.clf()
    #plt.plot(c_vector/1e-6, sx, '.-', label='$S_x$')
    #plt.plot(c_vector/1e-6, sy, '.-', label='$S_y$')
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
    
    c = c_vector[20:39]
    s_exp = ((f2/f3)*(4*c/Dslm)*f_la)/(0.5*NpixperSub*pixel_size)
    plt.plot(c/1e-6, s_exp*0.5*NpixperSub, 'r-', label='$S_{exp}$')
    plt.legend(loc='best')
    
    
    sexp_in_pixel = s_exp*0.5*NpixperSub
    smeas_in_pixel = sy[20:39]*0.5*NpixperSub
    err_meas_in_pixel = err_sy[20:39]*0.5*NpixperSub

    a = c/1e-6
    b = sexp_in_pixel

    # Formatta la colonna "c ± d" come stringhe
    c_pm_d = [f"{c_val:.2f} ± {d_val:.2f}" for c_val, d_val in zip(smeas_in_pixel, err_meas_in_pixel)]
    
    # Stampa intestazione
    print(f"{'c3[um]rms wf':>10} {'Sexp [pixels]':>10} {'Sy_meas [pixels]':>15}")
    print("-" * 40)
    
    # Stampa i dati riga per riga
    for a_val, b_val, c_str in zip(a, b, c_pm_d):
        print(f"{a_val:10.3f} {b_val:10.3f} {c_str:>15}")
        
    
    plt.figure();
    plt.clf()
    plt.imshow(ref_frame + 1000*sva._subaperture_grid_map)
    plt.colorbar()
    
    plt.figure()
    plt.clf()
    plt.plot(s_ref*0.5*NpixperSub)
    plt.xlabel('2Nsubap')
    plt.ylabel('Slopes [Pixels]')
    plt.grid('--', alpha = 0.3)
    
    sva.display2Dslope_maps_from_slope_vector(s_ref)
    
    return frame_cube, c_vector, subap_tag

    