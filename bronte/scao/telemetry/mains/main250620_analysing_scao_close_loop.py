from bronte import startup
from bronte.scao.telemetry.scao_telemetry_data_analyser import ScaoTelemetryDataAnalyser
import numpy as np
import matplotlib.pyplot as plt
from arte.types.wavefront import Wavefront
from bronte.utils.slopes_covariance_matrix_analyser import SlopesCovariaceMatrixAnalyser
from bronte.package_data import shframes_folder, other_folder
from astropy.io import fits


def main(telemetry_ftag,
          slope_offset_tag = '250625_145500',
          subap_tag = '250612_143100',
          rec_tag = '250616_103300'
          ):
    
    #subap_tag = '250612_143100'
    #slope_offset_tag = '250625_145500'#'250613_140600'
    #rec_tag = '250616_103300'#'250619_141800'
    #telemetry_ftag = '250625_143500'#'250625_113600'#'250625_102000'#'250625_095600'#'250625_102000'#'250625_095600'#'250619_171100'
    
    sf = startup.specula_startup()
    sf.SUBAPS_TAG = subap_tag
    sf.REC_MAT_TAG = rec_tag
    sf.SLOPE_OFFSET_TAG = slope_offset_tag
    
    slope_offset = sf.slope_offset
    rec_mat = sf.reconstructor.recmat.recmat
    c_offset_in_nm = np.dot(rec_mat, slope_offset)
    wf_ref = sf.slm_rasterizer.zernike_coefficients_to_raster(c_offset_in_nm).toNumpyArray()
    
    plt.figure()
    plt.clf()
    plt.plot(slope_offset, label = r'$s_{offset}=IM c_{0}$')
    plt.xlabel('2Nsubap index')
    plt.ylabel('Slopes [normalized]')
    plt.grid('--', alpha=0.3)
    plt.legend(loc='best')
    
    j_noll_vector = np.arange(0, len(c_offset_in_nm))+2
    plt.figure()
    plt.plot(j_noll_vector, c_offset_in_nm, '.-',label = r'$c_{offset}=Rs_{offset}$')
    plt.xlabel('j mode index')
    plt.ylabel('cj nm rms wf')
    plt.grid('--', alpha=0.3)
    plt.legend(loc='best')
    
    plt.figure()
    plt.clf()
    plt.title('Reference WF')
    plt.imshow(wf_ref)
    plt.colorbar(label = 'nm rms wf')
    
    stda = ScaoTelemetryDataAnalyser(telemetry_ftag)
    
    stda.display_rms_slopes()
    stda.display_delta_cmds_temporal_evolution()
    
    Nstep_conv = 40#30 # from rms slopes temporal evolution
    integ_modal_cmd = stda._integ_cmds[Nstep_conv:,:]/1e-9
    mean_integ_modal_cmd = integ_modal_cmd.mean(axis=0)
    
    wf_final = sf.slm_rasterizer.zernike_coefficients_to_raster(mean_integ_modal_cmd).toNumpyArray()
    
    plt.figure()
    plt.clf()
    plt.title('WF Integ')
    plt.imshow(wf_final)
    plt.colorbar(label = 'nm rms wf')
    
    vmin = np.min((wf_ref.min(),wf_final.min()))
    vmax = np.max((wf_ref.max(),wf_final.max()))
    plt.subplots(1,2,sharex=True, sharey=True)
    plt.subplot(1,2,1)
    plt.imshow(wf_ref, vmin=vmin, vmax=vmax)
    plt.colorbar(label = 'nm rms wf')
    plt.title('Reference WF')
    plt.subplot(1,2,2)
    plt.imshow(wf_final, vmin=vmin, vmax=vmax)
    plt.colorbar(label = 'nm rms wf')
    plt.title('Integ')
    
    res_wf = wf_ref + wf_final
    fit_err = (res_wf).std()
    plt.figure()
    plt.clf()
    plt.imshow(wf_ref + wf_final)
    plt.colorbar(label = 'nm rms wf')
    plt.title(f'fitting error {fit_err} nm rms wf')
    
    resWF = Wavefront(res_wf)
    res_coeff = sf.slm_rasterizer._zernike_modal_decomposer.measureModalCoefficientsFromWavefront(resWF, sf.slm_rasterizer.slm_pupil_mask, sf.slm_rasterizer.slm_pupil_mask)   
    plt.figure()
    plt.clf()
    plt.plot( res_coeff.toNumpyArray(), '.-', label='decomposed res WF')
    plt.xlabel('mode index')
    plt.ylabel('cj nm rms wf')
    
    fit_err = (wf_ref + wf_final).std()
    
    return slope_offset, rec_mat, stda



def compute_ol_no_turb_data():
    
    startup.set_data_dir()
    
    subap_tag = '250612_143100'
    rec_tag = '250616_103300'
    sh_frames_tag = '250625_144900'
    
    sh_frames_fname = shframes_folder() / (sh_frames_tag + '.fits')
    hduList = fits.open(sh_frames_fname)
    ol_frame_cube = hduList[0].data
    
    
    scma = SlopesCovariaceMatrixAnalyser(subap_tag)
    pix_thr = 0.18
    scma.set_slopes_from_frame_cube(ol_frame_cube, pix_thr_ratio=pix_thr, abs_pix_thr=0)
    ol_slopes_cube = scma._slopes_cube
    ol_rms_slopes_x_cube = scma._rms_slopes_x_cube
    ol_rms_slopes_y_cube = scma._rms_slopes_y_cube
    scma.load_reconstructor(rec_tag)
    scma.compute_delta_modal_command()
    ol_delta_cmds_cube_in_nm = scma.get_delta_modal_command()
    
    hdr = fits.Header()
    hdr['SUB_TAG'] = subap_tag
    hdr['REC_TAG'] = rec_tag
    hdr['SHFR_TAG'] = sh_frames_tag
    hdr['SH_THR'] = pix_thr
    filename = other_folder() / ('250709_101300.fits')
    fits.writeto(filename, ol_slopes_cube, hdr)
    fits.append(filename, ol_delta_cmds_cube_in_nm)
    fits.append(filename, ol_rms_slopes_x_cube)
    fits.append(filename, ol_rms_slopes_y_cube)

def load_ol_noturb_data():
    
    startup.set_data_dir()
    fname = other_folder()/('250709_101300.fits')
    hdr = fits.getheader(fname)
    hduList = fits.open(fname)
    ol_slopes_cube = hduList[0].data
    ol_delta_cmds_cube_in_nm = hduList[1].data
    ol_rms_slopes_x_cube = hduList[2].data
    ol_rms_slopes_y_cube = hduList[3].data
    
    return ol_slopes_cube, ol_delta_cmds_cube_in_nm, ol_rms_slopes_x_cube, ol_rms_slopes_y_cube, hdr
    
    
def main250709_noturb_loop():
    
    rec_tag = '250616_103300'
    subap_tag = '250612_143100'
    slope_offset_tag = '250625_145500' # used ONLY for OL data
    cl_tag = '250625_102000'
    Nmodes = 200
    j_vector = np.arange(Nmodes)+2
    
    ol_slopes_cube, ol_delta_cmds_cube_in_nm, ol_rms_slopes_x_cube, ol_rms_slopes_y_cube, olhdr = load_ol_noturb_data()
    mean_ol_delta_cmd_in_nm = ol_delta_cmds_cube_in_nm.mean(axis=0)
    
    stda_cl = ScaoTelemetryDataAnalyser(cl_tag)
    cmask = stda_cl.get_circular_mask(radius=545, coord_yx=(579,968))
    stda_cl.load_slm_rasterizer(cmask, Nmodes=Nmodes)
    
    ol_wf = stda_cl._slm_rasterizer.zernike_coefficients_to_raster(mean_ol_delta_cmd_in_nm).toNumpyArray()
    mean_ol_delta_cmd_in_nm_filtered = mean_ol_delta_cmd_in_nm.copy()
    mean_ol_delta_cmd_in_nm_filtered [:3] = 0
    ol_wf_filtered = stda_cl._slm_rasterizer.zernike_coefficients_to_raster(mean_ol_delta_cmd_in_nm_filtered).toNumpyArray()
    
    
    
    plt.figure()
    plt.clf()
    plt.plot(j_vector, mean_ol_delta_cmd_in_nm, '.-', label='OL')
    plt.xlabel('j mode')
    plt.ylabel(r'$< \Delta c >_t$'+'[nm rms wf]')
    plt.grid('--', alpha=0.3)
    plt.legend(loc='best')
    
    plt.subplots(1, 2, sharex = True, sharey = True)
    plt.subplot(1, 2, 1)
    plt.title('OL Res WF')
    plt.imshow(ol_wf)
    plt.colorbar(orientation='horizontal', label='nm rms wf')
    print(f"OL Mean WF: PtV = {np.ptp(ol_wf)} nm rms wf \t Amp = {ol_wf.std()} nm rms wf")
    
    plt.subplot(1, 2, 2)
    plt.title('OL Res WF Filtered')
    plt.imshow(ol_wf_filtered)
    plt.colorbar(orientation='horizontal', label='nm rms wf')
    print(f"OL Mean WF (Filtered): PtV = {np.ptp(ol_wf_filtered)} nm rms wf \t Amp = {ol_wf_filtered.std()} nm rms wf")
    
    stda_cl._ol_cmds = ol_delta_cmds_cube_in_nm*1e-9
    stda_cl.display_residual_wavefront(display_ol = True)
    print(f"Residual WF nm rms: OL = {stda_cl._ol_residual_wf.mean()/1e-9} \t CL = {stda_cl._residual_wf[60:].mean(axis=0)/1e-9}")
    
    
    stda_cl._ol_rms_slopes_x = ol_rms_slopes_x_cube
    stda_cl._ol_rms_slopes_y = ol_rms_slopes_y_cube
    stda_cl.display_rms_slopes(display_ol = True)
    
    cl_delta_cmds = stda_cl._delta_cmds[60:,:]
    stda_cl.show_modal_plot(cl_delta_cmds, 'rms')
    
    cl_rms_delta_cmds = stda_cl._rootm_mean_squared(cl_delta_cmds, axis=0)
    ol_rms_delta_cmds = stda_cl._rootm_mean_squared(stda_cl._ol_cmds, axis = 0)
    plt.figure()
    plt.clf()
    plt.loglog(j_vector, ol_rms_delta_cmds/cl_rms_delta_cmds, '.-')
    plt.ylabel('Rejection ratio')
    plt.xlabel('j mode')
    plt.grid('--', alpha=0.3)
    
    last_integ_cmd_in_nm = stda_cl._integ_cmds[-1,:]/1e-9
    cl_integ_wf = stda_cl._slm_rasterizer.zernike_coefficients_to_raster(last_integ_cmd_in_nm).toNumpyArray()
    last_integ_cmd_in_nm_filtered = last_integ_cmd_in_nm.copy()
    last_integ_cmd_in_nm_filtered[:3] = 0
    cl_integ_wf_filtered = stda_cl._slm_rasterizer.zernike_coefficients_to_raster(last_integ_cmd_in_nm_filtered).toNumpyArray()
    
    plt.subplots(1, 2, sharex = True, sharey= True)
    plt.subplot(1, 2, 1)
    plt.title('CL Integ WF')
    plt.imshow(cl_integ_wf)
    plt.colorbar(orientation='horizontal', label='nm rms wf')
    print(f"CL Integ WF: PtV = {np.ptp(cl_integ_wf)} nm rms wf \t Amp = {cl_integ_wf.std()} nm rms wf")
    
    plt.subplot(1, 2, 2)
    plt.title('CL Integ WF Filtered')
    plt.imshow(cl_integ_wf_filtered)
    plt.colorbar(orientation='horizontal', label='nm rms wf')
    print(f"CL Integ WF (Filtered): PtV = {np.ptp(cl_integ_wf_filtered)} nm rms wf \t Amp = {cl_integ_wf_filtered.std()} nm rms wf")
    
    
    mean_cl_delta_cmds = cl_delta_cmds.mean(axis=0)
    cl_res_wf = stda_cl._slm_rasterizer.zernike_coefficients_to_raster(mean_cl_delta_cmds).toNumpyArray()/1e-9
    #mean_cl_delta_cmds_filterd = mean_cl_delta_cmds.copy()
    #mean_cl_delta_cmds_filterd[:3] = 0
    #cl_res_wf_filtered = stda_cl._slm_rasterizer.zernike_coefficients_to_raster(mean_cl_delta_cmds_filterd).toNumpyArray()/1e-9
    
    plt.figure()
    plt.clf()
    plt.title('CL Res WF')
    plt.imshow(cl_res_wf)
    plt.colorbar(orientation='horizontal', label='nm rms wf')
    print(f"CL Res WF: PtV = {np.ptp(cl_res_wf)} nm rms wf \t Amp = {cl_res_wf.std()} nm rms wf")
    

    
    plt.figure()
    plt.clf()
    plt.plot(j_vector, mean_cl_delta_cmds/1e-9, '.-', label='CL')
    plt.xlabel('j mode')
    plt.ylabel(r'$< \Delta c >_t$'+'[nm rms wf]')
    plt.grid('--', alpha=0.3)
    plt.legend(loc='best')
    
    return stda_cl

def main250709_turb_loop():
    
    cl_ftag = '250626_184200'
    ol_ftag = '250626_184300'
    Nmodes = 200
    j_vector = np.arange(Nmodes)+2
    
    stda_cl = ScaoTelemetryDataAnalyser(cl_ftag)
    stda_ol = ScaoTelemetryDataAnalyser(ol_ftag)
    
    stda_cl._ol_cmds = stda_ol._delta_cmds
    stda_cl.display_residual_wavefront(display_ol = True)
    
    stda_cl._ol_rms_slopes_x = stda_ol._rms_slopes_x
    stda_cl._ol_rms_slopes_y = stda_ol._rms_slopes_y
    
    stda_cl.display_rms_slopes(display_ol = True)

    cl_delta_cmds  = stda_cl._delta_cmds[50:,]
    stda_cl.show_modal_plot(cl_delta_cmds,'rms')
    
    cl_rms_delta_cmds = stda_cl._rootm_mean_squared(cl_delta_cmds, axis=0)
    ol_rms_delta_cmds = stda_cl._rootm_mean_squared(stda_cl._ol_cmds, axis = 0)
    plt.figure()
    plt.clf()
    plt.loglog(j_vector, ol_rms_delta_cmds/cl_rms_delta_cmds, '.-')
    plt.ylabel('Rejection ratio')
    plt.xlabel('j mode')
    plt.grid('--', alpha=0.3)
    
    print(f"Residual WF nm rms: OL = {stda_cl._ol_residual_wf.mean()/1e-9} \t CL = {stda_cl._residual_wf[50:].mean(axis=0)/1e-9}")
    
    return stda_cl, stda_ol
    
def main250710_inspect_sampling(slm_rasterizer, mode_index):
    
    amp = 100e-9
    zc = np.zeros(200)
    zc[mode_index] = amp
    yc = 579
    xc = 968
    R = 545
    wf =  slm_rasterizer.zernike_coefficients_to_raster(zc).toNumpyArray()
    #wf_shape = wf.shape
    
    plt.figure(101)
    plt.clf()
    plt.imshow(wf)
    plt.colorbar()
    
    subap_size_on_slm = 26

    plt.figure(102)
    plt.clf()
    plt.title('Mode index %d X profile'%mode_index)
    plt.plot(wf[yc,:])
    vline_vect_alongx = np.arange(-R, R+subap_size_on_slm, subap_size_on_slm) + xc
    plt.vlines(x=vline_vect_alongx, ymin = wf[yc,:].min(), ymax=wf[yc,:].max(), colors='g', linestyles='--')
    plt.grid('--', alpha=0.3)
    
    plt.figure(103)
    plt.clf()
    plt.title('Mode index %d Y profile'%mode_index)
    plt.plot(wf[:,xc])
    vline_vect_alongy = np.arange(-R, R+subap_size_on_slm, subap_size_on_slm) + yc
    plt.vlines(x=vline_vect_alongy, ymin = wf[:,xc].min(), ymax=wf[:,xc].max(), colors='g', linestyles='--')
    plt.grid('--', alpha=0.3)