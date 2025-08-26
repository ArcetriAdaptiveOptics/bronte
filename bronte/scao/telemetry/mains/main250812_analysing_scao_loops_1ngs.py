from bronte.scao.telemetry.scao_telemetry_data_analyser import ScaoTelemetryDataAnalyser
import numpy as np
import matplotlib.pyplot as plt
from bronte.types.slm_pupil_mask_generator import SlmPupilMaskGenerator
from bronte.wfs.kl_slm_rasterizer import KLSlmRasterizer
from bronte.wfs.slm_rasterizer import SlmRasterizer


def main_kl_loop(ol_ftag, cl_ftag, base, ifs_ftag, k):
    '''
    Telemetry data using measured KL control matrices
    '''
    
    ## loading data sets
    stda_cl = ScaoTelemetryDataAnalyser(cl_ftag)
    stda_ol = ScaoTelemetryDataAnalyser(ol_ftag)
    
    stda_cl._ol_cmds = stda_ol._delta_cmds
    stda_cl._ol_rms_slopes_x = stda_ol._rms_slopes_x
    stda_cl._ol_rms_slopes_y = stda_ol._rms_slopes_y
    Nmodes = stda_cl._delta_cmds.shape[-1]
    
    # Inspecting temporal evolution of residual WF
    ol_dcmd_std = stda_ol._delta_cmds.std(axis=0)#in nm
    # this is true only for OL without turbulence
    measurement_error_in_res_wf = stda_ol._root_squared_sum(ol_dcmd_std, axis=0)
    # convergence thr
    res_wf_thr_in_nm = k*measurement_error_in_res_wf/1e-9
    print(f"Measurement Error(rms(ol_dcmds.std())): {measurement_error_in_res_wf/1e-9:.2f}[nm rms]")
    stda_cl.display_residual_wavefront(display_ol = True, res_wf_thr=res_wf_thr_in_nm)
    conv_idx = stda_cl._get_convergence_idx_from_res_wf_thr(res_wf_thr_in_nm)
    
    #computing mean residual wavefront in CLOSE loop and convergence regime
    cl_res_wf_in_nm = stda_cl._residual_wf[conv_idx:].mean()/1e-9
    # computing mean residual wf in OPEN loop
    ol_res_wf_in_nm = stda_cl._ol_residual_wf.mean()/1e-9
    
    print(f"Residual WF [nm rms]: OL = {ol_res_wf_in_nm:.0f}, \t CL = {cl_res_wf_in_nm:.0f}")
    
    # inspecting convergence regime and temporal stability 
    # looking at reconstructed kl coefficinets (delta command)
    #low order modes in common with zernike temporal evolution
    stda_cl.display_delta_cmds_temporal_evolution([2,3,4,5,6])#, display_ol=True)
    #high order modes temporal evolution
    stda_cl.display_delta_cmds_temporal_evolution([100,133,175,199])
    
    
    # modal plot an rejection ratio inspection and analysis
    conv_cl_dcmd = stda_cl._delta_cmds[conv_idx:,:]
    stda_cl.show_modal_plot(cl_delta_cmds = conv_cl_dcmd, rms_or_std='rms')
    
    cl_rms_delta_cmds = stda_cl._rootm_mean_squared(conv_cl_dcmd, axis=0)
    ol_rms_delta_cmds = stda_cl._rootm_mean_squared(stda_cl._ol_cmds, axis = 0)
    j_vector = np.arange(Nmodes)+2
    plt.figure()
    plt.clf()
    plt.loglog(j_vector,ol_rms_delta_cmds/cl_rms_delta_cmds, '.-')
    plt.ylabel('Rejection ratio')
    plt.xlabel('Mode Index')
    plt.grid('--', alpha=0.3)
    
    # Inspecting temporal evolution of slopes rms
    stda_cl.display_rms_slopes(display_ol = True)
    Nsub = stda_ol._slopes_vect.shape[-1]//2
    #computing temporal fluctuation of sloeps in each subap as temporal std
    sigma_ol_sx = stda_ol._slopes_vect[:, :Nsub].std(axis=0)
    sigma_ol_sy = stda_ol._slopes_vect[:, Nsub:].std(axis=0)
    
    
    #computing rms along subaps of slopes temporal fluctuations
    rms_sigma_x = stda_ol._rootm_mean_squared(sigma_ol_sx,axis=0)
    rms_sigma_y = stda_ol._rootm_mean_squared(sigma_ol_sy,axis=0)
    wfs_noise_rms = np.sqrt(rms_sigma_x**2+rms_sigma_y**2)
    dla = 144e-6
    fla = 8.31477e-3
    local_residual_ptv = (wfs_noise_rms * dla**2 * 0.5/fla)/1e-9
    #local_residual_rms = local_residual_ptv/4
    print(f"WFS Noise RMS = {wfs_noise_rms} [nu] --> local-PtV = {local_residual_ptv:0.2f} nm")
    
    cl_slope_res_x = stda_cl._slopes_vect[conv_idx:,:Nsub].std(axis=0)
    cl_slope_res_y = stda_cl._slopes_vect[conv_idx:, Nsub:].std(axis=0)
    rms_fluc_x = stda_cl._rootm_mean_squared(cl_slope_res_x)
    rms_fluc_y = stda_cl._rootm_mean_squared(cl_slope_res_y)
    rms_fluc = np.sqrt(rms_fluc_x**2+rms_fluc_y**2)
    
    print(f"RMS Fluctuant CL = {rms_fluc}")
    
    # Inspecting residual and integrated WF map

    
    ol_delta_kl_coeff = stda_ol._delta_cmds.mean(axis=0)/1e-9
    ol_delta_kl_coeff_filtered = filter_tt_and_focus(ol_delta_kl_coeff)
    
    cl_integ_kl_coeff = stda_cl._integ_cmds[conv_idx:,:].mean(axis=0)/1e-9
    cl_integ_kl_coeff_filtered = filter_tt_and_focus(cl_integ_kl_coeff)
    cl_delta_kl_coeff = stda_cl._delta_cmds[-1,:]/1e-9#stda_cl._delta_cmds[conv_idx:,:].mean(axis=0)/1e-9
    cl_delta_kl_coeff_filtered = filter_tt_and_focus(cl_delta_kl_coeff)
    
    slm_radius = 545
    slm_pup_center = (579, 968)
    spg = SlmPupilMaskGenerator(pupil_radius = slm_radius, pupil_center = slm_pup_center)
    slm_pupil_mask = spg.circular_pupil_mask()
    mc2r = ModalCoefficients2Raster(slm_pupil_mask, base, Nmodes, ifs_ftag)
    coeff2rast = mc2r.get_wf_from_modal_coefficients
    
    ol_wf = coeff2rast(ol_delta_kl_coeff)
    ol_wf_filtered = coeff2rast(ol_delta_kl_coeff_filtered)
    display_filtered_and_full_wf(ol_wf, ol_wf_filtered, 'Open-Loop')
    print(f"OL WF: PtV = {np.ptp(ol_wf):.0f} nm rms wf \t Amp = {ol_wf.std():.0f} nm rms wf")
    print(f"OL WF (Filtered): PtV = {np.ptp(ol_wf_filtered):.0f} nm rms wf \t Amp = {ol_wf_filtered.std():.0f} nm rms wf")
    
    cl_integ_wf = coeff2rast(cl_integ_kl_coeff)
    cl_integ_wf_filtered = coeff2rast(cl_integ_kl_coeff_filtered)
    display_filtered_and_full_wf(cl_integ_wf, cl_integ_wf_filtered,'CL-Integrated WF')
    print(f"CL INTEG WF: PtV = {np.ptp(cl_integ_wf):.0f} nm rms wf \t Amp = {cl_integ_wf.std():.0f} nm rms wf")
    print(f"CL INTEG WF (Filtered): PtV = {np.ptp(cl_integ_wf_filtered):.0f} nm rms wf \t Amp = {cl_integ_wf_filtered.std():.0f} nm rms wf")
    
    cl_delta_wf = coeff2rast(cl_delta_kl_coeff)
    cl_delta_wf_filtered = coeff2rast(cl_delta_kl_coeff_filtered)
    display_filtered_and_full_wf(cl_delta_wf, cl_delta_wf_filtered,'CL-Residual WF')
    print(f"CL RES WF: PtV = {np.ptp(cl_delta_wf):.0f} nm rms wf \t Amp = {cl_delta_wf.std():.2f} nm rms wf")
    print(f"CL RES WF (Filtered): PtV = {np.ptp(cl_delta_wf_filtered):.0f} nm rms wf \t Amp = {cl_delta_wf_filtered.std():.2f} nm rms wf")
    
    
    return stda_cl, stda_ol


def display_filtered_and_full_wf(full_wf_in_nm, filtered_wf_in_nm, sup_title_str='CL'):
    
    plt.subplots(1, 2, sharex = True, sharey= True)
    plt.suptitle(sup_title_str)
    plt.subplot(1, 2, 1)
    plt.title('Full WF')
    plt.imshow(full_wf_in_nm)
    plt.colorbar(orientation='horizontal', label='nm rms wf')
    plt.subplot(1, 2, 2)
    plt.title('Filtered WF')
    plt.imshow(filtered_wf_in_nm)
    plt.colorbar(orientation='horizontal', label='nm rms wf')
    
def filter_tt_and_focus(modal_coeff):
    temp = modal_coeff.copy()
    temp[:3] = 0.
    filtered_modal_cmd = temp
    return filtered_modal_cmd

def load_coefficients2raster(slm_pupil_mask, base = 'kl', Nmodes=200, ifs_ftag=None):
    
    if base == 'kl':
        kl_sr =  KLSlmRasterizer(slm_pupil_mask, ifs_ftag)
        return kl_sr.kl_coefficients_to_raster
    
    if base == 'zernike':
        zc_sr = SlmRasterizer(slm_pupil_mask, Nmodes)
        zc_sr.RAST_AS_NUMPY = True
        return zc_sr.zernike_coefficients_to_raster

class ModalCoefficients2Raster():
    
    def __init__(self, slm_pupil_mask, base = 'kl', Nmodes=200, ifs_ftag=None):
        
        self._modal_base = base
        self._slm_pupil_mask = slm_pupil_mask
        self._Nmodes = Nmodes
        self._ifs_ftag = ifs_ftag
        
        if self._modal_base == 'kl':
            kl_sr =  KLSlmRasterizer(self._slm_pupil_mask, self._ifs_ftag)
            self._coef2rast = kl_sr.kl_coefficients_to_raster
            self._sign_corr = 1
    
        if self._modal_base == 'zernike':
            zc_sr = SlmRasterizer(self._slm_pupil_mask, self._Nmodes)
            self._coef2rast = zc_sr.zernike_coefficients_to_raster
            self._sign_corr = -1
            
    def get_wf_from_modal_coefficients(self, modal_coeff):
        
        wf = self._coef2rast(modal_coeff)
        
        if isinstance(wf, np.ndarray):
            return wf
        else:
            return wf.toNumpyArray()
        
            
#####___________mains____________

def main250812_084900():
    '''
    Telemetry data without turbulence using measured
    KL control matrices of 200 modes
    '''
    modal_base = 'kl'
    ol_ftag = '250808_151100' # Nstep=300 dt=1ms Nmodes=200
    cl_ftag = '250808_151500' # gain=-0.3
    ifs_ftag = '250806_170800' # L0=25m,r0=15cm,D=8.2m
    stda_cl, stda_ol = main_kl_loop(ol_ftag, cl_ftag, modal_base, ifs_ftag, k=3)
    
    return stda_cl, stda_ol 

def main250813_101300():
    '''
    Telemetry data analysis with turbulence using measured
    KL control matrices of 200 modes
    '''
    modal_base = 'kl'
    ol_ftag = '250808_152700_tris' # Nstep=300 dt=1ms Nmodes=200
    cl_ftag = '250808_153900_tris' # gain=-0.3
    ifs_ftag = '250806_170800'# L0=25m,r0=15cm,D=8.2m
    stda_cl, stda_ol = main_kl_loop(ol_ftag, cl_ftag, modal_base, ifs_ftag, k=3/12)
    
    return stda_cl, stda_ol 

def main250813_110600():
    '''
    Telemetry data without turbulence using measured
    Zernike control matrices of 200 modes
    '''
    modal_base = 'zernike'
    ol_ftag = '250808_161900' # Nstep=300 dt=1ms Nmodes=200
    cl_ftag = '250808_162500' # gain=-0.3
    ifs_ftag = None 
    stda_cl, stda_ol = main_kl_loop(ol_ftag, cl_ftag, modal_base, ifs_ftag, k=12)
    
    return stda_cl, stda_ol

def main250813_113300():
    '''
    Telemetry data with turbulence using measured
    Zernike control matrices of 200 modes
    '''
    modal_base = 'zernike'
    ol_ftag = '250808_155500' # Nstep=300 dt=1ms Nmodes=200
    cl_ftag = '250808_160500' # gain=-0.3 # L0=25m,r0=15cm,D=8.2m
    ifs_ftag = None 
    stda_cl, stda_ol = main_kl_loop(ol_ftag, cl_ftag, modal_base, ifs_ftag, k=12)
    
    return stda_cl, stda_ol 

def main250813_114600():
    '''
    Telemetry data without turbulence using measured
    Zernike control matrices of 200 modes
    '''
    modal_base = 'kl'
    ol_ftag = '250808_134700' # Nstep=100 dt=1ms Nmodes=200
    cl_ftag = '250808_135900' # gain=-0.1
    ifs_ftag = '250808_092602'#L0=40,seeng=0.5arcsec,D=8m 
    stda_cl, stda_ol = main_kl_loop(ol_ftag, cl_ftag, modal_base, ifs_ftag, k=3)
    
    return stda_cl, stda_ol

def main250813_123700():
    '''
    Telemetry data analysis with turbulence using measured
    KL control matrices of 200 modes
    '''
    modal_base = 'kl'
    ol_ftag = '250808_140700' # Nstep=100 dt=1ms Nmodes=200
    cl_ftag = '250808_141500' # gain=-0.1
    ifs_ftag = '250808_092602'# #L0=40,seeng=0.5arcsec,D=8m 
    stda_cl, stda_ol = main_kl_loop(ol_ftag, cl_ftag, modal_base, ifs_ftag, k=1.5)
    
    return stda_cl, stda_ol 

def main250813_125400():
    '''
    Telemetry data without turbulence using measured
    Zernike control matrices of 200 modes
    '''
    modal_base = 'zernike'
    ol_ftag = '250804_111500' # Nstep=100 dt=1ms Nmodes=200
    cl_ftag = '250804_112600' # gain=-0.1
    ifs_ftag = None #L0=40,seeng=0.5arcsec,D=8m 
    stda_cl, stda_ol = main_kl_loop(ol_ftag, cl_ftag, modal_base, ifs_ftag, k=3)
    
    return stda_cl, stda_ol

def main250813_151700():
    '''
    Telemetry data analysis with turbulence using measured
    KL control matrices of 200 modes
    '''
    modal_base = 'kl'
    ol_ftag = '250808_140700' # Nstep=100 dt=1ms Nmodes=200
    cl_ftag = '250808_142500' # gain=-0.3
    ifs_ftag = '250808_092602'# #L0=40,seeng=0.5arcsec,D=8m 
    stda_cl, stda_ol = main_kl_loop(ol_ftag, cl_ftag, modal_base, ifs_ftag, k=1.5)
    
    return stda_cl, stda_ol 

def main250825_162400():
    '''
    Telemetry data analysis with turbulence using measured
    KL control matrices of 200 modes
    '''
    modal_base = 'kl'
    ol_ftag = '250825_153300_bis' # Nstep=300 dt=1ms Nmodes=200
    cl_ftag = '250825_154200_bis' # gain=-0.3
    ifs_ftag = '250806_170800'# L0=25m,r0=15cm,D=8.2m
    stda_cl, stda_ol = main_kl_loop(ol_ftag, cl_ftag, modal_base, ifs_ftag, k=0.25)
    
    return stda_cl, stda_ol 
