from bronte.scao.telemetry.scao_telemetry_data_analyser import ScaoTelemetryDataAnalyser
import numpy as np
import matplotlib.pyplot as plt
from bronte.wfs.kl_slm_rasterizer import KLSlmRasterizer
from arte.types.mask import CircularMask
from bronte.scao.telemetry.mains.main250812_analysing_scao_loops_1ngs import filter_tt_and_focus
from bronte.scao.telemetry.mains.main250829_compute_scao_wf_res import load_residual_wf
from bronte.utils.scao_error_budget_computer import ScaoErrorBudgetComputer

def main(turb_cl_ftag, turb_ol_ftag, mifs_ftag, conv_index = 75, dispWFmap = False):
    
    stda_cl = ScaoTelemetryDataAnalyser(turb_cl_ftag)
    stda_ol = ScaoTelemetryDataAnalyser(turb_ol_ftag)
    
    stda_cl._ol_cmds = stda_ol._delta_cmds
    Nmodes = stda_cl._delta_cmds.shape[-1]
    stda_cl._ol_rms_slopes_x = stda_ol._rms_slopes_x
    stda_cl._ol_rms_slopes_y = stda_ol._rms_slopes_y
    
    stda_cl.display_residual_wavefront(display_ol = True)
    stda_cl.display_rms_slopes(display_ol=True)
    measured_cl_res_wf_in_nm = stda_cl._residual_wf[conv_index:].mean()/1e-9
    measured_ol_wfe_in_nm =  stda_cl._ol_residual_wf.mean()/1e-9
    print(f"slope_vector shape: {stda_cl._slopes_vect.shape}")
    print(f"Measured OL WFE [nm rms wf]: {measured_ol_wfe_in_nm:.0f} (on a base of 200 kl modes)")
    print(f"Measured CL Residual WF [nm rms wf]: {measured_cl_res_wf_in_nm:.0f} (on a base of 200 kl modes)")
    
    # modal plot an rejection ratio inspection and analysis
    conv_cl_dcmd = stda_cl._delta_cmds[conv_index:,:]
    stda_cl.show_modal_plot(cl_delta_cmds = conv_cl_dcmd, rms_or_std='std')
    
    cl_rms_delta_cmds = stda_cl._rootm_mean_squared(conv_cl_dcmd, axis=0)
    ol_rms_delta_cmds = stda_cl._rootm_mean_squared(stda_cl._ol_cmds, axis = 0)
    
    
    j_vector = np.arange(Nmodes)+2
    plt.figure()
    plt.clf()
    plt.loglog(j_vector,ol_rms_delta_cmds/cl_rms_delta_cmds, '.-')
    plt.ylabel('Rejection ratio ' + "$\sigma_{OL}/\sigma_{CL}$")
    plt.xlabel('Mode Index')
    plt.grid('--', alpha=0.3)
    
    plt.figure()
    plt.clf()
    plt.loglog(j_vector,1 - (cl_rms_delta_cmds**2/ol_rms_delta_cmds**2), '.-')
    plt.ylabel('Removed variance ratio ' + "$1 \ - \ (\sigma^2_{CL}/\sigma^2_{OL})$")
    plt.xlabel('Mode Index')
    plt.grid('--', alpha=0.3)
    
    rejection_total = ((ol_rms_delta_cmds**2).sum())/((cl_rms_delta_cmds**2).sum())
    print(f"TOTAL REJECTION: {rejection_total}")
    #computing average wf map residual in convergence regime
    if dispWFmap is True:
        mean_cl_dcmd_conv_in_nm = conv_cl_dcmd.mean(axis=0)/1e-9 
        compute_and_display_wf_from_modal_coeff(mean_cl_dcmd_conv_in_nm, mifs_ftag, 'Mean Res WF')
    
    tot_res_wf_in_nm = get_residual_wf_from_slm_displayed_wfs(turb_cl_ftag) 
    
    tot_var_at500nm = (tot_res_wf_in_nm*2*np.pi/500)**2
    sr_at500nm = np.exp(-tot_var_at500nm)
    tot_var_at633nm = (tot_res_wf_in_nm*2*np.pi/633)**2
    sr_at633nm = np.exp(-tot_var_at633nm)
    
    # plt.figure()
    # plt.clf()
    # plt.plot(tot_res_wf_in_nm)
    # plt.xlabel('N steps')
    # plt.ylabel('Total Wavefront Error  [nm rms wf]')
    # plt.grid('--',alpha=0.3)
    #
    # sr_exp_at_500 = np.exp(-1*(2*np.pi*(142/500)**2))
    # sr_exp_at_633 = np.exp(-1*(2*np.pi*(142/633)**2))
    #
    # plt.figure()
    # plt.clf()
    # plt.plot(sr_at500nm, '.-', label = 'SR@500nm')
    # plt.plot(sr_at633nm, '.-', label = 'SR@633nm')
    # plt.xlabel('N steps')
    # plt.ylabel('Strhel Ratio')
    # plt.legend(loc='best')
    # plt.grid('--', alpha=0.3)
    sigma_nm = 146.0  # WFE residuo in nm

    sr_exp_at_500 = np.exp(-(2*np.pi * (sigma_nm/500.0))**2)
    sr_exp_at_633 = np.exp(-(2*np.pi * (sigma_nm/633.0))**2)
    
    plt.figure()
    plt.clf()
    # serie misurate
    plt.plot(sr_at500nm, '.-', label='SR @ 500 nm', color='C0')
    plt.plot(sr_at633nm, '.-', label='SR @ 633 nm', color='C1')
    
    # linee orizzontali attese (dashed)
    plt.axhline(sr_exp_at_500, color='C0', linestyle='--', linewidth=1.5,
                label=f'expected @ 500 nm ≈ {sr_exp_at_500:.3f}')
    plt.axhline(sr_exp_at_633, color='C1', linestyle='--', linewidth=1.5,
                label=f'expected @ 633 nm ≈ {sr_exp_at_633:.3f}')
    
    plt.xlabel('N steps')
    plt.ylabel('Strehl Ratio')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    print('SR diff meas-exp STD')
    print((sr_at633nm[50:]-sr_exp_at_633).std())
    print((sr_at500nm[50:]-sr_exp_at_500).std())
    
    mean_tot_res_wf_in_nm = tot_res_wf_in_nm[50:].mean()
    exp_fitting_err = compute_approximated_exp_fitting_error_from_vk()
    print(f"Total Residual WF (CL): {mean_tot_res_wf_in_nm :.0f} nm rms wf")
    print(f"Expected Fitting error (VK): {exp_fitting_err :.0f} nm rms wf")
    
def compute_and_display_wf_from_modal_coeff(modal_coeff_in_nm, mifs_ftag, sup_title_str = 'RES WF'):
      
    cmask = CircularMask(frameShape=(1152,1920), maskCenter=(579, 968), maskRadius=545)
    sr = KLSlmRasterizer(cmask, mifs_ftag)
    slm_pupil_mask = sr.slm_pupil_mask
    
    full_wf = sr.kl_coefficients_to_raster(modal_coeff_in_nm)
    filtered_modal_coeff_in_nm = filter_tt_and_focus(modal_coeff_in_nm)
    filtered_wf = sr.kl_coefficients_to_raster(filtered_modal_coeff_in_nm)
    print(f"{sup_title_str} (full) [nm rms wf]: PtV = {np.ptp(full_wf ):.0f}  Amp rms = {full_wf.std():.0f}")
    print(f"{sup_title_str} (TTF-filtered) [nm rms wf]: PtV = {np.ptp(filtered_wf):.0f}  Amp rms = {filtered_wf.std():.0f}")
    
    display_filtered_and_full_wf(full_wf, filtered_wf, sup_title_str)


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

def get_residual_wf_from_slm_displayed_wfs(ftag):
    res_wf_in_nm, _ = load_residual_wf(ftag)
    return res_wf_in_nm

def compute_approximated_exp_fitting_error_from_vk():
    wl = 500e-9
    r0 = 0.15
    L0 = 25
    Nmodes  = 200
    Dtel = 8.2
    single_cell_area = (np.pi*(Dtel/2)**2)/Nmodes
    single_cell_radius = np.sqrt(single_cell_area/np.pi)
    eq_dm_pitch = 2*single_cell_radius
    seb = ScaoErrorBudgetComputer(wl, r0, L0)
    var = seb.get_fitting_var_vk_closed(None, d_dm = eq_dm_pitch)
    fitting_err_in_nm = seb.phase_var2wfe_in_nm(var, wl)
    
    return fitting_err_in_nm
    
####
# mains 

def main250829_120000():
    
    turb_cl_ftag = '250829_120000'
    turb_ol_ftag = '250829_114300'
    mifs_ftag = '250806_170800'
    
    main(turb_cl_ftag, turb_ol_ftag, mifs_ftag, conv_index = 75)#, dispWFmap = True)

def main250901_122900():
    
    turb_cl_ftag = '250901_122900'
    turb_ol_ftag = '250901_121100'
    mifs_ftag = '250806_170800'
    main(turb_cl_ftag, turb_ol_ftag, mifs_ftag, conv_index = 75)
    
def main250901_125700():
    
    turb_cl_ftag = '250901_125700'
    turb_ol_ftag = '250901_124500'
    mifs_ftag = '250806_170800'
    main(turb_cl_ftag, turb_ol_ftag, mifs_ftag, conv_index = 75)
    
def main250903_111200():
    turb_cl_ftag = '250903_111200'
    turb_ol_ftag = '250829_114300'
    mifs_ftag = '250806_170800'
    main(turb_cl_ftag, turb_ol_ftag, mifs_ftag, conv_index = 75)

def main250903_155600():
    turb_cl_ftag = '250903_155600'
    turb_ol_ftag = '250829_114300'
    mifs_ftag = '250806_170800'
    main(turb_cl_ftag, turb_ol_ftag, mifs_ftag, conv_index = 75)