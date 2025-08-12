from bronte.scao.telemetry.scao_telemetry_data_analyser import ScaoTelemetryDataAnalyser
import numpy as np
import matplotlib.pyplot as plt
#from astropy.io import fits
#from bronte.utils.slopes_covariance_matrix_analyser import SlopesCovariaceMatrixAnalyser


def main250812_084900():
    '''
    Telemetry data without turbulence using measured
    KL control matrices of 200 modes
    '''
    
    ## loading data sets
    ol_ftag = '250808_151100'
    cl_ftag = '250808_151500'
    
    stda_cl = ScaoTelemetryDataAnalyser(cl_ftag)
    stda_ol = ScaoTelemetryDataAnalyser(ol_ftag)
    
    stda_cl._ol_cmds = stda_ol._delta_cmds
    stda_cl._ol_rms_slopes_x = stda_ol._rms_slopes_x
    stda_cl._ol_rms_slopes_y = stda_ol._rms_slopes_y
    Nmodes = stda_cl._delta_cmds.shape[-1]
    
    # Inspecting temporal evolution of residual WF
    ol_dcmd_std = stda_ol._delta_cmds.std(axis=0)#in nm
    measurement_error_in_res_wf = stda_ol._root_squared_sum(ol_dcmd_std, axis=0)
    # convergence thr
    res_wf_thr_in_nm = 3*measurement_error_in_res_wf/1e-9
    stda_cl.display_residual_wavefront(display_ol = True, res_wf_thr=res_wf_thr_in_nm)
    conv_idx = stda_cl._get_convergence_idx_from_res_wf_thr(res_wf_thr_in_nm)
    
    #computing mean residual wavefront in CLOSE loop and convergence regime
    cl_res_wf_in_nm = stda_cl._residual_wf[conv_idx:].mean()/1e-9
    # computing mean residual wf in OPEN loop
    ol_res_wf_in_nm = stda_cl._ol_residual_wf.mean()/1e-9
    
    print(f"Residual WF [nm rms]: OL = {ol_res_wf_in_nm:.0f}, \t CL = {cl_res_wf_in_nm:.0f}")
    
    # Inspecting temporal evolution of slopes rms
    stda_cl.display_rms_slopes(display_ol = True)
    
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
    
    
    return stda_cl, stda_ol
   