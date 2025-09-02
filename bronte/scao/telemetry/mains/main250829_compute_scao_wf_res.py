import numpy as np
import matplotlib.pyplot as plt
from bronte.scao.telemetry.scao_telemetry_data_analyser import ScaoTelemetryDataAnalyser
from bronte.scao.telemetry.displayed_wavefront_analyser import DisplayedWavefrontAnalyser
from bronte.wfs.kl_slm_rasterizer import KLSlmRasterizer
from bronte.utils.scao_error_budget_computer import ScaoErrorBudgetComputer
from arte.types.mask import CircularMask
from astropy.io import fits
from bronte.startup import set_data_dir
from bronte.package_data import other_folder



def compare_res_wf():
    set_data_dir()
    ftag_list = ['250829_120000','250901_122900', '250901_125700' ]
    res_wf_list = []
    
    for ftag in ftag_list:
        res_wf_in_nm, _ = load_residual_wf(ftag)
        res_wf_list.append(res_wf_in_nm)
    
    plt.figure()
    plt.clf()
    
    for idx in range(len(res_wf_list)):
        res_wf = res_wf_list[idx]
        plt.plot(res_wf, label = ftag_list[idx])
        mean_res_in_nm = res_wf[50:].mean()
        print(f"res wf : {mean_res_in_nm :.0f} nm rms wf")
        
    plt.xlabel('N steps')
    plt.ylabel('Wavefront Error '+'$\sigma_{res}$'+' [nm rms wf]')
    plt.grid('--',alpha=0.3)
    plt.legend(loc='best')
    
def get_wfe(ftag, slm_pupil_mask, static_wf, save_res_wf = False):
    
    dwa = DisplayedWavefrontAnalyser(ftag)
    dwa.set_slm_pupil_mask(slm_pupil_mask)
    dwa.apply_slm_pupil_mask_on_displayed_wf()
    wfe = np.zeros(dwa._Nwf)
    for idx in range(dwa._Nwf):
        wf_diff = dwa._wf_cube_on_slm[idx] - static_wf
        wfe[idx] = wf_diff.std()
    
    if save_res_wf is True:
        fname = other_folder() / (ftag + '_res_wf.fits')
        fits.writeto(fname, wfe, dwa._hdr)
    return wfe

def load_residual_wf(ftag):
    set_data_dir()
    fname = other_folder() /(ftag + '_res_wf.fits')
    hdr = fits.getheader(fname)
    hdulist = fits.open(fname)
    res_wf_in_nm = hdulist[0].data
    
    return res_wf_in_nm, hdr
    
def show_wf_res(ftag):
    
    wf_res,_ = load_residual_wf(ftag)
    
    plt.figure()
    plt.clf()
    plt.plot(wf_res)
    
    plt.xlabel('N steps')
    plt.ylabel('Wavefront Error '+'$\sigma_{res}$'+' [nm rms wf]')
    plt.grid('--',alpha=0.3)
    mean_res_in_nm = wf_res[50:].mean()

    print(f"res wf for g=-0.3: {mean_res_in_nm :.0f} nm rms wf")
    
####
# mains
def compute_wf_res_from250829_120000():
    
    ol_noturb_ftag = '250829_111600'
    cl_turb_ftag = '250829_120000'
    mifs_ftag = '250806_170800'
    
    cmask = CircularMask(frameShape=(1152,1920), maskCenter=(579, 968), maskRadius=545)
    sr = KLSlmRasterizer(cmask, mifs_ftag)
    slm_pupil_mask = sr.slm_pupil_mask
    
    stda_ol_noturb = ScaoTelemetryDataAnalyser(ol_noturb_ftag)
    mean_static_dcmds = stda_ol_noturb._delta_cmds.mean(axis=0)
    static_wf = sr.kl_coefficients_to_raster(mean_static_dcmds)/1e-9
    
    res_wf_in_nm = get_wfe(cl_turb_ftag, slm_pupil_mask, static_wf, True)
    
    plt.figure()
    plt.clf()
    plt.plot(res_wf_in_nm)
    plt.xlabel('N steps')
    plt.ylabel('Wavefront Error '+'$\sigma_{res}$'+' [nm rms wf]')
    plt.grid('--',alpha=0.3)
    mean_res_in_nm = res_wf_in_nm[50:].mean()
    print(f"res wf : {mean_res_in_nm :.0f} nm rms wf")
    
def compute_wf_res_from250901_122900():
    
    ol_noturb_ftag = '250902_101600'
    cl_turb_ftag = '250901_122900'
    mifs_ftag = '250806_170800'
    
    cmask = CircularMask(frameShape=(1152,1920), maskCenter=(579, 968), maskRadius=545)
    sr = KLSlmRasterizer(cmask, mifs_ftag)
    slm_pupil_mask = sr.slm_pupil_mask
    
    stda_ol_noturb = ScaoTelemetryDataAnalyser(ol_noturb_ftag)
    mean_static_dcmds = stda_ol_noturb._delta_cmds.mean(axis=0)
    static_wf = sr.kl_coefficients_to_raster(mean_static_dcmds)/1e-9
    
    res_wf_in_nm = get_wfe(cl_turb_ftag, slm_pupil_mask, static_wf, True)
    
    plt.figure()
    plt.clf()
    plt.plot(res_wf_in_nm)
    plt.xlabel('N steps')
    plt.ylabel('Wavefront Error '+'$\sigma_{res}$'+' [nm rms wf]')
    plt.grid('--',alpha=0.3)
    mean_res_in_nm = res_wf_in_nm[50:].mean()
    print(f"res wf : {mean_res_in_nm :.0f} nm rms wf")

def compute_wf_res_from250901_125700():
    
    ol_noturb_ftag = '250902_101600'
    cl_turb_ftag = '250901_125700'
    mifs_ftag = '250806_170800'
    
    cmask = CircularMask(frameShape=(1152,1920), maskCenter=(579, 968), maskRadius=545)
    sr = KLSlmRasterizer(cmask, mifs_ftag)
    slm_pupil_mask = sr.slm_pupil_mask
    
    stda_ol_noturb = ScaoTelemetryDataAnalyser(ol_noturb_ftag)
    mean_static_dcmds = stda_ol_noturb._delta_cmds.mean(axis=0)
    static_wf = sr.kl_coefficients_to_raster(mean_static_dcmds)/1e-9
    
    res_wf_in_nm = get_wfe(cl_turb_ftag, slm_pupil_mask, static_wf, True)
    
    plt.figure()
    plt.clf()
    plt.plot(res_wf_in_nm)
    plt.xlabel('N steps')
    plt.ylabel('Wavefront Error '+'$\sigma_{res}$'+' [nm rms wf]')
    plt.grid('--',alpha=0.3)
    mean_res_in_nm = res_wf_in_nm[50:].mean()
    print(f"res wf : {mean_res_in_nm :.0f} nm rms wf")
