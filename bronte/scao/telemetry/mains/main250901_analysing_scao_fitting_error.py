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

def main_compute():
    
    ol_tag = '250901_121100'
    mifs_ftag = '250806_170800'
    Nmodes = 200
    cmask = CircularMask(frameShape=(1152,1920), maskCenter=(579, 968), maskRadius=545)
    sr = KLSlmRasterizer(cmask, mifs_ftag)
    slm_pupil_mask = sr.slm_pupil_mask
    sr.load_synthetic_kl_intmat(mifs_ftag)
    
    dwa = DisplayedWavefrontAnalyser(ol_tag)
    dwa.set_slm_pupil_mask(slm_pupil_mask)
    dwa.apply_slm_pupil_mask_on_displayed_wf()
    
    kl_modes = np.zeros((dwa._Nwf, Nmodes))
    wf_diff_rms_amp = np.zeros(dwa._Nwf)
    wf_on_slm_rms_amp = np.zeros(dwa._Nwf)
    
    for idx in range(dwa._Nwf):
        print(idx)
        wf_on_slm = dwa._wf_cube_on_slm[idx]
        wf_on_slm_rms_amp[idx] = wf_on_slm.std()
        
        kl_modes[idx] = sr.decompose_wf(wf_on_slm)
        rec_wf = sr.kl_coefficients_to_raster(kl_modes[idx])
        
        wf_diff = wf_on_slm - rec_wf
        wf_diff_rms_amp[idx] = wf_diff.std()
    
    
    set_data_dir()
    ftag = '250901_160700'
    fname = other_folder()/(ftag + '.fits')
    hdr = fits.Header()
    hdr['OL_TAG'] = ol_tag
    hdr['MIFS_TAG'] = mifs_ftag
    fits.writeto(fname,kl_modes, hdr)
    fits.append(fname, wf_on_slm_rms_amp)
    fits.append(fname, wf_diff_rms_amp)

def main_plot():
    
    ftag  = '250901_160700'
    set_data_dir()
    fname = other_folder()/(ftag + '.fits')
    hdulist = fits.open(fname)
    
    kl_modes = hdulist[0].data
    wf_on_slm_rms_amp = hdulist[1].data
    wf_diff_rms_amp = hdulist[2].data
    
    plt.figure()
    plt.clf()
    plt.plot(wf_on_slm_rms_amp)
    plt.plot(wf_diff_rms_amp)