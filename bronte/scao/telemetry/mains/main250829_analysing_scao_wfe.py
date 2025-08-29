import numpy as np
import matplotlib.pyplot as plt
from bronte.scao.telemetry.scao_telemetry_data_analyser import ScaoTelemetryDataAnalyser
from bronte.scao.telemetry.displayed_wavefront_analyser import DisplayedWavefrontAnalyser
from bronte.wfs.kl_slm_rasterizer import KLSlmRasterizer
from bronte.utils.scao_error_budget_computer import ScaoErrorBudgetComputer
from arte.types.mask import CircularMask

def main():
    
    ol_noturb_ftag = '250829_111600'
    #cl_dispwf_ftag = '250929_120000'
    cl_turb_ftag = '250829_120000'
    ol_turb_ftag = '250829_114300'
    mifs_ftag = '250806_170800'
    
    cmask = CircularMask(frameShape=(1152,1920), maskCenter=(579, 968), maskRadius=545)
    sr = KLSlmRasterizer(cmask, mifs_ftag)
    slm_pupil_mask = sr.slm_pupil_mask
    
    stda_ol_noturb = ScaoTelemetryDataAnalyser(ol_noturb_ftag)
    dwa = DisplayedWavefrontAnalyser(cl_turb_ftag)
    dwa.set_slm_pupil_mask(slm_pupil_mask)
    dwa.apply_slm_pupil_mask_on_displayed_wf()
    
    mean_static_dcmds = stda_ol_noturb._delta_cmds.mean(axis=0)
    static_wf = sr.kl_coefficients_to_raster(mean_static_dcmds)/1e-9
    
    wfe = np.zeros(dwa._Nwf)
    
    for idx in range(dwa._Nwf):
        wf_diff = dwa._wf_cube_on_slm[idx] - static_wf
        wfe[idx] = wf_diff.std()
    
    
    
    return stda_ol_noturb, dwa, sr, static_wf, wfe
    
