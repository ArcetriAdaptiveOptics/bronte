import numpy as np 
from bronte.ncpa import sharpening_analyzer
import matplotlib.pyplot as plt
from bronte.utils.raw_strehl_ratio_computer import StrehlRatioComputer

def load_low_order_sharpening():
    
    fpath = "D:\\phd_slm_edo\\bronte\\other_data\\"
    ftag =  "251001_100300.fits"
    fname = fpath + ftag
    
    sa = sharpening_analyzer.SharpeningAnalyzer(fname)

    #sa.display_sharpening_res()
    #sa.display_sr_interpolation()
    return sa

def load_low_order_finer_sharpening():
    
    fpath = "D:\\phd_slm_edo\\bronte\\other_data\\"
    ftag =  "251001_104600.fits"
    fname = fpath + ftag
    
    sa = sharpening_analyzer.SharpeningAnalyzer(fname)

    #sa.display_sharpening_res()
    #sa.display_sr_interpolation()
    return sa

def load_higer_order_sharpening():
    
    fpath = "D:\\phd_slm_edo\\bronte\\other_data\\"
    ftag =  "251001_123800.fits"
    fname = fpath + ftag
    
    sa = sharpening_analyzer.SharpeningAnalyzer(fname)
    #sa.display_sharpening_res()
    #sa.display_sr_interpolation()

# def main():
#
#     sa = load_low_order_sharpening()
#
#     comp_psf = sa._comp_psf
#     uncomp_psf = sa._uncomp_psf
#
#     amp_span = sa._amp_span
#     measured_sr = sa._measured_sr
#     measured_itot_in_roi = sa._mesured_i_roi
#
#     best_coeff = sa._best_amps
#     corrected_modes_index = sa._corrected_z_modes_indexes
#     interp_sr_func_list = sa._sr_func_list
#     damp = 5e-9
#     amps = np.arange(amp_span.min(), amp_span.max() + damp, damp)
#
#     plt.figure()
#     #ax = plt.gca()
#     for idx, j in enumerate(corrected_modes_index):
#
#
#         plt.plot(amp_span/1e-9, measured_sr[j-2],'o',label='j=%d'%j)
#         color = plt.gca().lines[-1].get_color()
#         plt.plot(amps/1e-9, interp_sr_func_list[idx],'-', color = color)
#
#     plt.xlabel('$c_j$'+' '+ '[nm] rms')
#     plt.ylabel('Strehl Ratio')
#     plt.grid(ls='--',alpha = 0.4)
#     plt.legend(loc='best')

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# ---------- utility: incertezze su SR per ciascun punto ----------
def compute_sigma_SR_per_point(SR_vec, AROI_vec, r_peak_over_sum, gain_e_per_adu, ron_adu, Nroi):
    """
    SR_vec: array SR(k) per un dato modo (lunghezza = len(amp_span))
    AROI_vec: somma ADU nella ROI per ciascun punto k (stessa lunghezza)
    r_peak_over_sum: max(DL_ROI) / sum(DL_ROI) (scalare)
    Ritorna: sigma_SR (stessa lunghezza)
    """
    SR = np.asarray(SR_vec, float)
    AROI = np.asarray(AROI_vec, float)
    g = float(gain_e_per_adu)
    ron = float(ron_adu)

    # picco DL atteso nella ROI
    peakDL = r_peak_over_sum * AROI

    # picco misurato (in ADU)
    Ap = SR * peakDL

    # varianze (in ADU) propagate (photon + RON)
    sigma_Aroi = np.sqrt(AROI / g + Nroi * (ron / g) ** 2)
    sigma_peakDL = r_peak_over_sum * sigma_Aroi
    sigma_Ap = np.sqrt(Ap / g + (ron / g) ** 2)

    eps = np.finfo(float).eps
    sigma_SR = SR * np.sqrt((sigma_Ap / (Ap + eps)) ** 2 + (sigma_peakDL / (peakDL + eps)) ** 2)
    return sigma_SR

# ---------- utility: stima incertezza del massimo via curvatura ----------
def sigma_c_from_curvature(amp_span, SR_vec, sigma_SR_vec, c_hat, dense_step):
    """
    Usa una CubicSpline per stimare la curvatura locale: kappa = - SR''(c_hat).
    Poi delta method:  sigma_c ≈ sqrt(2 * sigma_SR_eff / kappa), con sigma_SR_eff
    media dei 3–5 punti con SR più alto (sul grid originale).
    Somma in quadratura anche l'errore di griglia dense_step/sqrt(12).
    """
    amp_span = np.asarray(amp_span, float)
    SR = np.asarray(SR_vec, float)
    sSR = np.asarray(sigma_SR_vec, float)

    # spline naturale per avere la derivata seconda
    cs = CubicSpline(amp_span, SR, bc_type='natural')

    # curvatura (positiva vicino al massimo): kappa = - SR''(c_hat)
    kappa = float(-cs(c_hat, 2))  # seconda derivata

    # stima sigma_SR eff nei punti migliori del grid originale
    K = min(5, len(SR))
    top = np.argsort(-SR)[:K]
    sigma_SR_eff = float(np.mean(sSR[top]))

    # se la curvatura non è affidabile, torna solo errore di griglia
    if not np.isfinite(kappa) or kappa <= 0 or not np.isfinite(sigma_SR_eff) or sigma_SR_eff <= 0:
        return dense_step / np.sqrt(12.0)

    sigma_c_loc = np.sqrt(2.0 * sigma_SR_eff / kappa)

    # aggiungi errore di discretizzazione della ricerca densa
    sigma_grid = dense_step / np.sqrt(12.0)
    return float(np.sqrt(sigma_c_loc**2 + sigma_grid**2))

# -------------------------- MAIN --------------------------
def main():

    sa = load_low_order_finer_sharpening()

    comp_psf = sa._comp_psf
    uncomp_psf = sa._uncomp_psf

    amp_span = sa._amp_span                # shape (M_points,)
    measured_sr = sa._measured_sr          # shape (N_rows, M_points) (include tip/tilt nelle prime 2 righe)
    measured_itot_in_roi = sa._mesured_i_roi  # shape (N_rows, M_points) (10, 51) nel tuo caso

    best_coeff = sa._best_amps             # già calcolati dalla tua pipeline (argmax spline densa)
    corrected_modes_index = sa._corrected_z_modes_indexes  # es. [4,5,6,7,8,9,10,11]
    interp_sr_func_list = sa._sr_func_list # shape (8, 1001)
    damp = 5e-9                            # 5 nm in metri
    amps = np.arange(amp_span.min(), amp_span.max() + damp, damp)

    # ---- ricava r = peak/sum della PSF DL nella ROI (richiede che sia stata salvata)
    if hasattr(sa, "_au_dl_psf") and sa._au_dl_psf is not None:
        dl_roi = sa._au_dl_psf
        r_peak_over_sum = float(dl_roi.max() / dl_roi.sum())
    else:
        raise RuntimeError("Non trovo sa._au_dl_psf per calcolare r=peak/sum della DL nella ROI.")

    # dimensione ROI (numero di pixel) dalla PSF compensata
    Nroi = int(comp_psf.size)

    gain = 3.5409     # e-/ADU
    ron_adu = 2.4     # ADU

    # contenitori dei risultati
    best_coeff_err = []
    best_coeff_dense = []  # argmax sulla curva interpolata già calcolata (per coerenza di report)

    plt.figure()
    for idx_plot, j in enumerate(corrected_modes_index):

        # mappa indice Noll -> riga negli array (salta tip/tilt)
        row = j - 2  # Z2->row0, Z3->row1, Z4->row2, ...

        SR_row = measured_sr[row, :]
        AROI_row = measured_itot_in_roi[row, :]

        # incertezze su SR per ciascun punto
        sigma_SR_row = compute_sigma_SR_per_point(
            SR_row, AROI_row, r_peak_over_sum, gain_e_per_adu=gain, ron_adu=ron_adu, Nroi=Nroi
        )

        # coefficiente migliore dalla funzione interpolata densa (già precomputata)
        SR_dense = interp_sr_func_list[idx_plot]  # lunghezza = len(amps)=1001
        i_max = int(np.argmax(SR_dense))
        c_hat = float(amps[i_max])
        best_coeff_dense.append(c_hat)

        # incertezza sul massimo via curvatura della spline sui punti ORIGINALI
        sigma_c = sigma_c_from_curvature(amp_span, SR_row, sigma_SR_row, c_hat=c_hat, dense_step=damp)
        best_coeff_err.append(sigma_c)

        # --- plot
        plt.plot(amp_span/1e-9, SR_row, 'o', label=f'j={j}')
        color = plt.gca().lines[-1].get_color()
        plt.plot(amps/1e-9, SR_dense, '-', color=color)
        # segna il massimo con una banda ±1σ
        plt.axvline(c_hat/1e-9, color=color, ls='--', alpha=0.6)
        plt.fill_betweenx([0, 1.05*np.nanmax(SR_dense)],
                          (c_hat - sigma_c)/1e-9, (c_hat + sigma_c)/1e-9,
                          color=color, alpha=0.12)

        print(f"Zernike j={j}: c* = {c_hat*1e9:7.2f} nm  ±  {sigma_c*1e9:5.2f} nm")

    plt.xlabel('$c_j$ [nm RMS]')
    plt.ylabel('Strehl Ratio')
    plt.grid(ls='--', alpha=0.4)
    plt.legend(loc='best')
    plt.tight_layout()

    # se vuoi salvare gli errori insieme ai best coeff:
    sa._best_amps_dense = np.array(best_coeff_dense)
    sa._best_amps_err = np.array(best_coeff_err)
