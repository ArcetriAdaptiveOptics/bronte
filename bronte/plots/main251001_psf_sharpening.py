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
    return sa




from scipy.interpolate import CubicSpline
from matplotlib.lines import Line2D
import matplotlib as mpl


# ---------- utility: incertezze su SR per ciascun punto ----------
def compute_sigma_SR_per_point(SR_vec, AROI_vec, r_peak_over_sum, gain_e_per_adu, ron_adu, Nroi):
    SR = np.asarray(SR_vec, float)
    AROI = np.asarray(AROI_vec, float)
    g = float(gain_e_per_adu)
    ron = float(ron_adu)

    # picco DL atteso nella ROI e picco misurato
    peakDL = r_peak_over_sum * AROI
    Ap = SR * peakDL

    # varianze (photon + RON) in ADU
    sigma_Aroi = np.sqrt(AROI / g + Nroi * (ron / g) ** 2)
    sigma_peakDL = r_peak_over_sum * sigma_Aroi
    sigma_Ap = np.sqrt(Ap / g + (ron / g) ** 2)

    eps = np.finfo(float).eps
    sigma_SR = SR * np.sqrt((sigma_Ap / (Ap + eps)) ** 2 + (sigma_peakDL / (peakDL + eps)) ** 2)
    return sigma_SR

# ---------- utility: stima incertezza del massimo via curvatura ----------
def sigma_c_from_curvature(amp_span, SR_vec, sigma_SR_vec, c_hat, dense_step):
    amp_span = np.asarray(amp_span, float)
    SR = np.asarray(SR_vec, float)
    sSR = np.asarray(sigma_SR_vec, float)

    cs = CubicSpline(amp_span, SR, bc_type='natural')  # per avere SR'' continuo
    kappa = float(-cs(c_hat, 2))                       # curvatura positiva al massimo

    # rumore effettivo vicino al picco: media delle σ_SR dei 3–5 punti migliori
    K = min(5, len(SR))
    top = np.argsort(-SR)[:K]
    sigma_SR_eff = float(np.mean(sSR[top]))

    # fallback se la curvatura/rumore non sono affidabili
    if not np.isfinite(kappa) or kappa <= 0 or not np.isfinite(sigma_SR_eff) or sigma_SR_eff <= 0:
        return dense_step / np.sqrt(12.0)

    sigma_c_loc = np.sqrt(2.0 * sigma_SR_eff / kappa)
    sigma_grid = dense_step / np.sqrt(12.0)
    return float(np.sqrt(sigma_c_loc**2 + sigma_grid**2))

# -------------------------- MAIN --------------------------
# def main():
#
#     sa = load_low_order_sharpening()
#
#     comp_psf = sa._comp_psf
#     uncomp_psf = sa._uncomp_psf
#
#     amp_span = sa._amp_span                       # (M_points,)
#     measured_sr = sa._measured_sr                 # (N_rows, M_points) include tip/tilt
#     measured_itot_in_roi = sa._mesured_i_roi      # (N_rows, M_points)
#
#     corrected_modes_index = sa._corrected_z_modes_indexes  # es. [4,5,6,7,8,9,10,11]
#     interp_sr_func_list = sa._sr_func_list        # (8, 1001)
#     damp = 5e-9                                   # 5 nm in metri
#     amps = np.arange(amp_span.min(), amp_span.max() + damp, damp)
#
#     # rapporto peak/sum della DL nella ROI (serve per σ_SR)
#     if hasattr(sa, "_au_dl_psf") and sa._au_dl_psf is not None:
#         dl_roi = sa._au_dl_psf
#         r_peak_over_sum = float(dl_roi.max() / dl_roi.sum())
#     else:
#         raise RuntimeError("Manca sa._au_dl_psf per calcolare r = peak/sum della DL nella ROI.")
#
#     # N pixel nella ROI (per la varianza del fondo)
#     Nroi = int(comp_psf.size)
#
#     gain = 3.5409    # e-/ADU
#     ron_adu = 2.4    # ADU
#
#     best_coeff_err = []
#     best_coeff_dense = []
#
#     plt.figure()
#     for idx_plot, j in enumerate(corrected_modes_index):
#
#         # j (Noll) -> riga: salta Z2,Z3 (tip/tilt)
#         row = j - 2
#
#         SR_row = measured_sr[row, :]
#         AROI_row = measured_itot_in_roi[row, :]
#
#         # σ_SR per punto (photon + RON), usata sia per le barre d'errore sia per σ_c
#         sigma_SR_row = compute_sigma_SR_per_point(
#             SR_row, AROI_row, r_peak_over_sum, gain_e_per_adu=gain, ron_adu=ron_adu, Nroi=Nroi
#         )
#
#         # massimo dalla curva interpolata densa già fornita
#         SR_dense = interp_sr_func_list[idx_plot]   # lunghezza = len(amps)
#         i_max = int(np.argmax(SR_dense))
#         c_hat = float(amps[i_max])
#         best_coeff_dense.append(c_hat)
#
#         # incertezza sul massimo via curvatura
#         sigma_c = sigma_c_from_curvature(amp_span, SR_row, sigma_SR_row, c_hat=c_hat, dense_step=damp)
#         best_coeff_err.append(sigma_c)
#
#         # --- PLOT con barre d'errore sui punti misurati
#         # punti con errorbar
#         eb = plt.errorbar(
#             amp_span/1e-9, SR_row, yerr=sigma_SR_row,
#             fmt='o', ms=4, lw=1, elinewidth=1, capsize=2, alpha=0.9,
#             label=f'j={j}'
#         )
#         color = eb[0].get_color()
#
#         # curva interpolata densa
#         plt.plot(amps/1e-9, SR_dense, '-', color=color, alpha=0.9)
#
#         # massimo + banda ±1σ_c
#         plt.axvline(c_hat/1e-9, color=color, ls='--', alpha=0.6)
#         ymax = 1.05 * np.nanmax(SR_dense)
#         plt.fill_betweenx(
#             [0, ymax],
#             (c_hat - sigma_c)/1e-9, (c_hat + sigma_c)/1e-9,
#             color=color, alpha=0.12
#         )
#
#         print(f"Zernike j={j}: c* = {c_hat*1e9:7.2f} nm  ±  {sigma_c*1e9:5.2f} nm")
#
#     plt.xlabel('$c_j$ [nm RMS]')
#     plt.ylabel('Strehl Ratio')
#     plt.grid(ls='--', alpha=0.4)
#     plt.legend(loc='best', ncols=2)
#     plt.tight_layout()
#
#     # salva per uso successivo (opzionale)
#     sa._best_amps_dense = np.array(best_coeff_dense)
#     sa._best_amps_err = np.array(best_coeff_err)

def main():

    sa = load_low_order_finer_sharpening()

    comp_psf = sa._comp_psf
    uncomp_psf = sa._uncomp_psf

    amp_span = sa._amp_span
    measured_sr = sa._measured_sr
    measured_itot_in_roi = sa._mesured_i_roi

    corrected_modes_index = sa._corrected_z_modes_indexes   # es. [4,5,6,7,8,9,10,11]
    interp_sr_func_list = sa._sr_func_list                  # (8, 1001)
    damp = 5e-9
    amps = np.arange(amp_span.min(), amp_span.max() + damp, damp)

    # rapporto peak/sum DL nella ROI
    if hasattr(sa, "_au_dl_psf") and sa._au_dl_psf is not None:
        dl_roi = sa._au_dl_psf
        r_peak_over_sum = float(dl_roi.max() / dl_roi.sum())
    else:
        raise RuntimeError("Manca sa._au_dl_psf per calcolare r = peak/sum della DL nella ROI.")

    Nroi = int(comp_psf.size)
    gain = 3.5409
    ron_adu = 2.4

    best_coeff_err = []
    best_coeff_dense = []

    plt.figure(figsize=(9.5, 6.5))

    # --- palette a colori distinti per ogni modo
    n_modes = len(corrected_modes_index)
    if n_modes <= 10:
        cmap = mpl.cm.get_cmap('tab10', n_modes)
    else:
        cmap = mpl.cm.get_cmap('tab20', n_modes)
    colors = [cmap(i) for i in range(n_modes)]

    for idx_plot, j in enumerate(corrected_modes_index):
        color = colors[idx_plot]

        # j (Noll) -> riga (salta Z2,Z3)
        row = j - 2

        SR_row = measured_sr[row, :]
        AROI_row = measured_itot_in_roi[row, :]

        sigma_SR_row = compute_sigma_SR_per_point(
            SR_row, AROI_row, r_peak_over_sum, gain_e_per_adu=gain, ron_adu=ron_adu, Nroi=Nroi
        )

        # massimo dalla curva interpolata densa
        SR_dense = interp_sr_func_list[idx_plot]
        i_max = int(np.argmax(SR_dense))
        c_hat = float(amps[i_max])
        best_coeff_dense.append(c_hat)

        # incertezza sul massimo via curvatura
        sigma_c = sigma_c_from_curvature(amp_span, SR_row, sigma_SR_row, c_hat=c_hat, dense_step=damp)
        best_coeff_err.append(sigma_c)

        # --- PLOT ---
        # 1) punti misurati + barre d'errore (colore dedicato)
        plt.errorbar(
            amp_span/1e-9, SR_row, yerr=sigma_SR_row,
            fmt='o', ms=5, lw=1.2, elinewidth=1.2, capsize=2.5, alpha=0.95,
            color=color, label=f'j={j}', zorder=3
        )

        # 2) curva interpolata densa (più discreta)
        plt.plot(amps/1e-9, SR_dense, '-', color=color, alpha=1, lw=1.0, zorder=1)

        # 3) massimo stimato: stella grande con bordo nero, stesso colore di serie
        y_hat = SR_dense[i_max]
        plt.plot([c_hat/1e-9], [y_hat], marker='*', ms=10,
                 markerfacecolor=color, markeredgecolor='k', markeredgewidth=1.0,
                 linestyle='None', zorder=4)
        # --- horizontal error bar on best coeff (±1σ_c) ---
        hb = plt.errorbar(
            x=c_hat/1e-9, y=y_hat,
            xerr=sigma_c/1e-9, yerr=None,
            fmt='none', ecolor=color, elinewidth=2.0, capsize=3, alpha=0.9, zorder=6
        )

        # 4) banda ±1σ_c (tenue)
        ymax = 1.05 * np.nanmax(SR_dense)
        plt.fill_betweenx(
            [0, ymax],
            (c_hat - sigma_c)/1e-9, (c_hat + sigma_c)/1e-9,
            color=color, alpha=0.40, zorder=0
        )

        print(f"Zernike j={j}: c* = {c_hat*1e9:7.2f} nm  ±  {sigma_c*1e9:5.2f} nm")

    # assi / griglia / legende
    plt.xlabel(r'$c_j$ [nm RMS]')
    plt.ylabel('Strehl Ratio')
    plt.grid(ls='--', alpha=0.35)
    plt.tight_layout()

    # legenda per-modo (punti colorati)
    leg1 = plt.legend(loc='upper left', ncols=2, title='Zernike modes', frameon=True)
    plt.gca().add_artist(leg1)

    # legenda di stile
    handle_points = Line2D([0], [0], marker='o', color='k', linestyle='None', ms=6, alpha=0.9, label='Measured SR (±1σ)')
    handle_line   = Line2D([0], [0], color='k', lw=1.0, alpha=0.35, label='Interpolated SR')
    handle_star   = Line2D([0], [0], marker='*', color='k', markerfacecolor='w', markeredgecolor='k',
                           markersize=14, linestyle='None', label=r'Best coeff $\hat{c}_j$')
    handle_hxerr = Line2D([0], [0], color='k', lw=2.0, alpha=0.9)
    plt.legend(handles=[handle_points, handle_line, handle_star, handle_hxerr],
               labels=['Measured SR (±1σ)', 'Interpolated SR', r'Best coeff $\hat{c}_j$', r'$\hat{c}_j$ horizontal error bar'],
               loc='lower right', frameon=True, title='Plot styles')

    # salva risultati
    sa._best_amps_dense = np.array(best_coeff_dense)
    sa._best_amps_err = np.array(best_coeff_err)