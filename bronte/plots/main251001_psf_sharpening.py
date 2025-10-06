import numpy as np 
from bronte.ncpa import sharpening_analyzer
import matplotlib.pyplot as plt
from bronte.utils.raw_strehl_ratio_computer import StrehlRatioComputer
from scipy.interpolate import CubicSpline
from matplotlib.lines import Line2D
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

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
    ftag =  "251006_105400.fits"#"251001_104600.fits"
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


def compute_sigma_SR_per_point(SR_vec, AROI_vec, sr_computer, gain_e_per_adu, ron_adu, Nroi):
    """
    SR = Ap / (beta * Aroi), con beta = fitted_dl_max_au / total_dl_flux (come in StrehlRatioComputer).
    Propagazione al primo ordine con cov(Ap, Aroi) = Var(Ap). Tutto in ADU.
    """
    SR = np.asarray(SR_vec, float)
    AROI = np.asarray(AROI_vec, float)
    g = float(gain_e_per_adu)
    ron = float(ron_adu)

    # costante corretta (coerente con get_SR_from_image)
    beta = float(sr_computer._fitted_dl_max_au / sr_computer._total_dl_flux)

    # ricostruisci Ap dai dati salvati (ADU)
    Ap = SR * beta * AROI

    # varianze in ADU^2 (photon + RON)  [NB: ron è già in ADU -> NON dividere per g]
    var_Ap   = Ap / g + ron**2
    var_AROI = AROI / g + Nroi * ron**2

    # derivate e covarianza
    B = beta * AROI
    # evita divisioni per zero
    B[B == 0] = np.finfo(float).eps
    AROI_safe = np.where(AROI == 0, np.finfo(float).eps, AROI)

    dSR_dAp   = 1.0 / B
    dSR_dAroi = - SR / AROI_safe
    cov_Ap_Aroi = var_Ap  # Ap ⊂ AROI

    var_SR = (dSR_dAp**2) * var_Ap + (dSR_dAroi**2) * var_AROI + 2.0 * dSR_dAp * dSR_dAroi * cov_Ap_Aroi
    var_SR = np.clip(var_SR, 0.0, np.inf)
    return np.sqrt(var_SR)


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


def main_low_order():
    sa = load_low_order_sharpening()
    a,b = main(sa)
    plt.xlim(-450,450)
    plt.title('SR vs modal amplitude (lower $Z_j$ modes)')
    return a,b

def main_low_order_finer():
    sa = load_low_order_finer_sharpening()
    a,b = main(sa)
    plt.title('SR vs modal amplitude (lower $Z_j$ modes)')
    return a,b
def main_high_order():
    sa = load_higer_order_sharpening()
    a,b = main(sa)
    plt.title('SR vs modal amplitude (higher $Z_j$ modes)')
    return a,b
    
def main(sa):

    comp_psf = sa._comp_psf
    uncomp_psf = sa._uncomp_psf

    amp_span = sa._amp_span
    measured_sr = sa._measured_sr
    measured_itot_in_roi = sa._mesured_i_roi

    corrected_modes_index = sa._corrected_z_modes_indexes
    interp_sr_func_list = sa._sr_func_list
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

    # --- estetica per tesi ---
    plt.rcParams.update({
        "figure.dpi": 170,
        "axes.labelsize": 14,     # <- dimensione label assi
        "axes.titlesize": 15,     # <- dimensione titolo
        "xtick.labelsize": 12,    # <- dimensione tick X
        "ytick.labelsize": 12,    # <- dimensione tick Y
        "legend.fontsize": 11,    # <- dimensione legenda
    })

    fig, ax = plt.subplots(figsize=(9.5, 6.5))  # <- dimensioni figura

    # palette a colori distinti per ogni modo
    n_modes = len(corrected_modes_index)
    cmap = mpl.cm.get_cmap('tab10' if n_modes <= 10 else 'tab20', n_modes)
    colors = [cmap(i) for i in range(n_modes)]

    for idx_plot, j in enumerate(corrected_modes_index):
        color = colors[idx_plot]

        row = j - 2
        SR_row = measured_sr[row, :]
        AROI_row = measured_itot_in_roi[row, :]

        sigma_SR_row = compute_sigma_SR_per_point(
            SR_row, AROI_row, sa._sr_computer, gain_e_per_adu=gain, ron_adu=ron_adu, Nroi=Nroi
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
        ax.errorbar(
            amp_span/1e-9, SR_row, yerr=sigma_SR_row,
            fmt='o', ms=5, lw=1.2, elinewidth=1.2, capsize=2.5, alpha=0.95,
            color=color, label=f'j={j}', zorder=3
        )

        ax.plot(amps/1e-9, SR_dense, '-', color=color, alpha=1, lw=1.3, zorder=1)

        y_hat = SR_dense[i_max]
        ax.plot([c_hat/1e-9], [y_hat], marker='*', ms=12,
                markerfacecolor=color, markeredgecolor='k', markeredgewidth=1.0,
                linestyle='None', zorder=4)

        print(f"Zernike j={j}: c* = {c_hat*1e9:7.2f} nm  ±  {sigma_c*1e9:5.2f} nm")

    # assi / griglia / limiti
    ax.set_title('SR vs modal amplitude (measurements & interpolations)')
    ax.set_xlabel(r'$c_j$  [nm RMS] wavefront')
    ax.set_ylabel('Strehl Ratio')
    ax.set_ylim(-0.02, 0.92)
    ax.xaxis.set_major_locator(MaxNLocator(8))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which='major', linestyle='--', alpha=0.35)
    ax.grid(which='minor', linestyle=':',  alpha=0.25)

    # legenda per-modo (fuori dal grafico se affollata)
    ncols = 2 if n_modes <= 6 else 3
    leg1 = ax.legend(loc='upper left', ncols=ncols, title='Zernike modes', frameon=True, framealpha=0.95)
    ax.add_artist(leg1)

    # legenda di stile (compatta)
    handle_points = Line2D([0], [0], marker='o', color='k', linestyle='None', ms=6, alpha=0.9)
    handle_line   = Line2D([0], [0], color='k', lw=1.25, alpha=0.6)
    handle_star   = Line2D([0], [0], marker='*', color='k', markerfacecolor='w', markeredgecolor='k', markersize=14, linestyle='None')
    ax.legend(handles=[handle_points, handle_line, handle_star],
              labels=['Measured SR (±1σ)', 'Interpolated SR', r'Best coeff $\hat{c}_j$'],
              loc='lower right', frameon=True, title='Plot styles')

    fig.tight_layout()
    # salva risultati
    sa._best_amps_dense = np.array(best_coeff_dense)
    sa._best_amps_err = np.array(best_coeff_err)
    
    return np.array(best_coeff_dense)/1e-9, np.array(best_coeff_err)/1e-9

def final_results_best_coeff():
    
    # ----------------- dati (in nm) -----------------
    # low-order, dataset 1: ±2.5 um @ 100 nm
    j_lo = np.array([4,5,6,7,8,9,10,11])

    c_hat_lo1, c_err_lo1 = main_low_order()
    # low-order, dataset 2: ±200 nm @ 10 nm

    c_hat_lo2, c_err_lo2 = main_low_order_finer()
    
    # high-order, ±200 nm @ 20 nm
    j_hi = np.arange(12, 32)
    c_hat_hi,c_err_hi = main_high_order()
    plt.close('all')
    
    # ----------------- plotting -----------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    
    # --- pannello (a): low-order, confronto tra due scansioni
    ax = axes[0]
    # per separare leggermente i due set allo stesso j
    dx = 0.15
    ax.errorbar(j_lo - dx, c_hat_lo1, yerr=c_err_lo1, fmt='o', ms=6, capsize=3,
                elinewidth=1.2, label=r'$\pm 2.5\,\mu$m @ 100 nm', color='#1f77b4')
    ax.errorbar(j_lo + dx, c_hat_lo2, yerr=c_err_lo2, fmt='s', ms=6, capsize=3,
                elinewidth=1.2, label=r'$\pm 100$ nm @ 20 nm', color='#ff7f0e')
    
    ax.axhline(0, color='k', lw=1.0, alpha=0.6)
    ax.set_xlabel('Zernike index $j$')
    ax.set_ylabel(r'$\hat{c}_j$ [nm RMS WF]')
    ax.set_title('(a) Low-order modes')
    ax.grid(ls='--', alpha=0.35)
    ax.set_xticks(j_lo)
    ax.legend(frameon=True)
    
    # --- pannello (b): high-order, una sola scansione
    ax = axes[1]
    ax.errorbar(j_hi, c_hat_hi, yerr=c_err_hi, fmt='o', ms=5.5, capsize=3,
                elinewidth=1.2, color='#2ca02c', label=r'$\pm 200$ nm @ 20 nm')
    ax.axhline(0, color='k', lw=1.0, alpha=0.6)
    ax.set_xlabel('Zernike index $j$')
    ax.set_title('(b) High-order modes')
    ax.grid(ls='--', alpha=0.35)
    ax.set_xticks(j_hi[::2])  # etichette ogni 2 per pulizia
    ax.legend(frameon=True)
    
    plt.tight_layout()
    plt.show()


def final_results_psfold():

    sa_lor_coarse = load_low_order_sharpening()
    sa_lor_fine = load_low_order_finer_sharpening()
    sa_hor = load_higer_order_sharpening()

    comp_psf_lorc = sa_lor_coarse._comp_psf
    uncomp_psf_lorc = sa_lor_coarse._uncomp_psf

    Nroi = int(comp_psf_lorc.size)
    gain = 3.5409
    ron_adu = 2.4

    sr_lorc_uncomp = sa_lor_coarse._sr_computer.get_SR_from_image(uncomp_psf_lorc, True)
    sr_lorc_comp = sa_lor_coarse._sr_computer.get_SR_from_image(comp_psf_lorc, True)
    print(f" (low coarse) SR \t BEFORE: {sr_lorc_uncomp} \t AFTER:{sr_lorc_comp}")
    comp_psf_lorf = sa_lor_fine._comp_psf
    uncomp_psf_lorf = sa_lor_fine._uncomp_psf
    sr_lorf_uncomp = sa_lor_fine._sr_computer.get_SR_from_image(uncomp_psf_lorf, True)
    sr_lorf_comp = sa_lor_fine._sr_computer.get_SR_from_image(comp_psf_lorf, True)
    print(f" (low fine) SR \t BEFORE: {sr_lorf_uncomp} \t AFTER:{sr_lorf_comp}")

    comp_psf_hor = sa_hor._comp_psf
    uncomp_psf_hor = sa_hor._uncomp_psf
    sr_hor_uncomp = sa_hor._sr_computer.get_SR_from_image(uncomp_psf_hor, True)
    sr_lhor_comp = sa_hor._sr_computer.get_SR_from_image(comp_psf_hor, True)
    print(f" (high) SR \t BEFORE: {sr_hor_uncomp} \t AFTER:{sr_lhor_comp}")
    
    return sa_lor_fine

# --- ADD: background estimation/sottrazione su anello periferico ---
def _estimate_background_ring(img, inner_frac=0.5, outer_frac=0.95):
    h, w = img.shape
    yc, xc = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - yc)**2 + (xx - xc)**2)
    R = min(yc, xc)
    mask = (r >= inner_frac * R) & (r <= outer_frac * R)
    return float(np.median(img[mask]))

def _subtract_background(img, inner_frac=0.9, outer_frac=0.99):
    bkg = _estimate_background_ring(img, inner_frac, outer_frac)
    out = img.astype(float) - bkg
    out[out < 0] = 0.0   # evita negativi (utile per scala log)
    return out, bkg

def _strehl_and_sigma_from_image(img_roi_adu, sa, gain_e_per_adu=3.5409, ron_adu=2.4):
    """
    SR = Ap / (beta * Aroi), con beta = fitted_dl_max_au / total_dl_flux
    Propagazione al primo ordine con cov(Ap, Aroi) = Var(Ap).
    Tutto in ADU.
    """
    # SR calcolato con la tua routine (normalizzazione a pari flusso)
    SR = sa._sr_computer.get_SR_from_image(img_roi_adu, enable_display=False)

    # Costanti dal calcolatore SR (COERENTI con la definizione interna)
    beta = float(sa._sr_computer._fitted_dl_max_au / sa._sr_computer._total_dl_flux)  # peak/sum "assoluto"
    Aroi = float(img_roi_adu.sum())                                                    # somma ROI [ADU]
    Ap   = float(SR * beta * Aroi)                                                     # picco misurato [ADU]

    g   = float(gain_e_per_adu)
    ron = float(ron_adu)
    Nroi = img_roi_adu.size

    # Varianze (ADU^2)
    var_Ap   = Ap / g + ron**2
    var_Aroi = Aroi / g + Nroi * ron**2

    # Derivate e covarianza
    B = beta * Aroi if Aroi != 0 else np.finfo(float).eps
    dSR_dAp   = 1.0 / B
    dSR_dAroi = - SR / (Aroi if Aroi != 0 else np.finfo(float).eps)
    cov_Ap_Aroi = var_Ap  # Ap è parte di Aroi

    var_SR = (dSR_dAp**2) * var_Ap + (dSR_dAroi**2) * var_Aroi + 2.0 * dSR_dAp * dSR_dAroi * cov_Ap_Aroi
    var_SR = max(var_SR, 0.0)
    sigma_SR = np.sqrt(var_SR)

    return float(SR), float(sigma_SR)


# --- ADD: plot trio DL/Before/After con colorbar condivisa ---
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
def _plot_psf_triple(sa, img_before, img_after, title_prefix="", titles_mod = None):
    """
    - stima e sottrae il background su before/after
    - normalizza la DL al flusso della PSF 'after'
    - calcola SR ± σ per before/after
    - mostra 3 pannelli in log10(ADU) con colorbar unica
    """
    # background subtraction (anello periferico)
    before_bkg_sub, bkg_bef = _subtract_background(img_before)
    after_bkg_sub,  bkg_aft = _subtract_background(img_after)

    # DL rinormalizzata al flusso della 'after' in ADU, costruita dalla DL completa e poi centrata/croppata alla ROI
    dl_full = sa._sr_computer._dl_psf.astype(float)  # PSF DL completa (AU)
    total_dl_flux = float(sa._sr_computer._total_dl_flux)
    
    Aroi_after = float(after_bkg_sub.sum())          # [ADU]
    scale = (Aroi_after / total_dl_flux) if total_dl_flux != 0 else 1.0
    dl_full_scaled = dl_full * scale                 # [ADU]
    
    # center-crop alla dimensione della ROI
    h, w = after_bkg_sub.shape
    H, W = dl_full_scaled.shape
    ys = int(H//2 - h//2); ye = ys + h
    xs = int(W//2 - w//2); xe = xs + w
    dl_norm = dl_full_scaled[ys:ye, xs:xe]          # [ADU], stessa shape della ROI


    # SR ± σ (prima/dopo)
    sr_bef, sig_bef = _strehl_and_sigma_from_image(img_before, sa)
    sr_aft, sig_aft = _strehl_and_sigma_from_image(img_after,  sa)

    # stampa su terminale
    print(f"{title_prefix}  BEFORE: SR = {sr_bef*100:.1f}%  ± {sig_bef*100:.1f}%   (bkg={bkg_bef:.2f} ADU)")
    print(f"{title_prefix}  AFTER : SR = {sr_aft*100:.1f}%  ± {sig_aft*100:.1f}%   (bkg={bkg_aft:.2f} ADU)")

    eps = 1e-7
    imgs = [dl_norm, before_bkg_sub, after_bkg_sub]
    vmax = max(np.max(m) for m in imgs)
    vmin_pos = [np.min(m[m > 0]) if np.any(m > 0) else 1.0 for m in imgs]
    vmin = max(min(vmin_pos), 4e-1)
    log_imgs = [np.log10(m + eps) for m in imgs]
    vmin_log = np.log10(vmin + eps)
    vmax_log = np.log10(vmax + eps)



    # --- layout: niente tight_layout, controlliamo noi margini e colorbar ---
    fig = plt.figure(figsize=(10.5, 3.6), constrained_layout=False)
    # margini calibrati per NON tagliare il suptitle e ridurre lo spacing verticale
    fig.subplots_adjust(left=0.06, right=0.92, bottom=0.08, top=0.86, wspace=0.25)

    gs = GridSpec(1, 3, figure=fig)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    titles = [
        "DL PSF (SR=100%)",
        f"Before, SR={sr_bef*100:.1f}%",# ± {sig_bef*100:.1f}%",
        f"After,  SR={sr_aft*100:.1f}%"# ± {sig_aft*100:.1f}%"
    ]
    
    if titles_mod is not None:
        titles = titles_mod

    im = None
    for ax, L, title in zip(axes, log_imgs, titles):
        im = ax.imshow(L, origin='lower', vmin=vmin_log, vmax=vmax_log, cmap='inferno')
        ax.set_title(title, fontsize=11, pad=6)   # titoli compatti
        ax.set_xticks([]); ax.set_yticks([])

    # --- colorbar della stessa altezza dell'ULTIMO subplot, stessa scala di 'im' ---
    divider = make_axes_locatable(axes[-1])
    cax = divider.append_axes("right", size="4.5%", pad=0.08)  # adiacente al 3° pannello
    cb = fig.colorbar(im, cax=cax)  # eredita automaticamente la normalizzazione di 'im'
    cb.set_label(r'$\log_{10}(\mathrm{ADU})$', fontsize=11)

    # --- suptitle non tagliato, con poco spazio sopra i subplot ---
    if title_prefix:
        fig.suptitle(title_prefix, fontsize=13, y=0.98)

    plt.show()

def final_results_psf():
    # carica i tre dataset
    sa_lor_coarse = load_low_order_sharpening()
    sa_lor_fine   = load_low_order_finer_sharpening()
    sa_hor        = load_higer_order_sharpening()
    
    titles1 = [
        "DL PSF (SR=100%)",
        f"Before, $<SR>_t$=(55.8 ± 0.7)%",
        f"After,  $<SR>_t$=(79.9 ± 0.9)%"]
    
    titles2 = [
        "DL PSF (SR=100%)",
        f"Before, $<SR>_t$=(80.1 ± 0.9)%",
        f"After,  $<SR>_t$=(80.8 ± 0.9)%"]

    #
    titles3 = [
        "DL PSF (SR=100%)",
        f"Before, $<SR>_t$=(80.4 ± 0.9)%",
        f"After,  $<SR>_t$=(81 ± 1)%"]
    
    # LOW ORDER, coarse
    _plot_psf_triple(sa_lor_coarse,
                     sa_lor_coarse._uncomp_psf,
                     sa_lor_coarse._comp_psf,
                     title_prefix="Low-order: $Z_{4}$–$Z_{11}$ (coarse scan)",
                     titles_mod = titles1)

    # LOW ORDER, fine
    _plot_psf_triple(sa_lor_fine,
                     sa_lor_fine._uncomp_psf,
                     sa_lor_fine._comp_psf,
                     title_prefix="Low-order: $Z_{4}$–$Z_{11}$ (fine scan)",
                     titles_mod = titles2)

    # HIGH ORDER
    _plot_psf_triple(sa_hor,
                     sa_hor._uncomp_psf,
                     sa_hor._comp_psf,
                     title_prefix="High-order: $Z_{12}$–$Z_{31}$",
                     titles_mod = titles3)

