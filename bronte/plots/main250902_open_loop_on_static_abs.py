
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from bronte.scao.telemetry.mains.main250902_analysing_bench_static_abs import load_ol_rec_modes
from arte.types.mask import CircularMask
from bronte.wfs.slm_rasterizer import SlmRasterizer
from bronte.wfs.kl_slm_rasterizer import KLSlmRasterizer
from bronte.startup import set_data_dir

# ---------- stile generale (senza grid globale) ----------
def _setup_matplotlib_for_thesis():
    mpl.rcParams.update({
        "figure.dpi": 170,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "lines.linewidth": 1.6,
        "errorbar.capsize": 2.5,
    })

# ---------- helper estetico (aggiungo anche logx per gestione tick su semilog-x) ----------
def _beautify(ax, xlabel=None, ylabel=None, title=None, xmaj=7, ymaj=6, logy=False, logx=False):
    if title:  ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)

    # Locator X: se logx True, lasciamo i locator di default (log), altrimenti MaxNLocator
    if not logx:
        ax.xaxis.set_major_locator(MaxNLocator(xmaj))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))

    # Locator Y: se logy True, lasciamo i locator di default (log)
    if not logy:
        ax.yaxis.set_major_locator(MaxNLocator(ymaj))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax.grid(which="minor", linestyle=":", alpha=0.25)
    ax.tick_params(direction="out", length=4, width=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------- plotting base (PLot 1) — aggiungo ME in legenda e mode index 2..201 ----------
def _plot_means_per_dataset(ol_ftag_list, means_list, basis_name, save_prefix, meas_err_list=None):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    n_modes = len(means_list[0])
    x = np.arange(2, n_modes + 2)  # 2..201

    for i, (tag, m) in enumerate(zip(ol_ftag_list, means_list)):
        if meas_err_list is not None:
            lbl = f"{_fmt_tag(tag)}  (ME {meas_err_list[i]:.0f} nm)"
        else:
            lbl = _fmt_tag(tag)
        ax.plot(x, m, marker=".", markersize=2.5, label=lbl, alpha=0.95)

    ax.axhline(0, color="k", linewidth=0.8, alpha=0.6)
    _beautify(ax,
              xlabel="Mode index (starts at 2)",
              ylabel="Reconstructed mode mean [nm]",
              title=f"Static aberrations (means) — {basis_name} base ({n_modes} modes)")
    ax.legend(ncols=2, frameon=True)
    fig.tight_layout()


# ---------- std per dataset (mode index 2..201) ----------
def _plot_stds_per_dataset(ol_ftag_list, stds_list, basis_name, save_prefix, meas_err_list=None):
    """
    Plot della STD temporale per dataset con label che include (se fornito)
    il measurement error del dataset: '... (ME XX nm)'.
    """
    fig, ax = plt.subplots(figsize=(9, 5.2))
    n_modes = len(stds_list[0])
    x = np.arange(2, n_modes + 2)  # 2..201

    eps = 1e-12  # evita log(0)
    for i, (tag, s) in enumerate(zip(ol_ftag_list, stds_list)):
        y = np.maximum(s, eps)
        if meas_err_list is not None and i < len(meas_err_list):
            lbl = r" $\sigma_{s2c}$"+f"={meas_err_list[i]:.0f} nm"
        else:
            lbl = _fmt_tag(tag)
        ax.semilogy(x, y, marker=".", markersize=2.5, label=lbl, alpha=0.95)

    _beautify(ax,
              xlabel="Mode index",
              ylabel=r"$\sigma_{STD}$"+" [nm] RMS WF",
              title=f"STD of reconstructed modes per loop — {basis_name} base",
              logy=True)
    ax.legend(ncols=2, frameon=True)
    fig.tight_layout()


# ---------- summary across-datasets (mode index 2..201) ----------
def _plot_across_datasets_mean_of_means_errorbars(means_list, basis_name, save_prefix):
    y, yerr = _compute_across_stats(means_list)
    n_modes = y.size
    x = np.arange(2, n_modes + 2)  # 2..201

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.errorbar(x, y, yerr=yerr, fmt="-", marker=".", markersize=2.2,
                linewidth=1.2, alpha=0.95)

    ax.axhline(0, color="k", linewidth=0.8, alpha=0.6)
    _beautify(ax,
              xlabel="Mode index (starts at 2)",
              ylabel="Across-dataset mean of reconstructed mode [nm]\n(error bars: across-dataset std)",
              title=f"Across-dataset summary — {basis_name} base ({n_modes} modes)")
    fig.tight_layout()


# ---------- nuove figure di overlay SPLIT (sinistra primi 3 modi, destra tutti i restanti) ----------
def _plot_overlay_means_plus_summary_split(ol_ftag_list, means_list, wfe_list, basis_name, save_prefix):
    """
    Due subplot in una sola figura:
      - SX: soli primi 3 modi (indici 2,3,4) -> elementi 0..2 dell'array
      - DX: restanti (5..201) -> elementi 3..end
    Legende dataset con WFE.
    """
    # Statistiche across-datasets
    y_all, yerr_all = _compute_across_stats(means_list)

    n_modes = y_all.size
    x_all = np.arange(2, n_modes + 2)               # 2..201
    x_left = x_all[:3]                               # 2,3,4
    x_right = x_all[3:]                              # 5..201

    # Slices per i due pannelli
    def _split(arr):
        return arr[:3], arr[3:]

    y_L, y_R       = _split(y_all)
    yerr_L, yerr_R = _split(yerr_all)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.6))

    # --- PANNELLO SINISTRO: primi 3 modi
    for tag, m, wfe in zip(ol_ftag_list, means_list, wfe_list):
        axL.plot(x_left, m[:3], marker=".", markersize=3.0, linewidth=0.0,
                 alpha=0.7, label=f"WFE={wfe:.0f}nm")
    axL.fill_between(x_left, y_L - 3*yerr_L, y_L + 3*yerr_L, alpha=0.25, label="<WFE> ±3σ")
    axL.plot(x_left, y_L, color='m', ls='-',linewidth=0.8, label="<WFE>")
    axL.axhline(0, color="k", linewidth=0.8, alpha=0.6)
    _beautify(axL,
              xlabel="Mode index",
              ylabel="Reconstructed mode " +r"$\Delta c$"+" [nm] RMS WF",
              title=f"{basis_name} first 3 modes")

    # --- PANNELLO DESTRO: restanti modi
    for tag, m, wfe in zip(ol_ftag_list, means_list, wfe_list):
        axR.plot(x_right, m[3:], marker=".", markersize=2.0, linewidth=0.0,
                 alpha=0.5, label=f"WFE={wfe:.0f} nm")
    axR.fill_between(x_right, y_R - 3*yerr_R, y_R + 3*yerr_R, alpha=0.25, label="<WFE> ±3σ")
    axR.plot(x_right, y_R, color='m', ls='-',linewidth=0.8, label="<WFE>")
    axR.axhline(0, color="k", linewidth=0.8, alpha=0.6)
    _beautify(axR,
              xlabel="Mode index",
              title=f"{basis_name} remaining modes")

    # Legenda unica (metto sul destro per non sovraccaricare il sinistro)
    axR.legend(ncols=2, frameon=True)
    fig.tight_layout()
    # (Se vuoi salvare, riattiva i savefig qui)
    # fig.savefig(f"{save_prefix}_{basis_name.lower()}_overlay_split.png")
    # fig.savefig(f"{save_prefix}_{basis_name.lower()}_overlay_split.pdf")


# ---------- 1) Cumulative RMS — ora in semi log-x e con mode index 2..201 ----------
def _plot_cumulative_rms(means_list, basis_name, save_prefix, ol_ftag_list=None, wfe_list=None):
    """
    Per ogni dataset i: cum_rms_i[k] = sqrt( sum_{m<=k} mean_i[m]^2 ).
    x in scala logaritmica (log-x). Mode index: 2..201.
    Nelle label, se fornito, mostra il WFE del dataset.
    """
    stack = np.vstack(means_list)             # (n_datasets, n_modes)
    cum_sq = np.cumsum(stack**2, axis=1)      # (n_datasets, n_modes)
    cum_rms = np.sqrt(cum_sq)                 # same shape

    y = cum_rms.mean(axis=0)
    yerr = cum_rms.std(axis=0, ddof=1)

    n_modes = stack.shape[1]
    x = np.arange(2, n_modes + 2)  # 2..201

    fig, ax = plt.subplots(figsize=(9.5, 5.4))

    # Curve dei singoli dataset (con WFE in legenda se disponibile)
    if ol_ftag_list is not None:
        for i, (tag, c) in enumerate(zip(ol_ftag_list, cum_rms)):
            if wfe_list is not None and i < len(wfe_list):
                lbl = f"WFE {wfe_list[i]:.0f} nm"
            else:
                lbl = _fmt_tag(tag)
            ax.plot(x, c, alpha=0.55, linewidth=1.2, label=lbl)
    else:
        for c in cum_rms:
            ax.plot(x, c, alpha=0.55, linewidth=1.2)

    # Fascia summary
    ax.fill_between(x, y - yerr, y + yerr, alpha=0.25, label="Mean ±1σ")
    ax.plot(x, y, color='m', linewidth=1.5, label="Mean")

    ax.set_xscale('log')  # semilog-x
    _beautify(ax,
              xlabel="Mode index",
              ylabel="Cumulative RMS [nm] WF",
              title=f"Cumulative RMS of reconstructed modes — {basis_name} base ({n_modes} modes)",
              logx=True)
    if ol_ftag_list is not None:
        ax.legend(ncols=2, frameon=True)
    fig.tight_layout()




def _fmt_tag(tag):
    # "250808_151100" -> "2025-08-08 15:11"
    if "_" in tag and len(tag) >= 13:
        d, t = tag.split("_")
        d_fmt = f"20{d[:2]}-{d[2:4]}-{d[4:6]}"
        t_fmt = f"{t[:2]}:{t[2:4]}"
        return f"{d_fmt} {t_fmt}"
    return tag


# ---------- loading ----------
def get_data_lists(ol_ftag_list, rec_tag):
    mean_rec_modes_in_nm_list = []
    std_rec_modes_in_nm_list  = []
    meas_error_list           = []
    tot_wfe_list              = []

    for ol_ftag in ol_ftag_list:
        mean_rec_modes_in_nm, std_rec_modes_in_nm, _ = load_ol_rec_modes(ol_ftag, rec_tag)
        meas_error_in_nm = np.sqrt(np.sum(std_rec_modes_in_nm**2))
        tot_wfe          = np.sqrt(np.sum(mean_rec_modes_in_nm**2))

        mean_rec_modes_in_nm_list.append(mean_rec_modes_in_nm)
        std_rec_modes_in_nm_list.append(std_rec_modes_in_nm)
        meas_error_list.append(meas_error_in_nm)
        tot_wfe_list.append(tot_wfe)

    return mean_rec_modes_in_nm_list, std_rec_modes_in_nm_list, meas_error_list, tot_wfe_list


# ---------- utility statistica ----------
def _compute_across_stats(means_list):
    """
    Ritorna (y, yerr) dove:
      y[k]    = media sui dataset dei mean_per_dataset[k]
      yerr[k] = std  sui dataset dei mean_per_dataset[k] (ddof=1)
    """
    stack = np.vstack(means_list)  # (n_datasets, n_modes)
    y    = stack.mean(axis=0)
    yerr = stack.std(axis=0, ddof=1)
    return y, yerr





# ---------- nuove figure di overlay ----------
def _plot_overlay_means_plus_summary(ol_ftag_list, means_list, basis_name, save_prefix):
    """
    Sovrappone:
      - curve dei singoli dataset (fig. 1 o 4)
      - media tra dataset con fascia ±std (fig. 3 o 6)
    """
    y, yerr = _compute_across_stats(means_list)
    n_modes = y.size
    x = np.arange(1, n_modes + 1)

    fig, ax = plt.subplots(figsize=(10, 5.6))

    # Curve dei dataset
    for tag, m in zip(ol_ftag_list, means_list):
        ax.plot(x, m, marker=".", markersize=2.2, linewidth=0, alpha=0.6, label=_fmt_tag(tag))

    # Media + fascia ±std
    ax.fill_between(x, y - 3*yerr, y + 3*yerr, alpha=0.25, label="Across-dataset ±3σ")
    ax.plot(x, y, linewidth=1,color='m', label="Across-dataset mean")

    ax.axhline(0, color="k", linewidth=0.8, alpha=0.6)
    _beautify(ax,
              xlabel="Mode index",
              ylabel="Reconstructed mode [nm]",
              title=f"Overlay: datasets vs across-dataset summary — {basis_name} basis ({n_modes} modes)")
    ax.legend(ncols=2, frameon=True)
    fig.tight_layout()

def _plot_normalized_cumulative_wfe(means_list, basis_name, save_prefix,
                                    ol_ftag_list=None, wfe_list=None, target_frac=0.9989):
    """
    Cumulativa normalizzata del WFE.
    - Ogni curva di dataset è normalizzata al proprio WFE totale (== cum_rms[-1]).
    - Niente banda d'errore.
    - Le label dei dataset includono il WFE del dataset.
    - La curva "Across-dataset mean" include il WFE del vettore medio (non normalizzato).
    - x in semilog, indici 2..201.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    stack   = np.vstack(means_list)           # (n_datasets, n_modes)
    cum_sq  = np.cumsum(stack**2, axis=1)
    cum_rms = np.sqrt(cum_sq)                 # (n_datasets, n_modes)

    final_rms = cum_rms[:, -1].copy()
    final_rms[final_rms == 0] = 1.0           # evita divisione per zero
    cum_norm = (cum_rms.T / final_rms).T      # normalizzato in [0,1]

    # WFE per label: se non passato, calcolalo dai mean per dataset (uguale a final_rms)
    if (wfe_list is None) or (len(wfe_list) != stack.shape[0]):
        wfe_list = final_rms

    # Curva media normalizzata (media delle cumulanti normalizzate)
    y_mean = cum_norm.mean(axis=0)

    # WFE del vettore medio across-dataset (non normalizzato)
    across_mean_vec = stack.mean(axis=0)
    across_mean_wfe = float(np.sqrt(np.sum(across_mean_vec**2)))

    n_modes = cum_norm.shape[1]
    x = np.arange(2, n_modes + 2)             # 2..201

    # Trova il primo indice (curva media) che supera la soglia target_frac
    idx_thr = np.argmax(y_mean >= target_frac) if np.any(y_mean >= target_frac) else None
    mode_at_thr = (idx_thr + 2) if idx_thr is not None else None

    fig, ax = plt.subplots(figsize=(9.8, 5.4))

    # Curve per dataset (label con WFE)
    if ol_ftag_list is not None:
        for tag, c, wfe in zip(ol_ftag_list, cum_norm, wfe_list):
            ax.plot(x, c, alpha=0.55, linewidth=1.0,
                    label=f"WFE={wfe:.0f} nm")
    else:
        for c, wfe in zip(cum_norm, wfe_list):
            ax.plot(x, c, alpha=0.55, linewidth=1.0,
                    label=f"WFE={wfe:.0f} nm")

    # Curva media con WFE del vettore medio
    ax.plot(x, y_mean, linewidth=1.2,color='m',
            label=f"<WFE>={across_mean_wfe:.0f} nm")

    # Guide soglia (orizzontale e verticale)
    ax.axhline(target_frac, color="k", linestyle="--", linewidth=1.0, alpha=0.6)
    if mode_at_thr is not None:
        ax.axvline(mode_at_thr, color="k", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.text(mode_at_thr, target_frac,
                f"{target_frac:.2%} at mode {mode_at_thr}",
                rotation=0, va="top", ha="left", fontsize=10)

    ax.set_xscale('log')  # semilog-x
    _beautify(ax,
              xlabel="Mode index",
              ylabel="Normalized cumulative WFE",
              title=f"Normalized cumulative WFE — {basis_name} base ({n_modes} modes)",
              logx=True)

    if ol_ftag_list is not None:
        ax.legend(ncols=2, frameon=True)

    fig.tight_layout()

    # stampa un promemoria sul terminale
    if mode_at_thr is not None:
        print(f"[{basis_name}] {int(target_frac*100)}% of WFE reached at mode index ≈ {mode_at_thr} "
              f"(across-mean WFE ≈ {across_mean_wfe:.0f} nm)")
    else:
        print(f"[{basis_name}] {int(target_frac*100)}% threshold not reached within available modes "
              f"(across-mean WFE ≈ {across_mean_wfe:.0f} nm)")

def _plot_cumulative_pair(means_list, basis_name, save_prefix,
                          ol_ftag_list=None, wfe_list=None, target_frac=0.90):
    """
    Figura a 2 subplot:
      - SX: Cumulative RMS (nm) stile _plot_cumulative_rms (banda ±1σ + mean, legenda)
      - DX: Cumulative RMS normalizzata (come _plot_normalized_cumulative_wfe), senza banda,
            con soglia orizzontale 'target_frac' e riga verticale al primo superamento (nessuna legenda).
    Colori/label coerenti tra i pannelli; label dei dataset includono il WFE.
    """
    # --- preparazione dati cumulativa (nm) ---
    stack   = np.vstack(means_list)            # (n_datasets, n_modes)
    cum_sq  = np.cumsum(stack**2, axis=1)
    cum_rms = np.sqrt(cum_sq)                  # (n_datasets, n_modes)
    y_nm    = cum_rms.mean(axis=0)
    yerr_nm = cum_rms.std(axis=0, ddof=1)

    # --- normalizzata ---
    final_rms = cum_rms[:, -1].copy()
    final_rms[final_rms == 0] = 1.0
    cum_norm = (cum_rms.T / final_rms).T       # [0,1]
    y_norm   = cum_norm.mean(axis=0)

    # WFE per label (se non fornito): usa final_rms
    if (wfe_list is None) or (len(wfe_list) != stack.shape[0]):
        wfe_list = final_rms

    # WFE del vettore medio across-dataset (per label della curva media normalizzata)
    across_mean_vec  = stack.mean(axis=0)
    across_mean_wfe  = float(np.sqrt(np.sum(across_mean_vec**2)))

    # asse x (2..201) e soglia per normalizzata
    n_modes = stack.shape[1]
    x = np.arange(2, n_modes + 2)
    idx_thr = np.argmax(y_norm >= target_frac) if np.any(y_norm >= target_frac) else None
    mode_at_thr = (idx_thr + 2) if idx_thr is not None else None

    # palette coerente tra i due pannelli
    base_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    def color_for(i):
        return base_colors[i % len(base_colors)] if base_colors else None

    # --- figura e assi ---
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.6, 5.6), sharex=False)

    # ====== SINISTRA: cumulativa (nm) ======
    if ol_ftag_list is not None:
        for i, (tag, c, wfe) in enumerate(zip(ol_ftag_list, cum_rms, wfe_list)):
            axL.plot(x, c, alpha=0.55, linewidth=1.2,
                     color=color_for(i),
                     label=f"WFE={wfe:.0f} nm")
    else:
        for i, (c, wfe) in enumerate(zip(cum_rms, wfe_list)):
            axL.plot(x, c, alpha=0.55, linewidth=1.2,
                     color=color_for(i),
                     label=f"WFE={wfe:.0f} nm")

    # banda mean ±1σ e curva media (magenta)
    axL.fill_between(x, y_nm - yerr_nm, y_nm + yerr_nm, alpha=0.25, label="<WFE> ±1σ")
    axL.plot(x, y_nm, color='m', linewidth=1.5, label=f"<WFE>={across_mean_wfe:.0f} nm")

    axL.set_xscale('log')
    _beautify(axL,
              xlabel="Mode index ",
              ylabel="Cumulative WFE [nm] RMS WF",
              title= "",
              logx=True)
    # legenda SOLO sul pannello sinistro
    axL.legend(ncols=2, frameon=True)

    # ====== DESTRA: cumulativa normalizzata ======
    if ol_ftag_list is not None:
        for i, (tag, c, wfe) in enumerate(zip(ol_ftag_list, cum_norm, wfe_list)):
            axR.plot(x, c, alpha=0.55, linewidth=1.0, color=color_for(i))
    else:
        for i, (c, wfe) in enumerate(zip(cum_norm, wfe_list)):
            axR.plot(x, c, alpha=0.55, linewidth=1.0, color=color_for(i))

    # curva media (magenta) + guide soglia
    axR.plot(x, y_norm, linewidth=1.5, color='m')
    axR.axhline(target_frac, color="k", linestyle="--", linewidth=1.0, alpha=0.6)
    if mode_at_thr is not None:
        axR.axvline(mode_at_thr, color="k", linestyle="--", linewidth=1.0, alpha=0.6)
        axR.text(mode_at_thr, target_frac,
                 f"{target_frac:.2%} at mode {mode_at_thr}",
                 rotation=90, va="top", ha="left", fontsize=10)

    axR.set_xscale('log')
    _beautify(axR,
              xlabel="Mode index ",
              ylabel="Normalized Cumulative WFE",
              title="",
              logx=True)

    # suptitle della figura
    fig.suptitle(f"Cumulative WFE - {basis_name} base", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    # messaggio di servizio
    if mode_at_thr is not None:
        print(f"[{basis_name}] {int(target_frac*100)}% of WFE (normalized) reached at mode ≈ {mode_at_thr} "
              f"| across-mean WFE ≈ {across_mean_wfe:.0f} nm")
    else:
        print(f"[{basis_name}] {int(target_frac*100)}% threshold not reached within available modes "
              f"| across-mean WFE ≈ {across_mean_wfe:.0f} nm")


    
from matplotlib.colors import TwoSlopeNorm


# ---------- funzione principale ----------
def display_results(save_prefix="bench_statics"):

    _setup_matplotlib_for_thesis()

    rec_ftag_kl   = '250808_144900'
    rec_ftag_zern = '250616_103300'
    ol_ftag_list  = ['250808_151100','250808_161900','250828_133300',
                     '250829_111600','250902_101600']

    # --- KL ---
    kl_means, kl_stds, kl_meas_errs, kl_tot_wfe = get_data_lists(ol_ftag_list, rec_ftag_kl)
    # --- Zernike ---
    z_means,  z_stds,  z_meas_errs,  z_tot_wfe  = get_data_lists(ol_ftag_list, rec_ftag_zern)

    # Log numerico
    print("\n=== KL basis ===")
    for tag, wfe, me in zip(ol_ftag_list, kl_tot_wfe, kl_meas_errs):
        print(f"[{_fmt_tag(tag)} | rec {rec_ftag_kl}]  Total WFE: {wfe:.0f} nm rms   |   Measurement error: {me:.0f} nm rms")

    print("\n=== Zernike basis ===")
    for tag, wfe, me in zip(ol_ftag_list, z_tot_wfe, z_meas_errs):
        print(f"[{_fmt_tag(tag)} | rec {rec_ftag_zern}]  Total WFE: {wfe:.0f} nm rms   |   Measurement error: {me:.0f} nm rms")

    

    # Figure per KL
    _plot_stds_per_dataset(ol_ftag_list, kl_stds, "KL", save_prefix, meas_err_list=kl_meas_errs)

    #_plot_means_per_dataset(ol_ftag_list, kl_means, "KL", save_prefix, meas_err_list=kl_meas_errs)  # legenda con ME
    _plot_overlay_means_plus_summary_split(ol_ftag_list, kl_means, kl_tot_wfe, "KL", save_prefix)    # overlay split con WFE
    
    # Figure per Zernike
    _plot_stds_per_dataset(ol_ftag_list, z_stds, "Zernike", save_prefix, meas_err_list=z_meas_errs)
    #_plot_means_per_dataset(ol_ftag_list, z_means, "Zernike", save_prefix, meas_err_list=z_meas_errs)  # legenda con ME
    _plot_overlay_means_plus_summary_split(ol_ftag_list, z_means, z_tot_wfe, "Zernike", save_prefix)    # overlay split con WFE
    
    # Cumulative RMS (log-x)
    # _plot_cumulative_rms(kl_means, "KL", save_prefix, ol_ftag_list, wfe_list=kl_tot_wfe)
    # _plot_cumulative_rms(z_means, "Zernike", save_prefix, ol_ftag_list, wfe_list=z_tot_wfe)
    # _plot_normalized_cumulative_wfe(kl_means, "KL", save_prefix, ol_ftag_list)
    # _plot_normalized_cumulative_wfe(z_means, "Zernike", save_prefix, ol_ftag_list)
    _plot_cumulative_pair(kl_means, "KL", save_prefix, ol_ftag_list, wfe_list=kl_tot_wfe, target_frac=0.9989)
    _plot_cumulative_pair(z_means, "Zernike", save_prefix, ol_ftag_list, wfe_list=z_tot_wfe, target_frac=0.9989)


    
    plt.show()
    # ---------- PRINT SUMMARY: KL, ZERNIKE, COMBINED ----------
    def _print_wfe_summary(label, wfe_list, meas_err_list):
        wfe = np.asarray(wfe_list, dtype=float)
        me  = np.asarray(meas_err_list, dtype=float)
        mean_nm = wfe.mean()
        std_nm  = wfe.std(ddof=1) if wfe.size > 1 else 0.0     # inter-session std
        me_mean_nm = me.mean()
    
        mean_um = mean_nm / 1000.0
        std_um  = std_nm  / 1000.0
        me_mean_um = me_mean_nm / 1000.0
    
        print(f"\n[{label}]  WFE medio = {mean_nm:.0f} ± {std_nm:.0f} nm RMS  "
              f"({mean_um:.3f} ± {std_um:.3f} µm)  |  Measurement error medio = {me_mean_nm:.1f} nm ({me_mean_um:.3f} µm)")
    
    _print_wfe_summary("KL",       kl_tot_wfe, kl_meas_errs)
    _print_wfe_summary("Zernike",  z_tot_wfe,  z_meas_errs)
    
    def _print_first3_modes_summary(means_list, label=""):
        """
        Stampa per i primi 3 modi (indici visuali 2,3,4) il valore medio tra dataset
        delle medie ricostruite e la deviazione standard tra dataset (errore).
        
        Parametri
        ---------
        means_list : list[np.ndarray]
            Lista di vettori 1D (uno per dataset) con le medie dei modi in nm.
        label : str
            Etichetta (es. "KL", "Zernike") mostrata nel print.
        """
        stack = np.vstack(means_list)  # shape: (n_datasets, n_modes)
        n_modes = stack.shape[1]
        if n_modes < 3:
            print(f"[{label}] Servono almeno 3 modi (trovati {n_modes}).")
            return
    
        for k in range(3):  # 0,1,2 -> mode index visuale 2,3,4
            vals = stack[:, k]
            mean_nm = vals.mean()
            std_nm  = vals.std(ddof=1) if vals.size > 1 else 0.0
            print(f"[{label}] Mode {k+2}: {mean_nm:.1f} ± {std_nm:.1f} nm (mean ± std across datasets)")
    _print_first3_modes_summary(kl_means, "KL")
    _print_first3_modes_summary(z_means, "Zernike")
    
    
    # kl_coeff_stat = _compute_across_stats(kl_means)
    # zc_coeff_stat = _compute_across_stats(z_means)
    #
    # zc2rast = zc_coeff_stat[0]*1e-9
    # klc2rast = kl_coeff_stat[0]*1e-9
    # cmask = _get_cmask()
    #
    # ol_wf_zc = _compute_ol_wf_zern(zc2rast, cmask) 
    # ol_wf_kl = _compute_ol_wf_kl(klc2rast, cmask)
    # return ol_wf_zc, ol_wf_kl

def _get_cmask():
    
    SLM_PUPIL_CENTER = (579, 968)#YX in pixel
    SLM_PUPIL_RADIUS = 545
    FRAME_SHAPE = (1152, 1920)
    cmask = CircularMask(
        frameShape = FRAME_SHAPE,
        maskRadius = SLM_PUPIL_RADIUS,
        maskCenter = SLM_PUPIL_CENTER)
    return cmask

def _compute_ol_wf_zern(sr, zc_coeff, cmask):
    
    wfz = sr.zernike_coefficients_to_raster(zc_coeff)
    return wfz.toNumpyArray()

def _compute_ol_wf_kl(sr, kl_coeff, cmask):

    wf_kl = sr.kl_coefficients_to_raster(kl_coeff)
    return wf_kl

# def display_rms_diff_wf_intra_dataset():
#
#     set_data_dir()
#     ftag_ifs_kl = '250806_170800'
#     cmask = _get_cmask()
#     N_MODES_TO_CORRECT = 200
#     sr_zern = SlmRasterizer(cmask, N_MODES_TO_CORRECT)
#     sr_kl = KLSlmRasterizer(cmask, ftag_ifs_kl)
#
#     _setup_matplotlib_for_thesis()
#
#     rec_ftag_kl   = '250808_144900'
#     rec_ftag_zern = '250616_103300'
#     ol_ftag_list  = ['250808_151100','250808_161900','250828_133300',
#                      '250829_111600','250902_101600']
#
#     # --- KL ---
#     kl_means, kl_stds, kl_meas_errs, kl_tot_wfe = get_data_lists(ol_ftag_list, rec_ftag_kl)
#     # --- Zernike ---
#     z_means,  z_stds,  z_meas_errs,  z_tot_wfe  = get_data_lists(ol_ftag_list, rec_ftag_zern)
#
#     kl_coef_cube = np.array(kl_means)*1e-9
#     z_coef_cube = np.array(z_means)*1e-9
# def display_rms_diff_wf_intra_dataset():
#     """
#     Per ogni dataset:
#       1) ricostruisce WF_zern e WF_kl (metri) dai coefficienti medi,
#       2) calcola la differenza WF_zern - WF_kl (masked sulla pupilla),
#       3) valuta la std (nm) della differenza sul pupillo.
#     Produce:
#       - bar plot delle std per dataset con linee guida 6 nm, 7 nm, sqrt(6^2+7^2) nm
#       - un esempio 2D della mappa differenza (nm) per un dataset scelto.
#     """
#     # --- setup e oggetti di rasterizzazione ---
#     set_data_dir()
#     ftag_ifs_kl = '250806_170800'
#     cmask = _get_cmask()
#     N_MODES_TO_CORRECT = 200
#     sr_zern = SlmRasterizer(cmask, N_MODES_TO_CORRECT)
#     sr_kl   = KLSlmRasterizer(cmask, ftag_ifs_kl)
#
#     _setup_matplotlib_for_thesis()
#
#     # tag ricostruttori e dataset
#     rec_ftag_kl   = '250808_144900'
#     rec_ftag_zern = '250616_103300'
#     ol_ftag_list  = ['250808_151100','250808_161900','250828_133300',
#                      '250829_111600','250902_101600']
#
#     # carica coefficienti medi/std ecc. (in nm) e converti in metri
#     kl_means, kl_stds, kl_meas_errs, kl_tot_wfe = get_data_lists(ol_ftag_list, rec_ftag_kl)
#     z_means,  z_stds,  z_meas_errs,  z_tot_wfe  = get_data_lists(ol_ftag_list, rec_ftag_zern)
#     kl_coef_cube = np.array(kl_means) * 1e-9   # (5, 200) in metri
#     z_coef_cube  = np.array(z_means)  * 1e-9   # (5, 200) in metri
#
#     # per-dataset: WF diff std (nm)
#     diff_std_nm = []
#
#     # salveremo anche la mappa differenza di un dataset di esempio
#     example_idx = 2  # ad es. il terzo dataset della lista
#     diff_map_example_nm = None
#     title_example = None
#
#     for i, tag in enumerate(ol_ftag_list):
#         # prendi i primi N_MODES_TO_CORRECT coefficienti (metri)
#         zc  = z_coef_cube[i, :N_MODES_TO_CORRECT]
#         klc = kl_coef_cube[i, :N_MODES_TO_CORRECT]
#
#         # ricostruisci le mappe (MaskedArray in metri)
#         wf_z  = _compute_ol_wf_zern(sr_zern, zc, cmask)
#         wf_kl = _compute_ol_wf_kl(sr_kl,   klc, cmask)
#
#         # assicurati che siano masked array
#         wf_z  = np.ma.array(wf_z)
#         wf_kl = np.ma.array(wf_kl)
#         # differenza sul pupillo
#         diff = (wf_z - wf_kl) - (wf_z - wf_kl).mean()  # metri, masked fuori dal pupillo
#
#         # std sul pupillo (metri) -> nm
#         std_m  = float(np.ma.std(diff))
#         std_nm = std_m * 1e9
#         diff_std_nm.append(std_nm)
#
#         # salva un esempio 2D (nm)
#         if i == example_idx:
#             diff_map_example_nm = diff * 1e9
#             title_example = _fmt_tag(tag)
#
#         # stampa numeri
#         print(f"[{_fmt_tag(tag)}]  std(WF_Z - WF_KL) = {std_nm:.1f} nm (on pupil)")
#
#     diff_std_nm = np.asarray(diff_std_nm)
#
#     # --- Plot 1: barre per dataset con linee guida (6, 7, sqrt(6^2+7^2)) nm ---
#     fig, ax = plt.subplots(figsize=(9.5, 4.8))
#     x = np.arange(len(ol_ftag_list)) + 1
#     ax.bar(x, diff_std_nm, width=0.55, alpha=0.85)
#
#     # etichette asse x = timestamp formattati
#     ax.set_xticks(x)
#     ax.set_xticklabels([_fmt_tag(t) for t in ol_ftag_list], rotation=0)
#
#     # linee guida: rumori singoli e combinati
#     sigma_kl = 6.0   # nm
#     sigma_z  = 7.0   # nm
#     sigma_comb = (sigma_kl**2 + sigma_z**2) ** 0.5
#     ax.axhline(sigma_kl,  color="k", linestyle="--", linewidth=0.9, alpha=0.6, label="6 nm (KL)")
#     ax.axhline(sigma_z,   color="k", linestyle=":",  linewidth=0.9, alpha=0.6, label="7 nm (Zernike)")
#     ax.axhline(sigma_comb,color="r", linestyle="-.", linewidth=1.0, alpha=0.8, label=r"$\sqrt{6^2+7^2}\,\approx\,9.2$ nm")
#
#     _beautify(ax,
#               xlabel="Dataset (open-loop tag)",
#               ylabel="std[ WF$_{\mathrm{Z}}$ − WF$_{\mathrm{KL}}$ ]  [nm]",
#               title="Intra-dataset WF difference (Zernike − KL)")
#     ax.legend(frameon=True, ncols=3)
#     fig.tight_layout()
#
#     # --- Plot 2: esempio 2D della mappa differenza (nm) per un dataset ---
#     if diff_map_example_nm is not None:
#         fig2, ax2 = plt.subplots(figsize=(5.6, 5.2))
#         # colormap centrata in 0
#         vmax = float(np.nanmax(np.abs(diff_map_example_nm)))
#         from matplotlib.colors import TwoSlopeNorm
#         norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
#
#         im = ax2.imshow(diff_map_example_nm, origin='upper', norm=norm)
#         cbar = fig2.colorbar(im, ax=ax2)
#         cbar.set_label("WF$_{\mathrm{Z}}$ − WF$_{\mathrm{KL}}$  [nm]")
#
#         _beautify(ax2,
#                   xlabel="Pixels",
#                   ylabel="Pixels",
#                   title=f"WF difference map (Z−KL) — {title_example}")
#         fig2.tight_layout()
#
#     plt.show()

def _erode_mask_from_center(base_mask_bool, center_yx, frac):
    """
    Erode radialmente la mask booleana del pupillo di una frazione 'frac' del raggio.
    base_mask_bool: True sui pixel del pupillo (2D np.bool_)
    center_yx: (cy, cx) in pixel
    frac: 0.0 ... 0.05 (cioè 0% ... 5%)
    """
    cy, cx = center_yx
    yy, xx = np.indices(base_mask_bool.shape)
    rr = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    # raggio massimo calcolato sui soli pixel True del pupillo
    R = rr[base_mask_bool].max()
    eroded = base_mask_bool & (rr <= (1.0 - float(frac)) * R)
    return eroded

from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable, inset_locator


def _plot_wf_diff_piston_removed(sr_zern, sr_kl, z_coeff_m, kl_coeff_m, cmask, dataset_label="",  overplot_zoom_roi=False):
    """
    Plotta WF_Z - WF_KL (piston-removed) con:
      - colorbar alta come l'asse
      - due inset di zoom su ROI fisse:
          ROI1: 20x30 px centrata in (x=522, y=279)
          ROI2: 20x30 px centrata in (x=1090, y=1109)
      - titolo con P–V e RMS (std) della differenza
    """
    # Ricostruisci (MaskedArray in metri; la mask è il pupillo)
    wf_z  = np.ma.array(_compute_ol_wf_zern(sr_zern, z_coeff_m, cmask))
    wf_kl = np.ma.array(_compute_ol_wf_kl(  sr_kl,   kl_coeff_m, cmask))
    
    tilt_z_modal_command = z_coeff_m.copy()
    tilt_z_modal_command[1:] = 0
    tilt_z_modal_command[0] = 2e-6
    tilt_kl_modal_command = kl_coeff_m.copy()
    tilt_kl_modal_command[1:] = 0
    tilt_kl_modal_command[0] = 2e-6
    tilt_z = np.ma.array(_compute_ol_wf_zern(sr_zern, tilt_z_modal_command, cmask))
    tilt_kl = np.ma.array(_compute_ol_wf_kl(sr_kl, tilt_kl_modal_command , cmask))
    
    # Rimuovi il pistone (media sul pupillo) da ciascuno
    wf_z0 = wf_z  - np.ma.mean(wf_z)
    wf_k0 = wf_kl - np.ma.mean(wf_kl)

    # Differenza (in nm)
    diff_nm = (wf_z0 - wf_k0) * 1e9

    # Statistiche
    std_nm = float(np.ma.std(diff_nm))                     # RMS come std
    pv_nm  = float(np.ma.max(diff_nm) - np.ma.min(diff_nm))

    # Setup figura
    fig, ax = plt.subplots(figsize=(6.6, 5.6))

    # Colormap centrata in 0
    vmax = np.ma.max(diff_nm)
    vmin = np.ma.min(diff_nm)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-120, vcenter=0.0, vmax=120)

    # Immagine principale
    im = ax.imshow(diff_nm, origin="upper", norm=norm)

    # Colorbar con altezza "legata" all'asse (non all'intera figura)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.06)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"W$_{\mathrm{Z}}$ − W$_{\mathrm{KL}}$ [nm]")

    # Titolo con P–V e RMS(std)
    _beautify(ax,
              xlabel="Pixels",
              ylabel="Pixels",
              title=("Open-loop WF difference W$_Z$-W$_{KL}$\n"
                     f"P–V = {pv_nm:.1f} nm   |   RMS (std) = {std_nm:.1f} nm"))
    
    fig, axes = plot_wf_triptych(wf_z, wf_kl, diff_nm,
                             title_z="W$_Z$", title_kl="W$_{KL}$", title_diff="Z−KL", v_min_max_diff=120)
    
    fig2, axes2 = plot_wf_triptych(tilt_z, tilt_kl, (tilt_z-tilt_kl)*1e9,
                             title_z="Tip$_Z$", title_kl="Tip$_{KL}$", title_diff="Z−KL", v_min_max_diff=20)
    
    fig3, axes3 = plot_radial_diagnostics(diff_nm_ma=diff_nm, center_yx=(579,968), nbins=48,
                                    basis_label="Z−KL")
    center = (579, 968)  # (y, x)
    plot_x_profile_through_center(diff_nm, center_yx=center, normalize_radius=True,
                              units_label="nm", title_prefix="W$_Z$ − W${_KL}$")
    
    plot_x_profile_through_center((tilt_z-tilt_kl)*1e9, center_yx=center, normalize_radius=True,
                              units_label="nm", title_prefix="Z − KL")
    # -------------------- ROI helper --------------------
    def _roi_bounds(center_xy, hw_y, hw_x, shape):
        """
        center_xy: (cx, cy) in pixel (attenzione: x=col, y=riga)
        hw_y: half-height (in px), hw_x: half-width (in px)
        shape: (H, W)
        Ritorna (x1, x2, y1, y2) inclusivo/esclusivo compatibile con set_xlim/set_ylim.
        """
        cx, cy = center_xy
        H, W = shape
        y1 = max(0, int(round(cy - hw_y)))
        y2 = min(H, int(round(cy + hw_y)))      # esclusivo per i limiti dell'inset
        x1 = max(0, int(round(cx - hw_x)))
        x2 = min(W, int(round(cx + hw_x)))
        return x1, x2, y1, y2
    
    if overplot_zoom_roi is True: 
        H, W = diff_nm.shape
    
        # --- ROI1: 20x30 px centrata in (x=522, y=279) ---
        roi1_center_xy = (445, 490)
        half_h1, half_w1 = 40,50 # 20x30
        x1a, x2a, y1a, y2a = _roi_bounds(roi1_center_xy, half_h1, half_w1, (H, W))
    
        # rettangolo ROI1
        rect1 = Rectangle((x1a, y1a), (x2a - x1a), (y2a - y1a),
                          fill=False, ec="k", lw=0.8, ls="-", alpha=0.9)
        ax.add_patch(rect1)
    
        # inset ROI1 (in alto a destra)
        axins1 = inset_locator.inset_axes(ax, width="28%", height="28%", loc="upper right", borderpad=1.0)
        axins1.imshow(diff_nm, origin="upper", norm=norm)
        axins1.set_xlim(x1a, x2a)
        axins1.set_ylim(y2a, y1a)  # invertito perché origin="upper"
        axins1.set_xticks([])
        axins1.set_yticks([])
        axins1.set_title("Zoom ROI1", fontsize=8, pad=4)
    
        # --- ROI2: 20x30 px centrata in (x=1090, y=1109) ---
        roi2_center_xy = (591, 959)
        half_h2, half_w2 = 80, 100   # 20x30
        x1b, x2b, y1b, y2b = _roi_bounds(roi2_center_xy, half_h2, half_w2, (H, W))
    
        # rettangolo ROI2
        rect2 = Rectangle((x1b, y1b), (x2b - x1b), (y2b - y1b),
                          fill=False, ec="r", lw=0.8, ls="-", alpha=0.9)
        ax.add_patch(rect2)
    
        # inset ROI2 (in basso a destra)
        axins2 = inset_locator.inset_axes(ax, width="28%", height="28%", loc="lower right", borderpad=1.0)
        axins2.imshow(diff_nm, origin="upper", norm=norm)
        axins2.set_xlim(x1b, x2b)
        axins2.set_ylim(y2b, y1b)  # invertito perché origin="upper"
        axins2.set_xticks([])
        axins2.set_yticks([])
        axins2.set_title("Zoom ROI2", fontsize=8, pad=4)

    fig.tight_layout()

    # log numerico riepilogo
    print(f"[{dataset_label}] std(diff, piston-removed) = {std_nm:.1f} nm   |   P–V = {pv_nm:.1f} nm")

def plot_wf_triptych(wf_z, wf_kl, diff_nm, title_z="WF (Zernike)", title_kl="WF (KL)", title_diff="Z − KL", v_min_max_diff = None):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharex=True, sharey=True)

    # --- normalizzazioni ---
    # stessa scala per wf_z e wf_kl
    vmin_wf = float(np.nanmin([np.ma.min(wf_z/1e-6), np.ma.min(wf_kl/1e-6)]))
    vmax_wf = float(np.nanmax([np.ma.max(wf_z/1e-6), np.ma.max(wf_kl/1e-6)]))
    # scala simmetrica per la differenza (centrata a 0)
    vmax_diff = float(np.nanmax(np.abs(diff_nm)))
    if not np.isfinite(vmax_diff) or vmax_diff == 0:
        vmax_diff = 1.0
    if v_min_max_diff is not None:
        norm_diff = TwoSlopeNorm(vmin=-v_min_max_diff, vcenter=0.0, vmax=v_min_max_diff)
    else:
        norm_diff = TwoSlopeNorm(vmin=-vmax_diff, vcenter=0.0, vmax=vmax_diff)
    # --- pannello 1: WF Zernike ---
    ax = axes[0]
    im0 = ax.imshow(wf_z/1e-6, origin="upper", vmin=vmin_wf, vmax=vmax_wf)
    ptv_wf_z = np.ptp(wf_z/1e-6)
    rms_wf_z = (wf_z/1e-6).std()
    ax.set_title(f"{title_z}\n PtV={ptv_wf_z:0.2f} um,  RMS={rms_wf_z:.2f} um")
    ax.set_xlabel("Pixels"); ax.set_ylabel("Pixels")
    div0 = make_axes_locatable(ax)
    cax0 = div0.append_axes("right", size="4%", pad=0.04)
    cb0 = fig.colorbar(im0, cax=cax0)
    cb0.set_label("Wavefront [um]")   # cambia unità se serve

    # --- pannello 2: WF KL ---
    ax = axes[1]
    im1 = ax.imshow(wf_kl/1e-6, origin="upper", vmin=vmin_wf, vmax=vmax_wf)
    ptv_wf_kl = np.ptp(wf_kl/1e-6)
    rms_wf_kl = (wf_kl/1e-6).std()
    ax.set_title(f"{title_kl}\n PtV={ptv_wf_kl:0.1f} um,  RMS={rms_wf_kl:.0f} um")
    ax.set_xlabel("Pixels")
    div1 = make_axes_locatable(ax)
    cax1 = div1.append_axes("right", size="4%", pad=0.04)
    cb1 = fig.colorbar(im1, cax=cax1)
    cb1.set_label("Wavefront [um]")   # cambia unità se serve

    # --- pannello 3: Differenza ---
    ptv_diff = np.ptp(diff_nm*1e-3)
    rms_diff = (diff_nm).std()
    ax = axes[2]
    im2 = ax.imshow(diff_nm, origin="upper", norm=norm_diff)
    ax.set_title(f"{title_diff}\n PtV={ptv_diff:0.2f} um,  RMS={rms_diff:.0f} nm")
    ax.set_xlabel("Pixels")
    div2 = make_axes_locatable(ax)
    cax2 = div2.append_axes("right", size="4%", pad=0.04)
    cb2 = fig.colorbar(im2, cax=cax2)
    cb2.set_label("WF Difference [nm]")

    fig.tight_layout()
    return fig, axes

def display_rms_diff_wf_intra_dataset():
    """
    Per dataset:
      - Ricostruisce WF_Z e WF_KL (metri) dai coefficienti medi (full e low-pass K=4),
      - Calcola std_pupil(WF_Z - WF_KL) [nm] (full e low-pass),
      - Calcola std su mask erosa per frac in [0%, 1%, 2%, 3%, 4%, 5%] (full e low-pass),
      - Produce:
          (A) bar plot std full-order,
          (B) bar plot std low-pass (K=4),
          (C) subplot 2×1: std vs erosione (sx full, dx low-pass) con curva media spessa.
    """
    # --- setup e rasterizer ---
    set_data_dir()
    ftag_ifs_kl = '250806_170800'
    cmask = _get_cmask()
    N_MODES_TO_CORRECT = 200
    sr_zern = SlmRasterizer(cmask, N_MODES_TO_CORRECT)
    sr_kl   = KLSlmRasterizer(cmask, ftag_ifs_kl)

    _setup_matplotlib_for_thesis()

    # ricostruttori e dataset
    rec_ftag_kl   = '250808_144900'
    rec_ftag_zern = '250616_103300'
    ol_ftag_list  = ['250808_151100','250808_161900','250828_133300',
                     '250829_111600','250902_101600']

    # carica coefficienti medi (nm) e converti in metri
    kl_means, kl_stds, kl_meas_errs, kl_tot_wfe = get_data_lists(ol_ftag_list, rec_ftag_kl)
    z_means,  z_stds,  z_meas_errs,  z_tot_wfe  = get_data_lists(ol_ftag_list, rec_ftag_zern)
    kl_coef_cube = np.array(kl_means) * 1e-9   # (5, 200) m
    z_coef_cube  = np.array(z_means)  * 1e-9   # (5, 200) m

    # impostazioni
    K_LOW = 4
    erosion_fracs = np.array([0.00, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05])  # 0%..5%

    # risultati
    std_full_nm = []
    std_low_nm  = []
    # shape: (n_datasets, n_fracs)
    std_erosion_full_nm = np.zeros((len(ol_ftag_list), len(erosion_fracs)), dtype=float)
    std_erosion_low_nm  = np.zeros_like(std_erosion_full_nm)

    # per coerenza con _get_cmask()
    SLM_PUPIL_CENTER = (579, 968)  # (y,x)

    # loop sui dataset
    for i, tag in enumerate(ol_ftag_list):
        zc_full  = z_coef_cube[i, :N_MODES_TO_CORRECT]
        klc_full = kl_coef_cube[i, :N_MODES_TO_CORRECT]
        zc_low   = z_coef_cube[i, :K_LOW]
        klc_low  = kl_coef_cube[i, :K_LOW]

        # WF full-order (MaskedArray in metri)
        wf_z_full  = np.ma.array(_compute_ol_wf_zern(sr_zern, zc_full, cmask))
        wf_kl_full = np.ma.array(_compute_ol_wf_kl(  sr_kl,   klc_full, cmask))
        diff_full  = wf_z_full - wf_kl_full

        # WF low-pass K=4
        wf_z_low   = np.ma.array(_compute_ol_wf_zern(sr_zern, zc_low, cmask))
        wf_kl_low  = np.ma.array(_compute_ol_wf_kl(  sr_kl,   klc_low, cmask))
        diff_low   = wf_z_low - wf_kl_low

        # std sul pupillo intero (nm)
        std_full_nm.append(float(np.ma.std(diff_full)) * 1e9)
        std_low_nm.append( float(np.ma.std(diff_low))  * 1e9)

        # mask booleana del pupillo (True dentro)
        base_mask_bool = ~np.ma.getmaskarray(wf_z_full)

        # sweep erosione 0..5%
        for j, frac in enumerate(erosion_fracs):
            m_eroded = _erode_mask_from_center(base_mask_bool, SLM_PUPIL_CENTER, frac)
            # Applica la mask erosa alla differenza (mantieni masked fuori)
            diff_full_e = np.ma.array(diff_full, mask=~m_eroded)
            diff_low_e  = np.ma.array(diff_low,  mask=~m_eroded)
            std_erosion_full_nm[i, j] = float(np.ma.std(diff_full_e)) * 1e9
            std_erosion_low_nm[i, j]  = float(np.ma.std(diff_low_e))  * 1e9

        # log numerico
        print(f"[{_fmt_tag(tag)}] std_full = {std_full_nm[-1]:.1f} nm   |   std_low(K=4) = {std_low_nm[-1]:.1f} nm")

    std_full_nm = np.asarray(std_full_nm)
    std_low_nm  = np.asarray(std_low_nm)


    # ===== Plot (C): std vs erosione (subplot: sx full, dx low) =====
    figC, (axC1, axC2) = plt.subplots(1, 2, figsize=(13.6, 5.6), sharey=False)

    # sinistra: full-order
    for i, tag in enumerate(ol_ftag_list):
        axC1.plot(100*erosion_fracs, std_erosion_full_nm[i], marker="o", markersize=3.0,
                  linewidth=1.0, alpha=0.65)#, label=_fmt_tag(tag))
    # curva media (spessa)
    axC1.plot(100*erosion_fracs, std_erosion_full_nm.mean(axis=0),
              linewidth=1.5,color='m')#, label="mean")

    _beautify(axC1,
              xlabel="Pupil reduction [% of radius]",
              ylabel=r"std[ W$_{Z}$ − W$_{KL}$ ]  [nm] ",
              title="Full (200 modes)")
    #axC1.legend(ncols=2, frameon=True)

    # destra: low-pass
    for i, tag in enumerate(ol_ftag_list):
        axC2.plot(100*erosion_fracs, std_erosion_low_nm[i], marker="o", markersize=3.0,
                  linewidth=1.0, alpha=0.65, label=_fmt_tag(tag))
    axC2.plot(100*erosion_fracs, std_erosion_low_nm.mean(axis=0),
              linewidth=1.5,color='m', label="mean")

    _beautify(axC2,
              xlabel="Pupil reduction [% of radius]",
              ylabel="",#r"std[ WF$_Z$ − WF$_{KL}$ ]  [nm]",
              title="Filtered (only first 4 modes)")
    # legenda solo a sinistra per non affollare; se la vuoi anche qui: axC2.legend(...)
    figC.suptitle("W$_Z$ − W$_{KL}$: sensitivity to pupil size", fontsize=14)
    figC.tight_layout(rect=[0, 0, 1, 0.95])
    
    i = 2  # ad es. il terzo dataset
    _plot_wf_diff_piston_removed(
        sr_zern, sr_kl,
        z_coeff_m = z_coef_cube[i, :200],
        kl_coeff_m = kl_coef_cube[i, :200],
        cmask = _get_cmask(),
        dataset_label = _fmt_tag(ol_ftag_list[i]),
        overplot_zoom_roi = True
    )
    
    

    
    plt.show()



def _annular_rms(diff_ma, center_yx, nbins=40):
    """
    RMS anulare della mappa 'diff_ma' (MaskedArray).
    Ritorna:
      r_norm: raggio normalizzato (centro dei bin) in [0,1]
      rms_nm: RMS per anello (stessa unità di diff_ma)
    """
    m = ~np.ma.getmaskarray(diff_ma)
    y, x = np.indices(diff_ma.shape)
    cy, cx = center_yx
    r = np.hypot(y - cy, x - cx)

    rb = r[m].astype(float)
    vb = np.asarray(diff_ma)[m].astype(float)

    R = rb.max()
    edges = np.linspace(0.0, R, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    rms = []
    for i in range(nbins):
        sel = (rb >= edges[i]) & (rb < edges[i+1])
        if sel.any():
            rms.append(np.sqrt((vb[sel]**2).mean()))
        else:
            rms.append(np.nan)

    r_norm = centers / R
    rms = np.array(rms)
    return r_norm, rms

def _cumulative_rms_vs_radius(diff_ma, center_yx, nbins=40):
    """
    RMS cumulativa della differenza entro un raggio crescente.
    Per ogni R_cut: RMS = sqrt( mean( diff^2 | r <= R_cut ) ).
    Ritorna r_norm (0..1) e cum_rms (stessa unità di diff_ma).
    """
    m = ~np.ma.getmaskarray(diff_ma)
    y, x = np.indices(diff_ma.shape)
    cy, cx = center_yx
    r = np.hypot(y - cy, x - cx)

    rb = r[m].astype(float)
    vb2 = (np.asarray(diff_ma)[m].astype(float))**2

    R = rb.max()
    edges = np.linspace(0.0, R, nbins)  # nbins cut radii (escluso il bordo finale)
    cum_rms = []

    for Rcut in edges:
        sel = rb <= Rcut
        if sel.any():
            cum_rms.append(np.sqrt(vb2[sel].mean()))
        else:
            cum_rms.append(np.nan)

    r_norm = edges / R
    return r_norm, np.array(cum_rms)

def plot_radial_diagnostics(diff_nm_ma, center_yx, nbins=40, basis_label="Z−KL (piston removed)"):
    """
    Crea due pannelli (assi condivisi):
      - SX: RMS anulare vs r/R
      - DX: RMS cumulativa vs r/R
    'diff_nm_ma' dev'essere in nm (MaskedArray) e 'center_yx' il centro pupilla (y,x).
    """
    r_norm_ann, rms_ann = _annular_rms(diff_nm_ma, center_yx, nbins=nbins)
    r_norm_cum, rms_cum = _cumulative_rms_vs_radius(diff_nm_ma, center_yx, nbins=nbins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True, sharey=False)

    # Pannello sinistro: RMS anulare
    ax1.plot(r_norm_ann, rms_ann, marker='o', linewidth=1.1, markersize=3.2, alpha=0.9)
    ax1.set_xlabel(r"Normalized radius $r/R$")
    ax1.set_ylabel(r"Annular RMS of W$_Z$ - W$_{KL}$ [nm]")
    ax1.set_title(f"")
    ax1.grid(True, alpha=0.25, linestyle=':')

    # Evidenzia l'ultimo 10% di raggio (tipico rim)
    ax1.axvspan(0.9, 1.0, color='k', alpha=0.06)

    # Pannello destro: RMS cumulativa entro r/R
    ax2.plot(r_norm_cum, rms_cum, marker='.', linewidth=1.1, markersize=3.0, alpha=0.9)
    ax2.set_xlabel(r"Normalized radius $r/R$")
    ax2.set_ylabel(r"Cumulative RMS within $r$ [nm]")
    ax2.set_title("Cumulative RMS within radius")
    ax2.grid(True, alpha=0.25, linestyle=':')

    fig.tight_layout()
    return fig, (ax1, ax2)



def plot_x_profile_through_center(diff_ma, center_yx, normalize_radius=True,
                                  units_label="nm", title_prefix="WF difference"):
    """
    Traccia il profilo lungo l'asse x (riga passante per il centro pupilla) della mappa 'diff_ma'.

    Parametri
    ---------
    diff_ma : np.ma.MaskedArray
        Mappa differenza (es. in nm). La mask deve rappresentare il pupillo.
    center_yx : tuple
        Centro della pupilla (y, x) in pixel.
    normalize_radius : bool
        Se True, l'asse x è normalizzato al raggio (x/R in [-1, +1]).
        Se False, mostra coordinate in pixel.
    units_label : str
        Etichetta dell'unità del profilo (es. "nm", "m").
    title_prefix : str
        Prefisso del titolo.

    Ritorna
    -------
    fig, ax : figure e axes di matplotlib
    """
    # estrai riga al centro (y0) e mask
    y0, x0 = int(round(center_yx[0])), int(round(center_yx[1]))
    arr = np.ma.array(diff_ma)  # assicurati sia MaskedArray
    H, W = arr.shape

    # protezioni sui limiti
    y0 = np.clip(y0, 0, H-1)
    x0 = np.clip(x0, 0, W-1)

    row_vals = arr[y0, :]
    row_mask = ~np.ma.getmaskarray(row_vals)  # True = valido (dentro pupilla)

    # indici validi (segmento pupilla sulla riga)
    valid_idx = np.where(row_mask)[0]
    if valid_idx.size == 0:
        raise ValueError("La riga centrale non interseca la pupilla (nessun pixel valido).")

    x_left, x_right = valid_idx[0], valid_idx[-1]
    prof = row_vals[valid_idx].astype(float)

    # coordinate x (relative al centro) e raggio
    x_pix = valid_idx
    x_rel = x_pix - x0
    R_pix = max(x0 - x_left, x_right - x0)  # raggio orizzontale sulla riga

    if normalize_radius and R_pix > 0:
        x_axis = x_rel / float(R_pix)  # in [-1, +1] (almeno idealmente)
        x_label = r"$x/R$ (centered)"
    else:
        x_axis = x_pix
        x_label = "Pixels (x-axis)"

    # statistiche del profilo (sul solo segmento in pupilla)
    prof_mean = float(np.mean(prof))
    prof_std  = float(np.std(prof, ddof=0))   # std come RMS del profilo (già zero-mean? no: std è robusta)
    prof_pv   = float(np.max(prof) - np.min(prof))

    # plot
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(x_axis, prof, lw=1.4, marker='.', ms=3.0, alpha=0.95)

    # linee verticali ai bordi della pupilla
    if normalize_radius and R_pix > 0:
        ax.axvline(- (x0 - x_left)/R_pix, color='k', ls='--', lw=0.9, alpha=0.7)
        ax.axvline(  (x_right - x0)/R_pix, color='k', ls='--', lw=0.9, alpha=0.7)
    else:
        ax.axvline(x_left,  color='k', ls='--', lw=0.9, alpha=0.7)
        ax.axvline(x_right, color='k', ls='--', lw=0.9, alpha=0.7)

    # estetica
    try:
        _beautify(ax,
                  xlabel=x_label,
                  ylabel=f"Difference [{units_label}]",
                  title=(f"{title_prefix} — X profile\n"
                         f"RMS (std) = {prof_std:.1f} {units_label}   |   P–V = {prof_pv:.1f} {units_label}"),
                  xmaj=7, ymaj=6)
    except NameError:
        # fallback se _beautify non è definita nel tuo ambiente
        ax.set_xlabel(x_label); ax.set_ylabel(f"Difference [{units_label}]")
        ax.set_title(f"{title_prefix} — X profile through pupil center\n"
                     f"RMS (std) = {prof_std:.1f} {units_label}   |   P–V = {prof_pv:.1f} {units_label}")
        ax.grid(alpha=0.25, linestyle=":")

    fig.tight_layout()
    return fig, ax
