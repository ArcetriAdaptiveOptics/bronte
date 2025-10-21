
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from bronte.scao.telemetry.mains.main250902_analysing_bench_static_abs import load_ol_rec_modes


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
            lbl = f" ME={meas_err_list[i]:.0f} nm"
        else:
            lbl = _fmt_tag(tag)
        ax.semilogy(x, y, marker=".", markersize=2.5, label=lbl, alpha=0.95)

    _beautify(ax,
              xlabel="Mode index",
              ylabel="STD of reconstructed modes [nm] RMS WF",
              title=f"Temporal variability — {basis_name} base ({n_modes} modes)",
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

    