
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from astropy.io import fits
from bronte.startup import set_data_dir
from bronte.package_data import other_folder
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Ellipse
from scipy.stats import chi2

_BETA_MC_CACHE = {}

# ----------------------------
# Parametri nominali e incertezze (modifica qui se serve)
# ----------------------------
LAM_C   = 632.8e-9        # m
LAM_SIG = 0.2e-9          # m (1σ)

D_C     = 1090 * 9.2e-6   # m
D_SIG   = 2.65e-6         # m (assoluto ~ quantizzazione)

F_C     = 250e-3          # m
F_SIG   = 0.01 * F_C      # m (±1% relativo)

P_C     = 4.65e-6         # m
P_SIG   = 0.0             # m (assunto nullo)

# ----------------------------------------------------
# Stile per figure leggibili in tesi (context manager)
# ----------------------------------------------------
def _thesis_style():
    return mpl.rc_context({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 11,
        "lines.linewidth": 1.2,
        "lines.markersize": 6,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })

# ----------------------------
# I/O FITS per risultati MC β
# ----------------------------
def _load_beta_mc_fits(fname):
    with fits.open(fname) as hdul:
        hdr = hdul[0].header
        tab_samp = hdul["SAMPLES"].data
        tab_sens = hdul["SENS"].data

        samples = {
            "lambda": np.array(tab_samp["lambda"], dtype=float),
            "D":      np.array(tab_samp["D"],      dtype=float),
            "f":      np.array(tab_samp["f"],      dtype=float),
            "p":      np.array(tab_samp["p"],      dtype=float),
            "beta":   np.array(tab_samp["beta"],   dtype=float),
        }
        beta0 = float(hdr["BETA0"])
        sens  = {
            "lambda": float(tab_sens["rel_d"][0]),
            "D":      float(tab_sens["rel_d"][1]),
            "f":      float(tab_sens["rel_d"][2]),
        }
        details = {
            "beta0": beta0,
            "beta_lambda_pm": (float(tab_sens["beta_min"][0]), float(tab_sens["beta_plu"][0])),
            "beta_D_pm":      (float(tab_sens["beta_min"][1]), float(tab_sens["beta_plu"][1])),
            "beta_f_pm":      (float(tab_sens["beta_min"][2]), float(tab_sens["beta_plu"][2])),
        }
        meta = {
            "N_SAMP": int(hdr["N_SAMP"]),
            "B_MEAN": float(hdr["B_MEAN"]),
            "B_STD":  float(hdr["B_STD"]),
            "B_CI_L": float(hdr["B_CI_L"]),
            "B_CI_H": float(hdr["B_CI_H"]),
            "LAM_C":  float(hdr["LAM_C"]),  "LAM_SD": float(hdr["LAM_SD"]),
            "D_C":    float(hdr["D_C"]),    "D_SD":   float(hdr["D_SD"]),
            "F_C":    float(hdr["F_C"]),    "F_SD":   float(hdr["F_SD"]),
            "P_C":    float(hdr["P_C"]),    "P_SD":   float(hdr["P_SD"]),
            "SEED":   int(hdr.get("SEED", -1)),
        }
    return samples, beta0, sens, details, meta

def load_beta_mc_data(ftag):
    """
    Carica dal FITS (other_folder()/<ftag>.fits) e mette in cache.
    Ritorna (samples, beta0, sens, details, meta).
    """
    set_data_dir()
    fname = other_folder() / (ftag + '.fits')
    samples, beta0, sens, details, meta = _load_beta_mc_fits(str(fname))
    _BETA_MC_CACHE[ftag] = (samples, beta0, sens, details, meta)
    return samples, beta0, sens, details, meta

# ----------------------------
# Helper: colori per seed
# ----------------------------
def _assign_colors_by_seed(seed_list):
    unique_seeds = []
    for s in seed_list:
        if s not in unique_seeds:
            unique_seeds.append(s)
    cmap = mpl.cm.get_cmap('tab10' if len(unique_seeds) <= 10 else 'tab20', len(unique_seeds))
    color_map = {s: cmap(i) for i, s in enumerate(unique_seeds)}
    return color_map, unique_seeds

# ----------------------------
# Plot β vs parametri (scatter)
# ----------------------------

    
def _plot_scatter_separati(samples, beta0, savepath_prefix=None, show_bands=True,
                           nbins=24, q_main=(0.16,0.84), q_wide=(0.025,0.975)):
    
    lam = samples["lambda"]; D = samples["D"]; f = samples["f"]; b = samples["beta"]

    with _thesis_style():
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)

        # ---------- 1) β vs λ ----------
        ax0 = axes[0]
        ax0.plot(lam*1e9, b, '.', ms=5, alpha=0.10)
        ax0.plot(LAM_C*1e9, beta0, marker='x', color='red', ms=9, mew=1.5)
        _confidence_ellipse(ax0, lam*1e9, b, level=0.68, color='g', lw=1)
        _confidence_ellipse(ax0, lam*1e9, b, level=0.95, color='g', lw=1, ls='--')

        ax0.set_xlabel(r'$\lambda$ [nm]')
        ax0.set_ylabel(r'$\beta$')
        ax0.set_title(r'$\beta$ vs $\lambda$')
        ax0.grid(ls='--', alpha=0.30)
        # formatter compatto dei tick x
        ax0.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
        ax0.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))

        # legenda con simboli nominale + ellissi
        h_nom = Line2D([], [], marker='x', color='red', lw=0, ms=9, mew=1.5, label=r'$(\lambda_0,\beta_0)$')
        h_68  = Line2D([], [], color='g', lw=1, label='68%')
        h_95  = Line2D([], [], color='g', lw=1, ls='--', label='95%')
        ax0.legend(handles=[h_nom, h_68, h_95], loc='lower left', frameon=True)

        # ---------- 2) β vs D ----------
        ax1 = axes[1]
        ax1.plot(D*1e6, b, '.', ms=4.5, alpha=0.10)
        ax1.plot(D_C*1e6, beta0, marker='x', color='red', lw=0, ms=9, mew=1.5, label=r'$(D_0,\beta_0)$')
        ax1.set_xlabel(r'$D$ [$\mu$m]')
        ax1.set_title(r'$\beta$ vs $D$')
        ax1.grid(ls='--', alpha=0.30)

        if show_bands:
            centers, ql, qmed, qh, ql95, qh95 = _quantile_bands(D*1e6, b, nbins=nbins,
                                                                q_main=q_main, q_wide=q_wide, min_per_bin=20)
            if np.isfinite(ql95).any():
                ax1.fill_between(centers, ql95, qh95, color='C1', alpha=0.15, label='2.5–97.5%')
            if np.isfinite(ql).any():
                ax1.fill_between(centers, ql, qh, color='C1', alpha=0.30, label='16–84%')
            ax1.legend(loc='lower left', frameon=True)

        # ---------- 3) β vs f ----------
        ax2 = axes[2]
        ax2.plot(f*1e3, b, '.', ms=5, alpha=0.01)
        ax2.plot(F_C*1e3, beta0, marker='x', color='red', ms=9, mew=1.5)
        _confidence_ellipse(ax2, f*1e3, b, level=0.68, color='g', lw=1)
        _confidence_ellipse(ax2, f*1e3, b, level=0.95, color='g', lw=1, ls='--')

        ax2.set_xlabel(r'$f$ [mm]')
        ax2.set_title(r'$\beta$ vs $f$')
        ax2.grid(ls='--', alpha=0.30)
        # formatter compatto dei tick x
        ax2.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
        ax2.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))

        # legenda con simboli nominale + ellissi
        h_nom_f = Line2D([], [], marker='x', color='red', lw=0, ms=9, mew=1.5, label=r'$(f_0,\beta_0)$')
        ax2.legend(handles=[h_nom_f, h_68, h_95], loc='lower left', frameon=True)

        # ----- inset zoom su β vs f -----
        axins = inset_axes(ax2, width="38%", height="28%", loc="upper right", borderpad=0.6)
        axins.plot(f*1e3, b, '.', ms=3, alpha=0.08)
        axins.plot(F_C*1e3, beta0, marker='x', color='red', ms=7, mew=1.2)
        _confidence_ellipse(axins, f*1e3, b, level=0.68, color='g', lw=1)
        _confidence_ellipse(axins, f*1e3, b, level=0.95, color='g', lw=1, ls='--')
        axins.set_xlim(249, 251)
        axins.set_ylim(0.0678, 0.0688)
        axins.grid(ls='--', alpha=0.25)
        axins.tick_params(labelsize=6)
        axins.set_title("zoom", fontsize=8, pad=1)
        

        fig.suptitle(r'Marginal distributions of $\beta$', y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        if savepath_prefix:
            plt.savefig(savepath_prefix + "_scatter.png", bbox_inches="tight")
        plt.show()


# ----------------------------
# Istogramma β
# ----------------------------
def _plot_hist_beta(samples, beta0, savepath_prefix=None):
    b = samples["beta"]
    with _thesis_style():
        plt.figure(figsize=(6.4, 4.0))
        plt.hist(b, bins=62, alpha=0.85, edgecolor='k')
        plt.axvline(beta0, color='r', ls='--', label=r'$\beta_0$ nominal')
        plt.xlabel(r'$\beta = I^{\rm DL}_{\rm peak} / I^{\rm DL}_{\rm tot}$')
        plt.ylabel('Counts'); plt.grid(ls='--', alpha=0.3)
        plt.title('β occurrence'); plt.legend(frameon=True)
        plt.tight_layout()
        if savepath_prefix:
            plt.savefig(savepath_prefix + "_hist.png", bbox_inches="tight")
        plt.show()

# ----------------------------
# One-at-a-time sensitivity
# ----------------------------
def _plot_sensitivity_bars(sens, savepath_prefix=None):
    labels = [r'$\lambda$', r'$D$', r'$f$']
    vals = np.array([sens["lambda"], sens["D"], sens["f"]]) * 100.0
    with _thesis_style():
        plt.figure(figsize=(5.6, 3.6))
        plt.bar(labels, vals)
        plt.ylabel(r'|Δβ| for ±1σ [%]')
        plt.title('β sensitivity')
        plt.grid(axis='y', ls='--', alpha=0.4)
        plt.tight_layout()
        if savepath_prefix:
            plt.savefig(savepath_prefix + "_sens.png", bbox_inches="tight")
        plt.show()

# ----------------------------
# Report (riutilizza i tre plot sopra)
# ----------------------------
def show_beta_data_res(ftag, save_fig=False):
    """
    Mostra scatter β vs parametri, istogramma, e barre di sensibilità
    leggendo il FITS salvato in precedenza.
    """
    if ftag not in _BETA_MC_CACHE:
        load_beta_mc_data(ftag)

    entry = _BETA_MC_CACHE[ftag]
    if len(entry) == 4:
        samples, beta0, sens, details = entry
        meta = None
    else:
        samples, beta0, sens, details, meta = entry

    if meta is not None:
        print(f"[replay] N_samples={meta.get('N_SAMP', len(samples['beta']))}")
        print(f"[replay] seed={meta.get('SEED', -1)}")

    b = samples["beta"]
    mu = np.mean(b); sd = np.std(b, ddof=1); lo, hi = np.percentile(b, [2.5, 97.5])
    print(f"[replay] β0={beta0:.6e} | mean={mu:.6e}, std={sd:.6e} ({sd/mu*100:.3f}%), 95%CI=[{lo:.6e},{hi:.6e}]")
    print("Sensitivities (±1σ):")
    print(f"  lambda: {sens['lambda']*100:.3f}%   β(-σ), β(+σ) = {details['beta_lambda_pm'][0]:.6e}, {details['beta_lambda_pm'][1]:.6e}")
    print(f"  D     : {sens['D']*100:.3f}%   β(-σ), β(+σ) = {details['beta_D_pm'][0]:.6e}, {details['beta_D_pm'][1]:.6e}")
    print(f"  f     : {sens['f']*100:.3f}%   β(-σ), β(+σ) = {details['beta_f_pm'][0]:.6e}, {details['beta_f_pm'][1]:.6e}")

    set_data_dir()
    fname = other_folder() / (ftag + '.fits')
    saveprefix = str(fname.with_suffix("")) if save_fig else None

    _plot_scatter_separati(samples, beta0, savepath_prefix=saveprefix)
    _plot_hist_beta(samples, beta0, savepath_prefix=saveprefix)
    _plot_sensitivity_bars(sens, savepath_prefix=saveprefix)

# ----------------------------
# Convergenza MC di σβ vs N
# ----------------------------
def show_mc_convergence(
    ftag_list=None,
    rel_ref='beta_mean',   # 'beta_mean' oppure 'beta0'
    save_fig=False
):
    """
    Carica una serie di simulazioni MC (ognuna salvata in un FITS)
    e mostra la convergenza di sigma_beta con N. Colori raggruppati per seed.
    Legenda compatta: un solo handle nero che spiega lo stile “one seed”.
    """
    if ftag_list is None:
        ftag_list = [
            '251009_111400','251009_114200','251009_115500','251009_122400',
            '251009_130100','251009_151200','251009_171500','251009_193500',
            '251010_080200','251010_100200'
        ]

    # raccogli statistiche da ciascun file
    records = []
    for ftag in ftag_list:
        samples, beta0, sens, details, meta = load_beta_mc_data(ftag)
        b = samples["beta"]
        N = int(meta.get("N_SAMP", len(b)))
        seed = int(meta.get("SEED", -1))
        sigma = float(np.std(b, ddof=1))
        mu = float(np.mean(b))
        se_mc = sigma / np.sqrt(2.0 * max(N-1, 1))
        ref = float(beta0) if str(rel_ref).lower() == 'beta0' else mu
        rel_err = sigma / ref if ref != 0.0 else np.nan
        records.append({"ftag": ftag, "N": N, "seed": seed, "sigma": sigma,
                        "se_mc": se_mc, "beta0": float(beta0), "mean": mu, "rel": rel_err})

    records = sorted(records, key=lambda r: (r["seed"], r["N"], r["ftag"]))
    color_map, seed_order = _assign_colors_by_seed([r["seed"] for r in records])

    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        groups[r["seed"]].append(r)

    with _thesis_style():
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
        ax1, ax2, ax3 = axes

        # 1) σβ vs N con barre SE_mc
        for seed in seed_order:
            grp = sorted(groups[seed], key=lambda r: r["N"])
            N_vals = np.array([g["N"] for g in grp], dtype=float)
            sigmas = np.array([g["sigma"] for g in grp], dtype=float)
            sevals = np.array([g["se_mc"] for g in grp], dtype=float)
            col = color_map[seed]
            ax1.errorbar(N_vals, sigmas, yerr=sevals, fmt='o-', ms=5, lw=1.2, elinewidth=1.0,
                         capsize=2.5, color=col, alpha=0.95)
        ax1.set_xscale('log')
        ax1.set_xlabel(r"$N$ samples")
        ax1.set_ylabel(r"$\hat{\sigma}_\beta$")
        ax1.set_title(r"$\hat{\sigma}_\beta$ vs $N$")
        ax1.grid(ls='--', alpha=0.35)
        # legenda compatta (un solo handle nero)
        seed_handle = Line2D([0],[0], color='k', marker='o', lw=1.2, label='i-th seed')
        ax1.legend(handles=[seed_handle], loc='best', frameon=True)

        # 2) errore relativo vs N
        for seed in seed_order:
            grp = sorted(groups[seed], key=lambda r: r["N"])
            N_vals = np.array([g["N"] for g in grp], dtype=float)
            rels = np.array([g["rel"] for g in grp], dtype=float)
            col = color_map[seed]
            ax2.plot(N_vals, rels*100.0, 'o-', ms=5, lw=1.2, color=col, alpha=0.95)
        ax2.set_xscale('log')
        ax2.set_xlabel(r"$N$ samples")
        ax2.set_ylabel(r"$\hat{\sigma}_\beta / \bar{\beta}$  [%]" if str(rel_ref).lower() != 'beta0'
                       else r"$\hat{\sigma}_\beta / \beta_0$  [%]")
        ax2.set_title(r"Relative error vs $N$")
        ax2.grid(ls='--', alpha=0.35)

        # 3) SE_mc/σβ [%] vs N con curva teorica 100/sqrt(2(N-1))
        for seed in seed_order:
            grp = sorted(groups[seed], key=lambda r: r["N"])
            N_vals = np.array([g["N"] for g in grp], dtype=float)
            frac = np.array([ (g["se_mc"]/g["sigma"] if g["sigma"]>0 else np.nan) for g in grp], dtype=float)
            col = color_map[seed]
            ax3.plot(N_vals, frac*100.0, 'o', ms=5, lw=1.2, color=col, alpha=0.95)
        N_cont = np.logspace(np.log10(max(min([r["N"] for r in records]), 2)),
                             np.log10(max([r["N"] for r in records])), 200)
        theo = 1.0 / np.sqrt(2.0*(N_cont-1.0))
        ax3.plot(N_cont, theo*100.0, 'k--', lw=1.3, label=r"$100/\sqrt{2(N-1)}$")
        ax3.set_xscale('log')
        ax3.set_xlabel(r"$N$ samples")
        ax3.set_ylabel(r"SE$_{\rm MC}(\hat{\sigma}_\beta)/\hat{\sigma}_\beta$  [%]")
        ax3.set_title(r"Sampling noise on $\hat{\sigma}_\beta$")
        ax3.grid(ls='--', alpha=0.35)
        ax3.legend(loc='best', frameon=True)

        fig.suptitle(r"Convergence of $\hat{\sigma}_\beta$", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if save_fig:
            set_data_dir()
            fname = other_folder() / (ftag_list[-1] + ".fits")
            out = fname.with_suffix("").as_posix() + "_convergence.png"
            plt.savefig(out, bbox_inches="tight")
            print(f"[saved figure] {out}")

        plt.show()

    # tabella di riepilogo in terminale
    print("\nSummary per run:")
    print("ftag                seed      N     sigma_beta       SE_MC       rel_err[%]")
    for r in sorted(records, key=lambda x: (x["seed"], x["N"], x["ftag"])):
        relp = 100.0 * r["rel"]
        print(f"{r['ftag']:<18} {r['seed']:>5d}  {r['N']:>6d}   {r['sigma']:>12.6e}  {r['se_mc']:>10.6e}   {relp:>9.3f}")
        


def _confidence_ellipse(ax, x, y, level=0.68, **kwargs):
    """
    Ellisse di confidenza per la gaussiana bivariata stimata da (x,y).
    level: 0.68 o 0.95, ecc. (probabilità contenuta nell'ellisse).
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    mx, my = x.mean(), y.mean()
    C = np.cov(x, y, ddof=1)  # 2x2
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]

    # raggio in sigma lungo gli autovettori
    s = np.sqrt(chi2.ppf(level, df=2))
    width = 2*s*np.sqrt(evals[0])
    height = 2*s*np.sqrt(evals[1])
    angle = np.degrees(np.arctan2(evecs[1,0], evecs[0,0]))
    ell = Ellipse((mx,my), width, height, angle=angle, fill=False, **kwargs)
    ax.add_patch(ell)
    return ell

def _quantile_bands(x, y, nbins=24, q_main=(0.16, 0.84), q_wide=(0.025, 0.975), min_per_bin=20):
    """
    Calcola bande di quantile condizionali di y|x.
    Ritorna: centers, qlow_main, qmed, qhigh_main, qlow_wide, qhigh_wide (alcuni possono essere None).
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    edges = np.linspace(np.nanmin(x), np.nanmax(x), nbins+1)
    centers = 0.5*(edges[1:]+edges[:-1])

    ql_main = np.full(nbins, np.nan)
    qmed    = np.full(nbins, np.nan)
    qh_main = np.full(nbins, np.nan)
    ql_wide = np.full(nbins, np.nan)
    qh_wide = np.full(nbins, np.nan)

    for i in range(nbins):
        m = (x >= edges[i]) & (x < edges[i+1]) & np.isfinite(y)
        if np.count_nonzero(m) >= min_per_bin:
            yy = y[m]
            qmed[i]    = np.quantile(yy, 0.50)
            ql_main[i] = np.quantile(yy, q_main[0]); qh_main[i] = np.quantile(yy, q_main[1])
            if q_wide is not None:
                ql_wide[i] = np.quantile(yy, q_wide[0]); qh_wide[i] = np.quantile(yy, q_wide[1])
    return centers, ql_main, qmed, qh_main, ql_wide, qh_wide
