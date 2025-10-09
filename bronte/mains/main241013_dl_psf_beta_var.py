import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from astropy.io import fits
from datetime import datetime

# --- importa le tue utility/progetto ---
from bronte.utils.raw_strehl_ratio_computer import StrehlRatioComputer
from bronte.startup import set_data_dir
from bronte.package_data import other_folder

# cache globale per il reload
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

# ----------------------------
# Helper: ricalcola β per dati (λ, D, f, p)
# ----------------------------
def _set_sr_params_and_recompute(sr: StrehlRatioComputer, lam, D, f, p):
    sr._wl = float(lam)
    sr._pupil_diameter = float(D)
    sr._telescope_focal_length = float(f)
    sr._ccd_pixel_size = float(p)

    # derivate coerenti con la tua classe
    sr._pixel_scale_in_arcsec = sr._ccd_pixel_size / sr._telescope_focal_length * sr.RAD2ARCSEC
    sr._dl_size_in_arcsec     = sr._wl / sr._pupil_diameter * sr.RAD2ARCSEC
    sr._dl_size_in_pixels     = sr._dl_size_in_arcsec / sr._pixel_scale_in_arcsec

    sr._compute_dl_psf()
    beta = float(sr._fitted_dl_max_au / sr._total_dl_flux)
    return beta

def _beta_nominale():
    sr = StrehlRatioComputer()
    return _set_sr_params_and_recompute(sr, LAM_C, D_C, F_C, P_C)

# ----------------------------
# Monte Carlo & Sensibilità
# ----------------------------
def _monte_carlo_beta(n_samples=800, seed=0):
    rng = np.random.default_rng(seed)
    lam = rng.normal(LAM_C, LAM_SIG, size=n_samples)
    
    SLM_PITCH = 9.2e-6  # m
    D = rng.uniform(D_C - SLM_PITCH/2, D_C + SLM_PITCH/2, size=n_samples)

    f   = rng.normal(F_C,   F_SIG,   size=n_samples)
    p   = np.full(n_samples, P_C)  # nessuna incertezza su p

    # clip minimi per evitare valori non fisici
    lam = np.clip(lam, 1e-9, None)
    D   = np.clip(D,   1e-6, None)
    f   = np.clip(f,   1e-3, None)

    sr = StrehlRatioComputer()
    beta = np.zeros(n_samples, dtype=float)
    for i in range(n_samples):
        beta[i] = _set_sr_params_and_recompute(sr, lam[i], D[i], f[i], p[i])
    
    samples = {"lambda": lam, "D": D, "f": f, "p": p, "beta": beta, "seed": int(seed)}
    return samples

def _one_at_a_time_sensitivity():
    sr = StrehlRatioComputer()
    beta0 = _set_sr_params_and_recompute(sr, LAM_C, D_C, F_C, P_C)

    def _rel_change(param, minus, plus):
        sr1, sr2 = deepcopy(sr), deepcopy(sr)
        if param == "lambda":
            b_minus = _set_sr_params_and_recompute(sr1, minus, D_C, F_C, P_C)
            b_plus  = _set_sr_params_and_recompute(sr2, plus,  D_C, F_C, P_C)
        elif param == "D":
            b_minus = _set_sr_params_and_recompute(sr1, LAM_C, minus, F_C, P_C)
            b_plus  = _set_sr_params_and_recompute(sr2, LAM_C, plus,  F_C, P_C)
        elif param == "f":
            b_minus = _set_sr_params_and_recompute(sr1, LAM_C, D_C, minus, P_C)
            b_plus  = _set_sr_params_and_recompute(sr2, LAM_C, D_C, plus,  P_C)
        else:
            raise ValueError("param non supportato")
        drel = 0.5 * (abs(b_plus - beta0) + abs(b_minus - beta0)) / beta0
        return float(drel), float(b_minus), float(b_plus)

    sens = {}
    sens["lambda"], bLminus, bLplus = _rel_change("lambda", LAM_C - LAM_SIG, LAM_C + LAM_SIG)
    sens["D"],      bDminus, bDplus = _rel_change("D",      D_C   - D_SIG,   D_C   + D_SIG)
    sens["f"],      bFminus, bFplus = _rel_change("f",      F_C   - F_SIG,   F_C   + F_SIG)

    details = {
        "beta0": beta0,
        "beta_lambda_pm": (bLminus, bLplus),
        "beta_D_pm":      (bDminus, bDplus),
        "beta_f_pm":      (bFminus, bFplus),
    }
    return sens, details

# ----------------------------
# FITS I/O
# ----------------------------
def _save_beta_mc_fits(fname, samples, beta0, sens, details, seed=None):
    """
    Salva TUTTI i campioni + meta in un unico FITS:
      - PrimaryHDU (header con meta)
      - BinTableHDU 'SAMPLES' con colonne: lambda, D, f, p, beta
      - BinTableHDU 'SENS'    con +/-σ e variazioni relative
    """
    lam = samples["lambda"]; D = samples["D"]; f = samples["f"]; p = samples["p"]; b = samples["beta"]
    n = len(b)
    mu = float(np.mean(b)); sd = float(np.std(b, ddof=1))
    lo, hi = np.percentile(b, [2.5, 97.5])

    hdr = fits.Header()
    hdr["DATE"]    = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    hdr["N_SAMP"]  = n
    hdr["BETA0"]   = beta0
    hdr["B_MEAN"]  = mu
    hdr["B_STD"]   = sd
    hdr["B_CI_L"]  = lo
    hdr["B_CI_H"]  = hi

    if "seed" in samples:
        seed = samples["seed"]
    if seed is not None:
        hdr["SEED"] = int(seed)
    # nominali e sigma
    hdr["LAM_C"]   = LAM_C;   hdr["LAM_SD"] = LAM_SIG
    hdr["D_C"]     = D_C;     hdr["D_SD"]   = D_SIG
    hdr["F_C"]     = F_C;     hdr["F_SD"]   = F_SIG
    hdr["P_C"]     = P_C;     hdr["P_SD"]   = P_SIG
    
    # Primary
    phdu = fits.PrimaryHDU(header=hdr)

    # Tabella campioni
    cols = [
        fits.Column(name="lambda", array=lam.astype(np.float64), format="D", unit="m"),
        fits.Column(name="D",      array=D.astype(np.float64),   format="D", unit="m"),
        fits.Column(name="f",      array=f.astype(np.float64),   format="D", unit="m"),
        fits.Column(name="p",      array=p.astype(np.float64),   format="D", unit="m"),
        fits.Column(name="beta",   array=b.astype(np.float64),   format="D"),
    ]
    hdu_samples = fits.BinTableHDU.from_columns(cols, name="SAMPLES")

    # Tabella sensibilità
    sens_labels = np.array(["lambda", "D", "f"])
    beta_minus  = np.array([details["beta_lambda_pm"][0], details["beta_D_pm"][0], details["beta_f_pm"][0]], dtype=np.float64)
    beta_plus   = np.array([details["beta_lambda_pm"][1], details["beta_D_pm"][1], details["beta_f_pm"][1]], dtype=np.float64)
    rel_change  = np.array([sens["lambda"], sens["D"], sens["f"]], dtype=np.float64)

    sens_cols = [
        fits.Column(name="param",     array=sens_labels.astype("S16"), format="16A"),
        fits.Column(name="beta_min",  array=beta_minus, format="D"),
        fits.Column(name="beta_plu",  array=beta_plus,  format="D"),
        fits.Column(name="rel_d",     array=rel_change, format="D"),  # relativo (fractions)
    ]
    hdu_sens = fits.BinTableHDU.from_columns(sens_cols, name="SENS")

    hdul = fits.HDUList([phdu, hdu_samples, hdu_sens])
    hdul.writeto(fname, overwrite=True)

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

# ----------------------------
# Plotting separato
# ----------------------------
def _plot_scatter_separati(samples, beta0, savepath_prefix=None):
    lam = samples["lambda"]; D = samples["D"]; f = samples["f"]; b = samples["beta"]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=True)
    # λ
    axes[0].plot(lam*1e9, b, '.', ms=4, alpha=0.6)
    axes[0].set_xlabel(r'$\lambda$ [nm]'); axes[0].set_ylabel(r'$\beta$'); axes[0].grid(ls='--', alpha=0.3)
    axes[0].set_title(r'$\beta$ vs $\lambda$')
    # D
    axes[1].plot(D*1e6, b, '.', ms=4, alpha=0.6)
    axes[1].set_xlabel(r'$D$ [$\mu$m]'); axes[1].grid(ls='--', alpha=0.3)
    axes[1].set_title(r'$\beta$ vs $D$')
    # f
    axes[2].plot(f*1e3, b, '.', ms=4, alpha=0.6)
    axes[2].set_xlabel(r'$f$ [mm]'); axes[2].grid(ls='--', alpha=0.3)
    axes[2].set_title(r'$\beta$ vs $f$')

    # linea orizzontale β0
    for ax in axes:
        ax.axhline(beta0, color='r', ls='--', lw=1.0, alpha=0.7)

    fig.suptitle(r'Monte Carlo samples: $\beta$ vs parameters', y=1.02)
    fig.tight_layout()
    if savepath_prefix:
        plt.savefig(savepath_prefix + "_scatter.png", dpi=180, bbox_inches="tight")
    plt.show()

def _plot_hist_beta(samples, beta0, savepath_prefix=None):
    b = samples["beta"]
    plt.figure(figsize=(6.2, 3.8))
    plt.hist(b, bins=40, alpha=0.85, edgecolor='k')
    plt.axvline(beta0, color='r', ls='--', label=r'$\beta_0$ nominal')
    plt.xlabel(r'$\beta = I^{\rm DL}_{\rm peak} / F^{\rm DL}_{\rm tot}$')
    plt.ylabel('Counts'); plt.grid(ls='--', alpha=0.3)
    plt.title('Monte Carlo of β')
    plt.legend()
    plt.tight_layout()
    if savepath_prefix:
        plt.savefig(savepath_prefix + "_hist.png", dpi=180, bbox_inches="tight")
    plt.show()

def _plot_sensitivity_bars(sens, savepath_prefix=None):
    labels = [r'$\lambda$', r'$D$', r'$f$']
    vals = np.array([sens["lambda"], sens["D"], sens["f"]]) * 100.0
    plt.figure(figsize=(5.2, 3.4))
    plt.bar(labels, vals)
    plt.ylabel(r'Relative |Δβ| for ±1σ [%]')
    plt.title('One-at-a-time sensitivity')
    plt.grid(axis='y', ls='--', alpha=0.4)
    plt.tight_layout()
    if savepath_prefix:
        plt.savefig(savepath_prefix + "_sens.png", dpi=180, bbox_inches="tight")
    plt.show()

# ----------------------------
# API richiesto
# ----------------------------
def _main(ftag, Nsamp=10, seed = 0):
    """
    Esegue la MC, salva su FITS (path = other_folder()/ (ftag + '.fits')),
    mette i dati in cache, e stampa il sommario su terminale.
    """
    set_data_dir()
    fname = other_folder() / (ftag + '.fits')

    # run MC + sensibilità
    samples = _monte_carlo_beta(n_samples=Nsamp, seed=seed)
    sens, details = _one_at_a_time_sensitivity()
    beta0 = details["beta0"]

    # salva FITS
    _save_beta_mc_fits(str(fname), samples, beta0, sens, details,seed)

    # cache in RAM
    _BETA_MC_CACHE[ftag] = (samples, beta0, sens, details)

    # stampa sommario
    b = samples["beta"]; mu = np.mean(b); sd = np.std(b, ddof=1); lo, hi = np.percentile(b, [2.5, 97.5])
    print(f"[saved] {fname}")
    print(f"Nominal beta: {beta0:.6e}")
    print(f"Monte Carlo (N={len(b)}): mean={mu:.6e}, std={sd:.6e} ({sd/mu*100:.3f}%), 95% CI=[{lo:.6e},{hi:.6e}]")
    print("One-at-a-time sensitivity (±1σ): |Δβ|/β0")
    print(f"  lambda: {sens['lambda']*100:.3f}%   β(-σ), β(+σ) = {details['beta_lambda_pm'][0]:.6e}, {details['beta_lambda_pm'][1]:.6e}")
    print(f"  D     : {sens['D']*100:.3f}%   β(-σ), β(+σ) = {details['beta_D_pm'][0]:.6e}, {details['beta_D_pm'][1]:.6e}")
    print(f"  f     : {sens['f']*100:.3f}%   β(-σ), β(+σ) = {details['beta_f_pm'][0]:.6e}, {details['beta_f_pm'][1]:.6e}")


def load_beta_mc_data(ftag):
    """
    Carica dal FITS e mette in cache. Ritorna (samples, beta0, sens, details, meta).
    """
    set_data_dir()
    fname = other_folder() / (ftag + '.fits')
    samples, beta0, sens, details, meta = _load_beta_mc_fits(str(fname))
    _BETA_MC_CACHE[ftag] = (samples, beta0, sens, details, meta)
    return samples, beta0, sens, details, meta

def show_beta_data_res(ftag, save_fig = False):
    """
    Mostra i plot separati e ristampa il sommario dal file FITS caricato.
    """
    if ftag not in _BETA_MC_CACHE:
        # prova a caricare da disco
        load_beta_mc_data(ftag)

    entry = _BETA_MC_CACHE[ftag]
    # entry può avere 4 o 5 elementi a seconda di chi l'ha riempita
    if len(entry) == 4:
        samples, beta0, sens, details = entry
        meta = None
    else:
        samples, beta0, sens, details, meta = entry

    # stampa
    if meta is not None:
        print(f"[replay] N_samples={meta.get('N_SAMP', len(samples['beta']))}")
        print(f"[replay] seed={meta.get('SEED', -1)}")

    b = samples["beta"]; mu = np.mean(b); sd = np.std(b, ddof=1); lo, hi = np.percentile(b, [2.5, 97.5])
    print(f"[replay] β0={beta0:.6e} | mean={mu:.6e}, std={sd:.6e} ({sd/mu*100:.3f}%), 95%CI=[{lo:.6e},{hi:.6e}]")
    print("Sensitivities (±1σ):")
    print(f"  lambda: {sens['lambda']*100:.3f}%   β(-σ), β(+σ) = {details['beta_lambda_pm'][0]:.6e}, {details['beta_lambda_pm'][1]:.6e}")
    print(f"  D     : {sens['D']*100:.3f}%   β(-σ), β(+σ) = {details['beta_D_pm'][0]:.6e}, {details['beta_D_pm'][1]:.6e}")
    print(f"  f     : {sens['f']*100:.3f}%   β(-σ), β(+σ) = {details['beta_f_pm'][0]:.6e}, {details['beta_f_pm'][1]:.6e}")

    # path per salvare i plot accanto al FITS
    set_data_dir()
    fname = other_folder() / (ftag + '.fits')
    if save_fig is True:
        saveprefix = str(fname.with_suffix(""))
    else:
        saveprefix  = None

    # plot separati
    _plot_scatter_separati(samples, beta0, savepath_prefix=saveprefix)
    _plot_hist_beta(samples, beta0, savepath_prefix=saveprefix)
    _plot_sensitivity_bars(sens, savepath_prefix=saveprefix)

# ----------------------------
# ESEMPIO DI ENTRY-POINT
# ----------------------------
# def main251009_092000():
#     ftag = '251009_092000'
#     _main(ftag, Nsamp=10)               # esegue MC e salva FITS
#     load_beta_mc_data(ftag)   # ricarica da FITS in cache
#     show_beta_data_res(ftag)  # grafici + riepilogo
#
#
# def main251009_111400():
#     ftag = '251009_111400'
#     _main(ftag, Nsamp=10,seed=0)               # esegue MC e salva FITS
#     load_beta_mc_data(ftag)   # ricarica da FITS in cache
#     show_beta_data_res(ftag, save_fig = False)  # grafici + riepilogo
    
def main251009_112700():
    ftag = '251009_111400'
    _main(ftag, Nsamp=500,seed=0)               # esegue MC e salva FITS
    load_beta_mc_data(ftag)   # ricarica da FITS in cache
    show_beta_data_res(ftag, save_fig = False)  # grafici + riepilogo

def main251009_114200():
    ftag = '251009_114200'
    _main(ftag, Nsamp=1000,seed=0)               # esegue MC e salva FITS
    load_beta_mc_data(ftag)   # ricarica da FITS in cache
    show_beta_data_res(ftag, save_fig = False)  # grafici + riepilogo
    
def main251009_115500():
    ftag = '251009_115500'
    _main(ftag, Nsamp=2000,seed=0)               # esegue MC e salva FITS
    load_beta_mc_data(ftag)   # ricarica da FITS in cache
    show_beta_data_res(ftag, save_fig = False)  # grafici + riepilogo
    
def main251009_122400():
    ftag = '251009_122400'
    _main(ftag, Nsamp=5000,seed=0)               # esegue MC e salva FITS
    load_beta_mc_data(ftag)   # ricarica da FITS in cache
    show_beta_data_res(ftag, save_fig = False)  # grafici + riepilogo
    
def main251009_130100():
    ftag = '251009_130100'
    _main(ftag, Nsamp=8000,seed=0)               # esegue MC e salva FITS
    load_beta_mc_data(ftag)   # ricarica da FITS in cache
    show_beta_data_res(ftag, save_fig = False)  # grafici + riepilogo

def main251009_151200():
    ftag = '251009_151200'
    _main(ftag, Nsamp=10000,seed=0)               # esegue MC e salva FITS
    load_beta_mc_data(ftag)   # ricarica da FITS in cache
    show_beta_data_res(ftag, save_fig = False)  # grafici + riepilogo
    
def main251009_171500():
    import time
    
    seed1 = int(time.time())
    ftag1 = '251009_171500'
    _main(ftag1, Nsamp=10000,seed=seed1)               # esegue MC e salva FITS
    load_beta_mc_data(ftag1)   # ricarica da FITS in cache
    show_beta_data_res(ftag1, save_fig = False)
    
    seed2 = int(time.time())
    ftag2 = '251009_193500'
    _main(ftag2, Nsamp=10000,seed=seed2)               # esegue MC e salva FITS
    load_beta_mc_data(ftag2)   # ricarica da FITS in cache
    show_beta_data_res(ftag2, save_fig = False)