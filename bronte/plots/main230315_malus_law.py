import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.optimize import curve_fit
from scipy.stats import chi2 as chi2_dist


# === TUO IMPORT ===
from tesi_slm import ghost_measurer

# ------------------------- DATA LOADING -------------------------
def load_ghost_lp_rot():
    fpath = r"D:\phd_slm_edo\old_data\230315\\"
    fname_tag = '230315gm_ang'
    angles = [300, 320, 330, 336, 340, 342, 344, 346, 348, 350, 356, 360, 370]

    agr = ghost_measurer.AnalyzeGhostRatio(angles, fpath, fname_tag)
    # agr.show_ratio()

    angle_vector_deg = np.asarray(agr._rot_angle, float)   # [deg]
    i0_norm2flat = np.asarray(agr._ghost_mean, float)      # 0th / flat(t0)
    i1_norm2flat = np.asarray(agr._mod_mean, float)        # 1st / flat(t0)

    # Per-frame (elimina drift e normalizzazione a t0):
    itot_norm = i0_norm2flat + i1_norm2flat
    eta0 = i0_norm2flat / itot_norm
    eta1 = i1_norm2flat / itot_norm
    return angle_vector_deg, eta0, eta1

# ------------------------- MODELLI -------------------------
def cos2_offset_model(phi_deg, A, theta0_deg, B):
    """eta1 = A * cos^2(phi - theta0) + B  -> modulabile = A"""
    return A * np.cos(np.deg2rad(phi_deg - theta0_deg))**2 + B

def sin2_offset_model(phi_deg, C, theta0_deg, D):
    """eta0 = C * sin^2(phi - theta0) + D  -> non modulabile = D"""
    return C * np.sin(np.deg2rad(phi_deg - theta0_deg))**2 + D

# ------------------------- ANGOLI -------------------------
def project_theta_into_range(theta_deg, angle_vector_deg):
    """Proietta theta (mod 180°) nel range degli angoli misurati (es. 300–370°)."""
    theta = theta_deg % 180.0
    a_min, a_max = float(np.min(angle_vector_deg)), float(np.max(angle_vector_deg))
    center = 0.5 * (a_min + a_max)
    k = int(np.round((center - theta) / 180.0))
    theta_proj = theta + 180.0 * k
    # clamp leggero ai bordi
    span = max(a_max - a_min, 1.0)
    pad = 0.02 * span
    return float(np.clip(theta_proj, a_min - pad, a_max + pad))

# ------------------------- PSF / SOMMA ADU -------------------------
def simulate_gaussian_sum_adu_from_peak(
    roi_size: int,
    fwhm_pix: float,
    peak_adu: float,
    ron_adu: float,
    gain_e_per_adu: float,
    rng: np.random.Generator
) -> float:
    """Simula SUM(ADU) in ROI 50x50 per una PSF gaussiana con picco dato."""
    N = roi_size
    y, x = np.mgrid[0:N, 0:N]
    cx = (N - 1) / 2.0
    cy = (N - 1) / 2.0
    sigma = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    model_adu = peak_adu * np.exp(-(((x - cx)**2 + (y - cy)**2) / (2.0 * sigma**2)))
    model_e = np.clip(model_adu * gain_e_per_adu, 0.0, None)
    noisy_e = rng.poisson(model_e)
    adu_shot = noisy_e / gain_e_per_adu
    adu_noisy = adu_shot + rng.normal(loc=0.0, scale=ron_adu, size=model_adu.shape)
    adu_noisy = np.clip(adu_noisy, 0.0, None)
    return float(np.sum(adu_noisy))

def build_angle_dependent_sum_adu(
    phi_deg: np.ndarray,
    theta0_deg: float,
    roi_size: int = 50,
    fwhm_pix: float = 3.3,
    peak_adu_at_align: float = 1495.0,
    ron_adu: float = 2.4,
    gain_e_per_adu: float = 3.5409,
    epsilon_floor: float = 0.03,
    jitter_rms: float = 0.05,
    seed: int = 0
) -> np.ndarray:
    """S(phi) ~ (epsilon + cos^2(..)) * SUM_max + jitter di sorgente (moltiplicativo)."""
    rng = np.random.default_rng(seed)
    sum_at_align = simulate_gaussian_sum_adu_from_peak(
        roi_size, fwhm_pix, peak_adu_at_align, ron_adu, gain_e_per_adu, rng
    )
    malus = epsilon_floor + (1.0 - epsilon_floor) * np.cos(np.deg2rad(phi_deg - theta0_deg))**2
    S = sum_at_align * malus
    jitter = rng.normal(1.0, jitter_rms, size=len(phi_deg))
    S = np.clip(S * jitter, 1.0, None)
    return S

# ------------------------- ERRORI SU ETA -------------------------
def eta_sigmas_from_counts(
    N0: np.ndarray, N1: np.ndarray,
    roi_size: int, ron_adu: float, gain_e_per_adu: float,
    eta0: np.ndarray, eta1: np.ndarray,
    eta_rel_floor: float = 0.01,  # 1% relativo (over-dispersion)
    eta_abs_floor: float = 0.003  # 0.3% assoluto
):
    """
    Var(Nk) ≈ Nk/gain + Npix*RON^2. Propaga su eta1 = N1/(N0+N1), eta0 = N0/(N0+N1).
    Poi aggiunge floor di incertezza su eta per modellare over-dispersion (ROI, PRNU, speckle).
    """
    Npix = roi_size * roi_size
    var0 = N0 / gain_e_per_adu + Npix * (ron_adu**2)
    var1 = N1 / gain_e_per_adu + Npix * (ron_adu**2)
    S = N0 + N1
    base = ((N0**2) * var1 + (N1**2) * var0) / np.maximum(S**4, 1e-24)
    var_eta1 = np.maximum(base, 1e-16)
    var_eta0 = var_eta1  # due-channel case

    # over-dispersion floors (relativa + assoluta)
    var_eta0 += (eta_rel_floor * eta0)**2 + (eta_abs_floor**2)
    var_eta1 += (eta_rel_floor * eta1)**2 + (eta_abs_floor**2)

    sigma_eta1 = np.sqrt(var_eta1)
    sigma_eta0 = np.sqrt(var_eta0)
    return sigma_eta0, sigma_eta1

# ------------------------- FIT + BOOTSTRAP -------------------------

def gof_stats(y, yhat, sigma, n_params: int):
    """
    Goodness-of-fit per LS pesata con incertezze note (absolute_sigma=True):
      - chi2 = sum(((y - yhat)/sigma)^2)
      - dof = N - k
      - redchi2 = chi2 / dof
      - p_value = Prob(Chi2 >= chi2 | dof)  (sf = survival function)
      - wrms = sqrt(mean(((y - yhat)/sigma)^2))  => atteso ~1 se il modello/σ sono corretti
      - max|z| = massimo valore assoluto dei residui normalizzati
      - AIC, AICc, BIC (approssimati su -2 ln L ≈ chi2, costanti ignorate)
    """
    y = np.asarray(y, float); yhat = np.asarray(yhat, float); sigma = np.asarray(sigma, float)
    z = (y - yhat) / np.maximum(sigma, 1e-24)
    chi2 = float(np.sum(z**2))
    N = y.size
    dof = max(N - n_params, 1)
    redchi2 = chi2 / dof
    p_value = float(chi2_dist.sf(chi2, dof))
    wrms = float(np.sqrt(np.mean(z**2)))
    max_abs_z = float(np.max(np.abs(z)))
    # Info criteria (comparativi su stesso dataset)
    k = n_params
    AIC  = chi2 + 2*k
    AICc = AIC + (2*k*(k+1)) / max(N - k - 1, 1)
    BIC  = chi2 + k * np.log(max(N,1))
    return {
        "chi2": chi2, "dof": dof, "redchi2": redchi2, "p_value": p_value,
        "wrms": wrms, "max_abs_z": max_abs_z, "AIC": AIC, "AICc": AICc, "BIC": BIC
    }


@dataclass
class FitResult:
    p: Tuple[float, float, float]        # (amp, theta0_canonical, offset)
    perr: Tuple[float, float, float]     # 1σ (cov)
    p_ci: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]  # 68% bootstrap
    eta_b: float                         # amp + offset (max del canale)
    eta_b_ci: Tuple[float, float]
    f_unmod: float
    f_unmod_ci: Tuple[float, float]
    theta_projected: float               # theta proiettato nel range
    theta_proj_err: float                # 1σ (uguale alla sigma canonica)
    theta_proj_ci: Tuple[float, float]   # 68% bootstrap proiettato

def _bootstrap_fit(model_fun, phi, y, p0, bounds, n_boot=2000, seed=123, sigma=None):
    rng = np.random.default_rng(seed)
    n = len(phi)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        phi_b = phi[idx]; y_b = y[idx]
        sigma_b = sigma[idx] if sigma is not None else None
        try:
            popt_b, _ = curve_fit(
                model_fun, phi_b, y_b, p0=p0, bounds=bounds,
                sigma=sigma_b, absolute_sigma=(sigma_b is not None), maxfev=20000
            )
            boots.append(popt_b)
        except Exception:
            pass
    boots = np.array(boots)
    def ci(col):
        if boots.size == 0:
            return (np.nan, np.nan)
        lo, hi = np.percentile(boots[:, col], [16, 84])
        return (float(lo), float(hi))
    return boots, (ci(0), ci(1), ci(2))


def fit_generic(phi_deg, y, model_fun, p0, bounds, angle_vec,
                sigma=None, n_boot=2000, seed=123):
    absolute_sigma = sigma is not None
    popt, pcov = curve_fit(model_fun, phi_deg, y, p0=p0, bounds=bounds,
                           sigma=sigma, absolute_sigma=absolute_sigma, maxfev=30000)
    amp, theta_can, off = popt
    perr = np.sqrt(np.diag(pcov)) if (pcov is not None and pcov.size) else np.array([np.nan, np.nan, np.nan])

    # proiezione nel range reale (+ stessa 1σ)
    theta_proj = project_theta_into_range(theta_can, angle_vec)
    theta_proj_err = float(perr[1])  # l'errore 1σ resta invariato con shift di 180k

    # bootstrap + proiezione degli estremi
    boots, ci_raw = _bootstrap_fit(model_fun, phi_deg, y, p0=popt, bounds=bounds, n_boot=n_boot, seed=seed, sigma=sigma)
    amp_ci_raw, theta_ci_raw, off_ci_raw = ci_raw
    theta_proj_ci = (
        project_theta_into_range(theta_ci_raw[0], angle_vec),
        project_theta_into_range(theta_ci_raw[1], angle_vec),
    )

    # metriche derivate
    eta_b = float(np.clip(amp + off, 0.0, 1.0))
    if boots.size:
        eta_b_samples = np.clip(boots[:,0] + boots[:,2], 0.0, 1.0)
        eta_b_ci = (np.percentile(eta_b_samples, 16), np.percentile(eta_b_samples, 84))
    else:
        eta_b_ci = (np.nan, np.nan)

    f_unmod = 1.0 - eta_b
    f_unmod_ci = (1.0 - eta_b_ci[1], 1.0 - eta_b_ci[0]) if not np.isnan(eta_b_ci[0]) else (np.nan, np.nan)

    return FitResult(
        p=(float(amp), float(theta_can % 180.0), float(off)),
        perr=(float(perr[0]), float(perr[1]), float(perr[2])),
        p_ci=(amp_ci_raw, theta_proj_ci, off_ci_raw),
        eta_b=eta_b, eta_b_ci=eta_b_ci,
        f_unmod=f_unmod, f_unmod_ci=f_unmod_ci,
        theta_projected=float(theta_proj),
        theta_proj_err=float(theta_proj_err),
        theta_proj_ci=theta_proj_ci
    )

# ------------------------- MAIN (DUE PASSI) -------------------------
def main(
    roi_size: int = 50,
    gain_e_per_adu: float = 3.5409,
    ron_in_adu: float = 2.4,
    fwhm_pix: float = 3.3,
    peak_adu_at_align: float = 1495.0,
    epsilon_floor: float = 0.03,
    jitter_rms: float = 0.05,
    eta_rel_floor: float = 0.01,
    eta_abs_floor: float = 0.003,
    use_bootstrap: bool = True,
    n_boot: int = 2000,
    seed: int = 0,
    make_plot: bool = True
):
    # 1) dati e quantità per-frame
    phi_deg, eta0, eta1 = load_ghost_lp_rot()

    # 2) PASSO 1: fit non pesato su eta1 per stimare theta
    p0_eta1 = [max(0.05, np.max(eta1) - np.min(eta1)), float(phi_deg[np.argmax(eta1)]), float(np.min(eta1))]
    bnds_eta1 = ([0.0, -np.inf, -0.2], [1.2, np.inf, 0.8])
    res1_pass1 = fit_generic(
        phi_deg, eta1, cos2_offset_model, p0_eta1, bnds_eta1, angle_vec=phi_deg,
        sigma=None, n_boot=(n_boot if use_bootstrap else 0), seed=123
    )
    theta_est = res1_pass1.theta_projected

    # 3) somma ADU S(phi) con Malus+jitter → incertezze realistiche
    S_phi = build_angle_dependent_sum_adu(
        phi_deg, theta_est, roi_size=roi_size, fwhm_pix=fwhm_pix,
        peak_adu_at_align=peak_adu_at_align, ron_adu=ron_in_adu,
        gain_e_per_adu=gain_e_per_adu, epsilon_floor=epsilon_floor,
        jitter_rms=jitter_rms, seed=seed
    )
    N0 = eta0 * S_phi
    N1 = eta1 * S_phi
    sigma_eta0, sigma_eta1 = eta_sigmas_from_counts(
        N0, N1, roi_size, ron_in_adu, gain_e_per_adu, eta0, eta1,
        eta_rel_floor=eta_rel_floor, eta_abs_floor=eta_abs_floor
    )

    # 4) PASSO 2: fit pesati (eta1 e eta0)
    res1 = fit_generic(
        phi_deg, eta1, cos2_offset_model, p0_eta1, bnds_eta1, angle_vec=phi_deg,
        sigma=sigma_eta1, n_boot=(n_boot if use_bootstrap else 0), seed=321
    )
    p0_eta0 = [max(0.05, np.max(eta0) - np.min(eta0)), float(phi_deg[np.argmax(eta0)]), float(np.min(eta0))]
    bnds_eta0 = ([0.0, -np.inf, -0.2], [1.2, np.inf, 0.8])
    res0 = fit_generic(
        phi_deg, eta0, sin2_offset_model, p0_eta0, bnds_eta0, angle_vec=phi_deg,
        sigma=sigma_eta0, n_boot=(n_boot if use_bootstrap else 0), seed=654
    )
    
    # 5) stampa risultati — usa θ proiettato (valore + errore)
    
    
    # --- Goodness-of-fit (pesi = sigma_eta*)
    yhat1 = cos2_offset_model(phi_deg, res1.p[0], res1.theta_projected, res1.p[2])
    yhat0 = sin2_offset_model(phi_deg, res0.p[0], res0.theta_projected, res0.p[2])

    gof1 = gof_stats(eta1, yhat1, sigma_eta1, n_params=3)
    gof0 = gof_stats(eta0, yhat0, sigma_eta0, n_params=3)

    print("\n--- Qualità del fit (η₁) ---")
    print(f"chi2 = {gof1['chi2']:.2f},  dof = {gof1['dof']},  chi2_red = {gof1['redchi2']:.3f},  p = {gof1['p_value']:.3f}")
    print(f"WRMS residui = {gof1['wrms']:.3f}  (atteso ≈ 1 se σ sono corretti),  max|z| = {gof1['max_abs_z']:.2f}")
    print(f"AIC = {gof1['AIC']:.1f},  AICc = {gof1['AICc']:.1f},  BIC = {gof1['BIC']:.1f}")

    print("\n--- Qualità del fit (η₀) ---")
    print(f"chi2 = {gof0['chi2']:.2f},  dof = {gof0['dof']},  chi2_red = {gof0['redchi2']:.3f},  p = {gof0['p_value']:.3f}")
    print(f"WRMS residui = {gof0['wrms']:.3f}  (atteso ≈ 1 se σ sono corretti),  max|z| = {gof0['max_abs_z']:.2f}")
    print(f"AIC = {gof0['AIC']:.1f},  AICc = {gof0['AICc']:.1f},  BIC = {gof0['BIC']:.1f}")

    print("\n=== FIT PESATO η₁(φ) = A cos²(φ-θ₀) + B ===")
    print(f"A (modulabile)   = {res1.p[0]:.4f} ± {res1.perr[0]:.4f}   (68% CI: [{res1.p_ci[0][0]:.4f}, {res1.p_ci[0][1]:.4f}])")
    print(f"B                = {res1.p[2]:.4f} ± {res1.perr[2]:.4f}   (68% CI: [{res1.p_ci[2][0]:.4f}, {res1.p_ci[2][1]:.4f}])")
    print(f"θ₀ proiettato    = {res1.theta_projected:.2f}° ± {res1.theta_proj_err:.2f}°  (68% CI: [{res1.theta_proj_ci[0]:.2f}°, {res1.theta_proj_ci[1]:.2f}°])")
    print(f"η₁ max (A+B)     = {res1.eta_b:.4f}  (68% CI: [{res1.eta_b_ci[0]:.4f}, {res1.eta_b_ci[1]:.4f}])")

    print("\n=== FIT PESATO η₀(φ) = C sin²(φ-θ₀) + D ===")
    print(f"C                = {res0.p[0]:.4f} ± {res0.perr[0]:.4f}   (68% CI: [{res0.p_ci[0][0]:.4f}, {res0.p_ci[0][1]:.4f}])")
    print(f"D (non modulabile)= {res0.p[2]:.4f} ± {res0.perr[2]:.4f}   (68% CI: [{res0.p_ci[2][0]:.4f}, {res0.p_ci[2][1]:.4f}])")
    print(f"θ₀ proiettato    = {res0.theta_projected:.2f}° ± {res0.theta_proj_err:.2f}°  (68% CI: [{res0.theta_proj_ci[0]:.2f}°, {res0.theta_proj_ci[1]:.2f}°])")
    print(f"η₀ max (C+D)     = {res0.eta_b:.4f}  ⇒ modulabile ≈ 1-(C+D) = {1.0-res0.eta_b:.4f}  "
          f"(68% CI: [{1.0-res0.eta_b_ci[1]:.4f}, {1.0-res0.eta_b_ci[0]:.4f}])")

    # 6) plot curato (mostra θ proiettato in legenda)
    if make_plot:
        plt.figure(figsize=(10.4, 5.8))

        # dati con error bar
        plt.errorbar(phi_deg, eta1, yerr=sigma_eta1, fmt='.', ms=6, lw=1, capsize=3,
                     alpha=0.95, label='η₁ data')
        plt.errorbar(phi_deg, eta0, yerr=sigma_eta0, fmt='.', ms=6, lw=1, capsize=3,
                     alpha=0.85, label='η₀ data')

        # curve lisce
        phi_fit = np.linspace(np.min(phi_deg)-5, np.max(phi_deg)+5, 800)
        eta1_fit = cos2_offset_model(phi_fit, res1.p[0], res1.theta_projected, res1.p[2])
        eta0_fit = sin2_offset_model(phi_fit, res0.p[0], res0.theta_projected, res0.p[2])
        plt.plot(phi_fit, eta1_fit, '-', lw=2.2, label='Fit η₁ = A cos²(θ-θ₀)+B')
        plt.plot(phi_fit, eta0_fit, '--', lw=2.2, label='Fit η₀ = C sin²(θ-θ₀)+D')

        # vline su θ proiettato (riporta valore in legenda)
        plt.axvline(res1.theta_projected, color='k', linestyle=':', lw=1.8, alpha=0.9,
                    label=f'θ₀ (η₁) projected = {res1.theta_projected:.2f}°')
        plt.axvline(res0.theta_projected, color='gray', linestyle='--', lw=1.4, alpha=0.9,
                    label=f'θ₀ (η₀) projected = {res0.theta_projected:.2f}°')

        plt.xlabel('Polarizator Angle θ [deg]', fontsize=11)
        plt.ylabel('Intensity ratio in ROI', fontsize=11)
        plt.title('Linear Polarizer orientation vs SLM modulation axis:', fontsize=12)
        plt.grid(True, alpha=0.25)
        leg = plt.legend(frameon=True)
        leg.get_frame().set_alpha(0.92)
        plt.tight_layout()
        plt.show()
        
        
        # 7) plot: separo η1 e η0 in due pannelli + plot del totale
    if make_plot:
        phi_fit = np.linspace(np.min(phi_deg)-5, np.max(phi_deg)+5, 800)
        eta1_fit = cos2_offset_model(phi_fit, res1.p[0], res1.theta_projected, res1.p[2])
        eta0_fit = sin2_offset_model(phi_fit, res0.p[0], res0.theta_projected, res0.p[2])

        # ---- Figura 1: η1 e η0 in due subplot separati ----
        fig, axs = plt.subplots(2, 1, figsize=(10.6, 7.8), sharex=True)

        # Top: η1
        axs[0].errorbar(phi_deg, eta1, yerr=sigma_eta1, fmt='o', ms=6, lw=1, capsize=3,
                        alpha=0.95, label='η₁ data')
        axs[0].plot(phi_fit, eta1_fit, '--',color='g',alpha=0.7, lw=2.2, label='Fit η₁ = A cos²(θ-θ₀)+B')
        axs[0].axvline(res1.theta_projected, color='k', linestyle='--', lw=1.6, alpha=0.9,
                       label=f'θ₀ (η₁) = {res1.theta_projected:.1f}°')
        axs[0].set_ylabel('η₁')
        axs[0].grid(True, alpha=0.25)
        axs[0].legend(frameon=True).get_frame().set_alpha(0.92)

        # Bottom: η0
        axs[1].errorbar(phi_deg, eta0, yerr=sigma_eta0, fmt='s', color='r', ms=6, lw=1, capsize=3,
                        alpha=0.90, label='η₀ data')
        axs[1].plot(phi_fit, eta0_fit, '--', lw=2.2,color='orange', label='Fit η₀ = C sin²(θ-θ₀)+D')
        axs[1].axvline(res0.theta_projected, color='k', linestyle='--', lw=1.6, alpha=0.9,
                       label=f'θ₀ (η₀) = {res0.theta_projected:.1f}°')
        axs[1].set_xlabel('Polarizator Angle θ [deg]')
        axs[1].set_ylabel('η₀')
        axs[1].grid(True, alpha=0.25)
        axs[1].legend(frameon=True).get_frame().set_alpha(0.92)

        fig.suptitle('Linear Polarizer orientation vs SLM modulation axis:', fontsize=12)
        plt.tight_layout()
        plt.show()

        # ---- Figura 2: Intensità totale "misurata" = η0 + η1 ----
        eta_sum = eta0 + eta1
        # Stima conservativa delle incertezze del totale (ignora anticorrelazione -> sovrastima)
        sigma_sum = np.sqrt(sigma_eta0**2 + sigma_eta1**2)

        plt.figure(figsize=(10.4, 4.4))
        plt.errorbar(phi_deg, eta_sum, yerr=sigma_sum, fmt='o', ms=6, lw=1, capsize=3,
                     alpha=0.9, label='η₀ + η₁')
        plt.axhline(1.0, color='k', lw=1.6, linestyle='--')
        plt.xlabel('Polarizator Angle θ [deg]')
        plt.ylabel('Total Normalized Intensity')
        # plt.title('Conservazione dell’intensità normalizzata')
        plt.grid(True, alpha=0.25)
        plt.legend(frameon=True).get_frame().set_alpha(0.92)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main(
        roi_size=25,
        gain_e_per_adu=3.5409,
        ron_in_adu=2.4,
        fwhm_pix=3.3,
        peak_adu_at_align=1000.0,
        epsilon_floor=0.03 ,   # leakage di fondo nel flat (~3%)
        jitter_rms=0.05,      # jitter sorgente 5% RMS
        eta_rel_floor=0.01,   # 1% relativo di over-dispersion su eta
        eta_abs_floor=0.003,  # 0.3% assoluto su eta
        use_bootstrap=True,
        n_boot=2000,
        seed=14741752,
        make_plot=True
    )

