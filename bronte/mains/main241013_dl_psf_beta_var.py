import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# importa la tua classe così com'è nel tuo ambiente
# (assicurati che il PYTHONPATH punti dove si trova la definizione)
from bronte.utils.raw_strehl_ratio_computer import StrehlRatioComputer  # <-- SOSTITUISCI: importa dal tuo progetto

# ----------------------------
# Parametri nominali e incertezze
# ----------------------------
LAM_C  = 632.8e-9      # m
LAM_SIG= 0.2e-9        # m (1-sigma)

D_C    = 1090 * 9.2e-6 # m
D_SIG  = 2.65e-6       # m (assoluto)

F_C    = 250e-3        # m
F_SIG  = 0.01 * F_C    # m (1% relativo)

P_C    = 4.65e-6       # m
# Pixel pitch: assumiamo nessun errore (sigma = 0)
P_SIG  = 0.0

# ----------------------------
# Helper per ricalcolare beta
# ----------------------------
def set_sr_params_and_recompute(sr: StrehlRatioComputer, lam, D, f, p):
    """
    Aggiorna i parametri fisici dell'istanza sr e ricostruisce la PSF DL.
    Ritorna beta = fitted_dl_max_au / total_dl_flux.
    """
    # aggiorna parametri fisici
    sr._wl = float(lam)
    sr._pupil_diameter = float(D)
    sr._telescope_focal_length = float(f)
    sr._ccd_pixel_size = float(p)

    # aggiorna grandezze derivate utilizzate dalla classe
    sr._pixel_scale_in_arcsec = sr._ccd_pixel_size / sr._telescope_focal_length * sr.RAD2ARCSEC
    sr._dl_size_in_arcsec     = sr._wl / sr._pupil_diameter * sr.RAD2ARCSEC
    sr._dl_size_in_pixels     = sr._dl_size_in_arcsec / sr._pixel_scale_in_arcsec

    # ricalcola la PSF DL e il fit Airy
    sr._compute_dl_psf()

    # calcola beta
    beta = float(sr._fitted_dl_max_au / sr._total_dl_flux)
    return beta

def compute_beta_nominal():
    sr = StrehlRatioComputer()
    return set_sr_params_and_recompute(sr, LAM_C, D_C, F_C, P_C)

# ----------------------------
# Analisi Monte Carlo
# ----------------------------
def monte_carlo_beta(n_samples=400, seed=0):
    rng = np.random.default_rng(seed)

    # Campionamento gaussiane (troncature minime per evitare valori non fisici)
    lam = rng.normal(LAM_C, LAM_SIG, size=n_samples)
    D   = rng.normal(D_C,   D_SIG,   size=n_samples)
    f   = rng.normal(F_C,   F_SIG,   size=n_samples)
    p   = np.full(n_samples, P_C)  # nessun errore sul pixel pitch

    # Evita valori non fisici
    lam = np.clip(lam, 1e-9, None)
    D   = np.clip(D,   1e-6, None)
    f   = np.clip(f,   1e-3, None)

    sr = StrehlRatioComputer()
    betas = np.zeros(n_samples, dtype=float)

    for i in range(n_samples):
        betas[i] = set_sr_params_and_recompute(sr, lam[i], D[i], f[i], p[i])
        # opzionale: stampa di progresso
        # if (i+1) % 50 == 0:
        #     print(f"MC {i+1}/{n_samples}")

    samples = {
        "lambda": lam,
        "D": D,
        "f": f,
        "p": p,
        "beta": betas,
    }
    return samples

# ----------------------------
# Sensibilità one-at-a-time (±1σ)
# ----------------------------
def one_at_a_time_sensitivity():
    sr = StrehlRatioComputer()
    beta0 = set_sr_params_and_recompute(sr, LAM_C, D_C, F_C, P_C)

    def delta_rel_beta(param_name, minus, plus):
        sr1 = deepcopy(sr)
        sr2 = deepcopy(sr)
        if param_name == "lambda":
            b_minus = set_sr_params_and_recompute(sr1, minus, D_C, F_C, P_C)
            b_plus  = set_sr_params_and_recompute(sr2, plus,  D_C, F_C, P_C)
        elif param_name == "D":
            b_minus = set_sr_params_and_recompute(sr1, LAM_C, minus, F_C, P_C)
            b_plus  = set_sr_params_and_recompute(sr2, LAM_C, plus,  F_C, P_C)
        elif param_name == "f":
            b_minus = set_sr_params_and_recompute(sr1, LAM_C, D_C, minus, P_C)
            b_plus  = set_sr_params_and_recompute(sr2, LAM_C, D_C, plus,  P_C)
        elif param_name == "p":
            b_minus = set_sr_params_and_recompute(sr1, LAM_C, D_C, F_C, minus)
            b_plus  = set_sr_params_and_recompute(sr2, LAM_C, D_C, F_C, plus)
        else:
            raise ValueError("unknown param")

        # variazione relativa media tra +1σ e -1σ
        d_rel = 0.5 * (abs(b_plus - beta0) + abs(b_minus - beta0)) / beta0
        return float(d_rel), float(b_minus), float(b_plus)

    sens = {}
    sens["lambda"], bL_minus, bL_plus = delta_rel_beta("lambda", LAM_C - LAM_SIG, LAM_C + LAM_SIG)
    sens["D"],      bD_minus, bD_plus = delta_rel_beta("D",      D_C   - D_SIG,   D_C   + D_SIG)
    sens["f"],      bF_minus, bF_plus = delta_rel_beta("f",      F_C   - F_SIG,   F_C   + F_SIG)
    # p ha sigma=0, mettiamo 0 per coerenza
    sens["p"] = 0.0
    beta0_val = set_sr_params_and_recompute(sr, LAM_C, D_C, F_C, P_C)

    details = {
        "beta0": beta0_val,
        "beta_lambda_pm": (bL_minus, bL_plus),
        "beta_D_pm":      (bD_minus, bD_plus),
        "beta_f_pm":      (bF_minus, bF_plus),
    }
    return sens, details

# ----------------------------
# Plotting
# ----------------------------
def make_plots(samples, beta0, sens):
    lam = samples["lambda"]
    D   = samples["D"]
    f   = samples["f"]
    p   = samples["p"]
    b   = samples["beta"]

    # 1) Istogramma beta
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    ax[0].hist(b, bins=30, alpha=0.8, edgecolor='k')
    ax[0].axvline(beta0, color='r', ls='--', label=r'$\beta_0$ nominal')
    ax[0].set_title(r"Monte Carlo of $\beta$")
    ax[0].set_xlabel(r'$\beta = I^{\rm DL}_{\rm peak} / F^{\rm DL}_{\rm tot}$')
    ax[0].set_ylabel('Counts')
    ax[0].legend()

    # 2) Scatter beta vs parametri (lambda, D, f)
    ax_sc = ax[1]
    ax_sc.plot(lam*1e9, b, '.', ms=4, alpha=0.5, label=r'$\lambda$ [nm]')
    ax_sc.plot(D*1e6,   b, '.', ms=4, alpha=0.5, label=r'$D$ [$\mu$m]')
    ax_sc.plot(f*1e3,   b, '.', ms=4, alpha=0.5, label=r'$f$ [mm]')
    ax_sc.set_xlabel('Parameter value (units as in legend)')
    ax_sc.set_ylabel(r'$\beta$')
    ax_sc.set_title(r'Parameter sweeps vs $\beta$ (MC samples)')
    ax_sc.legend(loc='best')

    plt.tight_layout()
    plt.show()

    # 3) Bar chart sensitività relative (±1σ)
    labels = [r'$\lambda$', r'$D$', r'$f$', r'$p$']
    vals   = [sens["lambda"]*100, sens["D"]*100, sens["f"]*100, sens["p"]*100]

    plt.figure(figsize=(6,3.2))
    plt.bar(labels, vals)
    plt.ylabel(r'Relative change in $\beta$ for $\pm 1\sigma$ [%]')
    plt.title('One-at-a-time sensitivity')
    plt.grid(axis='y', ls='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# ----------------------------
# Main
# ----------------------------
def main():
    # nominal
    beta0 = compute_beta_nominal()
    print(f"Nominal parameters:")
    print(f"  lambda = {LAM_C*1e9:.2f} nm")
    print(f"  D      = {D_C*1e6:.2f} um")
    print(f"  f      = {F_C*1e3:.2f} mm")
    print(f"  p      = {P_C*1e6:.2f} um")
    print(f"Nominal beta: {beta0:.6e}\n")

    # Monte Carlo
    samples = monte_carlo_beta(n_samples=400, seed=0)
    b = samples["beta"]
    mu = np.mean(b)
    sd = np.std(b, ddof=1)
    p2p = np.percentile(b, [2.5, 97.5])
    print(f"Monte Carlo results (N={len(b)}):")
    print(f"  mean(beta) = {mu:.6e}")
    print(f"  std(beta)  = {sd:.6e}   ({sd/mu*100:.3f} %)")
    print(f"  95% CI     = [{p2p[0]:.6e}, {p2p[1]:.6e}]")
    print(f"  nominal offset = {(beta0-mu)/mu*100:.3f} % wrt MC mean\n")

    # Sensibilità one-at-a-time
    sens, details = one_at_a_time_sensitivity()
    print("One-at-a-time sensitivity (±1σ): relative |Δβ|/β0 [%]")
    print(f"  lambda: {sens['lambda']*100:.3f} %   beta(-σ), beta(+σ) = {details['beta_lambda_pm'][0]:.6e}, {details['beta_lambda_pm'][1]:.6e}")
    print(f"  D     : {sens['D']*100:.3f} %   beta(-σ), beta(+σ) = {details['beta_D_pm'][0]:.6e}, {details['beta_D_pm'][1]:.6e}")
    print(f"  f     : {sens['f']*100:.3f} %   beta(-σ), beta(+σ) = {details['beta_f_pm'][0]:.6e}, {details['beta_f_pm'][1]:.6e}")
    print(f"  p     : {sens['p']*100:.3f} %   (no uncertainty assumed)\n")

    # Grafici
    make_plots(samples, beta0, sens)

if __name__ == "__main__":
    main()
