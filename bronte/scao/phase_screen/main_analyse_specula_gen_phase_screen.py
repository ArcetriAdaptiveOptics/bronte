#
# import numpy as np
# from bronte.scao.phase_screen.phase_screen_analyser import PhaseScreenAnalyser
#
# def main(ftag):
#
#     psa = PhaseScreenAnalyser(ftag)
#     noll_mode_vector, exp_zc_std, obs_zc_std = psa.display_modal_plot()

import numpy as np
import matplotlib.pyplot as plt
from bronte.scao.phase_screen.phase_screen_analyser import PhaseScreenAnalyser

# ---------- util: Noll -> (n, m) ----------
def noll_to_nm(j):
    """
    Restituisce (n, m) dal numero di Noll j (1-based).
    Qui serve n per raggruppare per grado radiale.
    """
    if j < 1:
        raise ValueError("Noll index j must be >= 1")
    j1 = j - 1
    n = 0
    while j1 > n:
        n += 1
        j1 -= n
    m = -n + 2*j1
    return n, m

def robust_std_from_mad(x, axis=None):
    """Stima sigma robusta da MAD (mediana(|x - mediana(x)|)) / 0.6745."""
    x = np.asarray(x)
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis)
    return mad / 0.6744897501960817  # ~inv CDF(0.75)

# ---------- simulatore ----------
def simulate_modal_std(exp_zc_std, obs_zc_std, noll_mode_vector, n_sims=300, rng=None):
    """
    Genera n_sims vettori simulati delle std modali:
    - stima un trend moltiplicativo per grado radiale in log-spazio
    - aggiunge rumore gaussiano a varianza robusta costante (in log-spazio)
    """
    exp = np.asarray(exp_zc_std, dtype=float)
    obs = np.asarray(obs_zc_std, dtype=float)
    jvec = np.asarray(noll_mode_vector, dtype=int)
    assert exp.shape == obs.shape == jvec.shape

    # Evita problemi di log(0): maschera/correggi eventuali zeri
    eps_floor = 1e-18
    exp_safe = np.clip(exp, eps_floor, None)
    obs_safe = np.clip(obs, eps_floor, None)

    # Lavoro sul log-rapporto (errore moltiplicativo)
    delta = np.log(obs_safe) - np.log(exp_safe)  # delta_j

    # Trend per grado radiale n: mediana per n
    n_all = np.array([noll_to_nm(int(j))[0] for j in jvec])
    n_vals = np.unique(n_all)
    trend_by_n = {}
    for n in n_vals:
        delta_n = delta[n_all == n]
        if delta_n.size > 0:
            trend_by_n[n] = np.median(delta_n)
        else:
            trend_by_n[n] = 0.0

    trend = np.array([trend_by_n[n] for n in n_all])

    # Dispersione robusta globale dei residui
    resid = delta - trend
    sigma_rob = float(robust_std_from_mad(resid))
    # fallback se degenerato
    if not np.isfinite(sigma_rob) or sigma_rob <= 0:
        sigma_rob = np.std(resid) if np.std(resid) > 0 else 1e-3

    # RNG
    rng = np.random.default_rng(None if rng is None else rng)

    # Simulazioni in log-spazio
    sims_log = rng.normal(loc=trend[None, :], scale=sigma_rob, size=(n_sims, len(exp)))
    sims = np.exp(np.log(exp_safe)[None, :] + sims_log)

    # Garanzia positività (già assicurata) e clip numerico
    sims = np.clip(sims, eps_floor, None)

    return sims, trend, sigma_rob

def simulate_modal_std_caseC(exp_zc_std, obs_zc_std, noll_mode_vector, n_sims=300, rng=None):
    """
    Case C (piecewise sigma):
    - Media delle simulazioni = valore osservato per-modo (obs_j).
    - Deviazione standard relativa per-modo:
        * j <= 27  -> sigma_rel = 0.17
        * j > 27   -> sigma_rel = 0.08
    - Rumore moltiplicativo indipendente per modo: sim = obs * (1 + N(0, sigma_rel(j)^2))
    """
    obs  = np.asarray(obs_zc_std, dtype=float)
    jvec = np.asarray(noll_mode_vector, dtype=int)
    assert obs.shape == jvec.shape

    # RNG
    rng = np.random.default_rng(None if rng is None else rng)

    # Sigma relativa per-modo (basata sull'indice di Noll)
    sigma_rel_per_mode = np.full_like(obs, 0.08, dtype=float)
    sigma_rel_per_mode[jvec <= 27] = 0.15

    # Rumore relativo per simulazione e per modo (broadcast su scale diverse)
    noise_rel = rng.normal(loc=0.0, scale=sigma_rel_per_mode[None, :], size=(n_sims, obs.size))

    # Simulazioni centrate su obs con variabilità per-modo
    sims = obs[None, :] * (1.0 + noise_rel)

    # Le std non possono essere negative
    sims = np.clip(sims, 0.0, None)

    # Ritorno: sigma assoluta per-modo e la mappa delle sigma relative

    sigma_abs_per_mode = sigma_rel_per_mode * obs
    sigma_rel_eff = float(np.mean(sigma_rel_per_mode))   # <- scalare
    return sims, sigma_abs_per_mode, sigma_rel_eff





# ---------- plotting ----------
def plot_all_simulations(noll_mode_vector, exp_zc_std, obs_zc_std, sims, title_suffix="A"):
    j = np.asarray(noll_mode_vector, dtype=int)
    exp = np.asarray(exp_zc_std, dtype=float)
    obs = np.asarray(obs_zc_std, dtype=float)

    plt.figure(figsize=(10.5, 5.8))
    # simulazioni: tanti tratti leggeri
    for k in range(sims.shape[0]):
        plt.plot(j, sims[k]/1e-6,'.', linewidth=0.8, alpha=0.12)
        
    mean_sim = sims.mean(axis=0)/1e-6
    std_sim = sims.std(axis=0, ddof=1)/1e-6
    # osservato e teoria evidenziati
    plt.plot(j, obs/1e-6,'.', linewidth=0.8, alpha=0.12)
    plt.plot(j, exp/1e-6, label="Theory", linewidth=2.5, linestyle="--")
    #plt.fill_between(j, mean_sim - std_sim, mean_sim + std_sim, alpha=1)
    plt.plot(j, mean_sim, '-m',linewidth=1.5, label="Mean")
    plt.xlabel("Zernike Noll index j")
    plt.ylabel("Modal coefficent STD [um] rms wf")
    plt.title(f"Modal plot: Case {title_suffix}")
    plt.grid(True, alpha=0.25)
    plt.legend(loc='best', ncol=2)
    plt.tight_layout()


# ---------- metrica di discrepanza ----------
def discrepancy_metrics(exp_zc_std, sims):
    """
    Ritorna bias medio relativo (%) e RMSE relativo (%) fra media simulata e teoria.
    """
    exp = np.asarray(exp_zc_std, dtype=float)
    mean_sim = sims.mean(axis=0)
    # Evita divisioni per zero
    eps_floor = 1e-18
    exp_safe = np.clip(exp, eps_floor, None)
    rel_err = (mean_sim - exp_safe) / exp_safe
    bias_pct = 100.0 * np.mean(rel_err)
    rmse_pct = 100.0 * np.sqrt(np.mean(rel_err**2))
    return bias_pct, rmse_pct

# ---------- main ----------
def main(ftag, n_sims=300, rng_seed=0, make_plots=True, case_sim="A"):
    psa = PhaseScreenAnalyser(ftag)
    noll_mode_vector, exp_zc_std, obs_zc_std = psa.display_modal_plot()
    
    if case_sim=="C":
        sims, trend_log_by_n, sigma_log = simulate_modal_std_caseC(
            exp_zc_std=exp_zc_std,
            obs_zc_std=obs_zc_std,
            noll_mode_vector=noll_mode_vector,
            n_sims=n_sims,
            rng=rng_seed
        )   
    else:
        sims, trend_log_by_n, sigma_log = simulate_modal_std(
            exp_zc_std=exp_zc_std,
            obs_zc_std=obs_zc_std,
            noll_mode_vector=noll_mode_vector,
            n_sims=n_sims,
            rng=rng_seed
        )

    if make_plots:
        plot_all_simulations(noll_mode_vector, exp_zc_std, obs_zc_std, sims, title_suffix=case_sim)
        #plot_mean_with_band(noll_mode_vector, exp_zc_std, sims, obs_zc_std=obs_zc_std, title_suffix=case_sim)
        # Se preferisci scala log per dinamiche ampie, decommenta:
        # for ax in plt.gcf().get_axes(): ax.set_yscale('log')
        plt.show()

    bias_pct, rmse_pct = discrepancy_metrics(exp_zc_std, sims)
    print(f"[Discrepanza media vs teoria] Bias relativo: {bias_pct:+.2f}%   RMSE relativo: {rmse_pct:.2f}%")
    print(f"[Modello] sigma_log (dispersione in log-spazio) = {sigma_log:.4f} (≈ {100*(np.exp(sigma_log)-1):.1f}% di deviazione relativa 1σ)")

    return {
        "noll_mode_vector": noll_mode_vector,
        "exp_zc_std": exp_zc_std,
        "obs_zc_std": obs_zc_std,
        "sims": sims,
        "bias_pct": bias_pct,
        "rmse_pct": rmse_pct,
        "sigma_log": sigma_log
    }
    
def main_250410_135700():
    main('250410_135700')

def main_250410_115000():
    main('250410_115000', case_sim="B")
    
def main_250410_092500():
    main('250410_092500', case_sim="C")
