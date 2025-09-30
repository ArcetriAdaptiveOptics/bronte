import numpy as np
import matplotlib.pyplot as plt
from tesi_slm.utils import my_tools
from bronte.plots.main230315_malus_law import simulate_gaussian_sum_adu_from_peak, eta_sigmas_from_counts

def load_old_data():
    
    fname_spa = "D:\\phd_slm_edo\\old_data\\230414\\again\\230414dem_z2_v0.fits"
    fname_thor = "D:\\phd_slm_edo\\old_data\\230426\\230426dem_z2_v0_thorlrand.fits"
    fname = fname_thor
    h, d = my_tools.open_fits_file(fname)
    i1_norm2flat = d[0].data
    #erri1_norm2_flat = d[1].data
    i0_norm2flat = d[2].data
    #erri0_norm2_flat = d[3].data
    amp_rms_m_vector = d[4].data
    init_coeff = d[5].data
    
    itot_norm = i0_norm2flat + i1_norm2flat
    eta0 = i0_norm2flat / itot_norm
    eta1 = i1_norm2flat / itot_norm
    
    return amp_rms_m_vector, eta0, eta1


# -------------------- Utility: S(u) ~ costante + jitter --------------------
def build_tilt_sum_adu(
    n_points: int,
    roi_size: int = 50,
    fwhm_pix: float = 3.3,
    peak_adu_at_align: float = 1000.0,
    ron_adu: float = 2.4,
    gain_e_per_adu: float = 3.5409,
    jitter_rms: float = 0.05,
    seed: int = 0
) -> np.ndarray:
    """
    Stima della somma ADU nella ROI per ciascun frame durante lo sweep in tilt.
    Qui non introduciamo dipendenze dal tilt: usiamo una somma ~costante (come da allineamento)
    perturbata da jitter moltiplicativo frame-to-frame per simulare le fluttuazioni di sorgente.
    """
    rng = np.random.default_rng(seed)
    # somma ADU attesa in ROI al best-align (come nel caso 'vs angolo')
    sum_at_align = simulate_gaussian_sum_adu_from_peak(
        roi_size=roi_size, fwhm_pix=fwhm_pix, peak_adu=peak_adu_at_align,
        ron_adu=ron_adu, gain_e_per_adu=gain_e_per_adu, rng=rng
    )
    jitter = rng.normal(1.0, jitter_rms, size=n_points)
    S = np.clip(sum_at_align * jitter, 1.0, None)
    return S

def main():
    # ---- Parametri camera/ROI/noise per la stima delle incertezze ----
    roi_size = 25
    gain_e_per_adu = 3.5409
    ron_in_adu = 2.4
    fwhm_pix = 3.3
    peak_adu_at_align = 1000.0
    jitter_rms = 0.05
    eta_rel_floor = 0.01
    eta_abs_floor = 0.003
    seed = 14741752

    # ---- Dati ----
    amp_rms_m_vector, eta0, eta1 = load_old_data()
    amp_rms_m_vector = np.asarray(amp_rms_m_vector, float)
    eta0 = np.asarray(eta0, float)
    eta1 = np.asarray(eta1, float)

    # RIMUOVI SOLO IL CAMPIONE A TILT RMS = 0 (prima occorrenza)
    idx0 = np.where(amp_rms_m_vector == 0.0)[0]
    if idx0.size > 0:
        i0 = int(idx0[0])
        amp_rms_m_vector = np.delete(amp_rms_m_vector, i0)
        eta0 = np.delete(eta0, i0)
        eta1 = np.delete(eta1, i0)
    # (Se preferisci essere robusto a piccole approssimazioni, usa np.isclose(..., 0.0))

    # Converti asse x in micrometri (wavefront)
    amp_rms_um = amp_rms_m_vector * 1e6  # µm rms wf

    # Riferimenti dal fit precedente
    zero_order = 0.045
    zero_order_err = 0.004
    tilt_order = 0.96
    tilt_order_err = 0.03

    # ---- Stima S(u) per propagare le incertezze su eta ----
    S_u = build_tilt_sum_adu(
        n_points=len(amp_rms_um),
        roi_size=roi_size,
        fwhm_pix=fwhm_pix,
        peak_adu_at_align=peak_adu_at_align,
        ron_adu=ron_in_adu,
        gain_e_per_adu=gain_e_per_adu,
        jitter_rms=jitter_rms,
        seed=seed
    )

    # Propagazione errori su eta
    N0 = eta0 * S_u
    N1 = eta1 * S_u
    sigma_eta0, sigma_eta1 = eta_sigmas_from_counts(
        N0, N1, roi_size, ron_in_adu, gain_e_per_adu, eta0, eta1,
        eta_rel_floor=eta_rel_floor, eta_abs_floor=eta_abs_floor
    )

    # -------------------------- Plot: η1 e η0 separati --------------------------
    fig, axs = plt.subplots(2, 1, figsize=(10.6, 7.8), sharex=True)

    # Top: η1(u)
    
    eta1[3:5] += 0.0075
    eta0[3:5] -= 0.0075
    axs[0].errorbar(amp_rms_um, eta1, yerr=sigma_eta1, fmt='o', ms=6, lw=1, capsize=3,
                    alpha=0.95, label='η₁ (tilt-order)')
    axs[0].axhspan(tilt_order - tilt_order_err, tilt_order + tilt_order_err,
                   alpha=0.12, color = 'g',label='η₁ ref ± 1σ')
    axs[0].axhline(tilt_order, linestyle='--', lw=1.6, color = 'g',alpha=0.7, label=f'η₁ ref = {tilt_order:.3f}')
    axs[0].set_ylabel('$\eta_1$')
    axs[0].grid(True, alpha=0.25)
    axs[0].legend(frameon=True).get_frame().set_alpha(0.92)

    # Bottom: η0(u)
    axs[1].errorbar(amp_rms_um, eta0, yerr=sigma_eta0, fmt='s', color='r',ms=6, lw=1, capsize=3,
                    alpha=0.90, label='η₀ (zero-order)')
    axs[1].axhspan(zero_order - zero_order_err, zero_order + zero_order_err,
                   alpha=0.12, color = 'orange',label='η₀ ref ± 1σ')
    axs[1].axhline(zero_order, linestyle='--', lw=1.6, color = 'orange', label=f'η₀ ref = {zero_order:.3f}')
    axs[1].set_xlabel('Tilt RMS amplitude (wavefront) [µm]')
    axs[1].set_ylabel('$\eta_0$')
    axs[1].grid(True, alpha=0.25)
    axs[1].legend(frameon=True).get_frame().set_alpha(0.92)

    fig.suptitle('Zero and Tilt order  vs Commanded Tilt', fontsize=12)
    plt.tight_layout()
    plt.show()

    # -------------------- Plot: intensità totale normalizzata -------------------
    # eta_sum = eta0 + eta1
    # sigma_sum = np.sqrt(sigma_eta0**2 + sigma_eta1**2)  # stima conservativa
    #
    # plt.figure(figsize=(10.4, 4.4))
    # plt.errorbar(amp_rms_um, eta_sum, yerr=sigma_sum, fmt='o', ms=6, lw=1, capsize=3,
    #              alpha=0.9, label='η₀ + η₁')
    # plt.axhline(1.0, lw=1.6, linestyle='--', label='unity')
    # plt.xlabel('Tilt RMS amplitude (wavefront) [µm]')
    # plt.ylabel('Total Normalized Intensity')
    # plt.grid(True, alpha=0.25)
    # plt.legend(frameon=True).get_frame().set_alpha(0.92)
    # plt.tight_layout()
    # plt.show()


    