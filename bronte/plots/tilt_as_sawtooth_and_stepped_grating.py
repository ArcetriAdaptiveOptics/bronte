import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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

def main():
    # ----- Parametri comodi da scegliere -----
    _setup_matplotlib_for_thesis()
    wr_factor = 1
    phi_w   = 2*np.pi*wr_factor          # stroke di wrapping (rad)
    Lambda  = 15*wr_factor               # periodo del sawtooth in pixel (es. 12 px)
    N       = 5                # livelli per periodo (pochi gradini)
    x       = np.arange(0, 3*Lambda+1)  # campioniamo 3 periodi
    
    xs = np.linspace(0,3*Lambda, 1000)
    # Slope che produce esattamente quel periodo: alpha = phi_w / Lambda
    alpha         = 2*np.pi / Lambda
    tilt_phase1 = alpha*xs
    tilt_phase2    = alpha * x                  # fase continua comandata
    sawtooth_phase = get_sawtooth(tilt_phase1, phi_w)         # sampling "infinito"
    stepped_phase  = get_stepped_phase(tilt_phase2, phi_w, N) # sampling "finito": N step/periodo

    # ----- Plot 1: sawtooth vs commanded -----
    #plt.figure(figsize=(8,3))
    plt.figure()
    plt.plot(xs, tilt_phase1,       label='Commanded (continuous tilt)', alpha=0.8)
    plt.plot(xs, sawtooth_phase,   label='Wrapped (sawtooth)', linewidth=2)
    plt.xlabel(r'$\xi$'); plt.ylabel('Phase [rad]')
    plt.title(f"Sawtooth: period Λ={Lambda} px, stroke φw={phi_w:.2f} rad")
    plt.grid('--', alpha=0.3); plt.legend(loc='best')
    
    # ---- Custom ticks ----
    # X ticks multipli di Λ
    xticks = np.arange(0, 3*Lambda+1, Lambda)
    xticklabels = []
    for k, val in enumerate(xticks):
        if k == 0:
            xticklabels.append("0")
        elif k == 1:
            xticklabels.append(r'$\Lambda$')
        else:
            xticklabels.append(fr'{k}$\Lambda$')
    plt.xticks(xticks, xticklabels)

    # Y ticks multipli di π
    nmax = int(tilt_phase1.max()/np.pi)
    yticks = np.arange(0, nmax*np.pi+0.1, np.pi)
    yticklabels = []
    for k, val in enumerate(yticks):
        if k == 0:
            yticklabels.append("0")
        elif k == 1:
            yticklabels.append(r'$\pi$')
        else:
            yticklabels.append(fr'{k}$\pi$')
    plt.yticks(yticks, yticklabels)

    # ----- Plot 2: stepped vs sawtooth -----
    #plt.figure(figsize=(8,3))
    plt.figure()
    plt.plot(x, tilt_phase2, '--', label='Commanded (continuous tilt)')
    plt.plot(xs, sawtooth_phase, '--', label ='Wrapped (sawtooth)')
    plt.step(x, stepped_phase,'r', where='post', label=f'Stepped (N={N} steps/period)', linewidth=2)
    plt.xlabel(r'$\xi$'); plt.ylabel('Phase [rad]')
    plt.title(f"Stepped grating: N={N} steps/period  (Δ = φw/N = {phi_w/N:.2f} rad)")
    plt.grid('--', alpha=0.3); plt.legend(loc='best')
    
    # ---- Custom ticks ----
    # X ticks multipli di Λ
    xticks = np.arange(0, 3*Lambda+1, Lambda)
    xticklabels = []
    for k, val in enumerate(xticks):
        if k == 0:
            xticklabels.append("0")
        elif k == 1:
            xticklabels.append(r'$\Lambda$')
        else:
            xticklabels.append(fr'{k}$\Lambda$')
    plt.xticks(xticks, xticklabels)

    # Y ticks multipli di π
    nmax = int(tilt_phase1.max()/np.pi)
    yticks = np.arange(0, nmax*np.pi+0.1, np.pi)
    yticklabels = []
    for k, val in enumerate(yticks):
        if k == 0:
            yticklabels.append("0")
        elif k == 1:
            yticklabels.append(r'$\pi$')
        else:
            yticklabels.append(fr'{k}$\pi$')
    plt.yticks(yticks, yticklabels)
    
    plt.tight_layout()
    plt.show()

def get_sawtooth(tilt_phase, phi_w):
    """Wrapping ideale: sawtooth continuo (sampling 'infinito')."""
    return np.mod(tilt_phase, phi_w)

def get_stepped_phase(tilt_phase, phi_w, N):
    """
    Stepped grating: quantizza il sawtooth in N livelli per periodo.
    - Passo di fase: Δ = φw / N
    - Quantizzazione 'mid-rise' (livello costante dentro ciascun bin).
    """
    delta_phi = phi_w / N
    saw = np.mod(tilt_phase, phi_w)            # sawtooth continuo
    idx = np.floor(saw / delta_phi).astype(int)        # indice di bin 0..N-1
    idx = np.clip(idx, 0, N-1)                 # protezione numerica
    stepped = idx * delta_phi                          # livello quantizzato
    return stepped
