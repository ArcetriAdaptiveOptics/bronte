import numpy as np
import matplotlib.pyplot as plt

def main():
    
    #PARAMETERS OF COMMANDED TILT
    wl=633e-9
    L = 256
    Dpx = 545*2
    #SETTING PHASE STROKE
    phi_w = 2*np.pi
    
    Npoints = 1000
    tilt_c2_vector = np.linspace(1e-6, 80e-6, Npoints)
    Nq = 10000
    eta0_vector = np.zeros(Npoints)
    eta1_vector = np.zeros(Npoints)
    etam1_vector = np.zeros(Npoints)
    eta2_vector = np.zeros(Npoints)
    etam2_vector = np.zeros(Npoints)
    Nstep_vector = np.zeros(Npoints)
    
    q_vect = np.arange(-0.5*Nq, 0.5*Nq+1)
    eta_2d = np.zeros((Npoints, Nq+1))
    
    for idx, amp in enumerate(tilt_c2_vector):
        
        c2 = amp
        wf_ptv = 4*c2
        phase_ptv = 2*np.pi*wf_ptv/wl
        phase_tilt = np.linspace(0, phase_ptv, Dpx)
        N = get_N(phase_tilt, phi_w, L, Dpx)
        Nstep_vector[idx] = N
        
        etam2_vector[idx] = eta(-2, N, phi_w)
        etam1_vector[idx] = eta(-1, N, phi_w)
        eta0_vector[idx] = eta(0, N, phi_w)
        eta1_vector[idx] = eta(1, N, phi_w)
        eta2_vector[idx] = eta(2, N, phi_w)
        
        eta_2d[idx] = eta(q_vect, N, phi_w)
    
    
    ptv_in_nm_vector = tilt_c2_vector*4/1e-9
    plt.figure()
    plt.clf()
    plt.plot(ptv_in_nm_vector, Nstep_vector)
    plt.xlabel('PtV [nm]')
    plt.ylabel('N steps per period')
    
    plt.figure()
    plt.clf()
    plt.title(fr'$\phi_w = {phi_w/np.pi:.2f}\pi, \ L = {L},  \ Dpx = {Dpx}$')
    plt.plot(tilt_c2_vector/1e-9, etam2_vector, '.-', label=r'$\eta_{-2}$')
    plt.plot(tilt_c2_vector/1e-9, etam1_vector, '.-', label=r'$\eta_{-1}$')
    plt.plot(tilt_c2_vector/1e-9, eta0_vector, '.-', label=r'$\eta_0$')
    plt.plot(tilt_c2_vector/1e-9, eta1_vector, '.-', label=r'$\eta_1$')
    plt.plot(tilt_c2_vector/1e-9, eta2_vector, '.-', label=r'$\eta_2$')
    plt.legend(loc='best')
    plt.xlabel('Tilt coefficient c2 nm rms wf')
    plt.ylabel('Diffraction efficiency' + r'$\eta_q$')    
    
   
    tot_en = []
    
    for idx in range(Npoints):
        
        tot_en.append(eta_2d[idx,:].sum())
    
    plt.figure()
    plt.clf()
    plt.plot(tilt_c2_vector, tot_en, '.-')
    plt.ylabel('Itot')
    plt.xlabel('tilt')
    
    return eta_2d


def sum_eta_until_converged(N, phi_w, q0=256, tol=1e-6, qmax=50000):
    """
    Aumenta simmetricamente il range [-Q, Q] finché 1 - sum(eta) < tol.
    Restituisce (Q, sum_eta, deficit).
    """
    Q = q0
    while True:
        q = np.arange(-Q, Q+1)
        s = np.sum(eta(q, N, phi_w, clip=False), dtype=np.longdouble)
        deficit = float(1.0 - s)
        if abs(deficit) < tol or Q >= qmax:
            return Q, float(s), deficit
        Q *= 2



def _sinc_sq_rad(z, tol=1e-12):
    """
    Restituisce (sin z / z)^2 in modo stabile, per scalari o array.
    Usa np.sinc(z/pi) per evitare divisioni esplicite e gestisce NaN/inf.
    """
    z = np.asarray(z, dtype=np.float64)

    # maschera di valori finiti
    finite = np.isfinite(z)

    out = np.zeros_like(z, dtype=np.float64)

    # calcolo sicuro solo sui punti finiti
    s = np.sinc(z[finite] / np.pi)  # nessun warning su z=0
    out[finite] = s * s

    # forza il limite 1 vicino a 0 (override utile per tolleranze molto strette)
    small = finite & (np.abs(z) < tol)
    out[small] = 1.0

    return out.item() if out.shape == () else out


def _interf_ratio_stable(A, N, tol=1e-12):
    """
    ((sin A / A)^2) / ((sin(A/N)/(A/N))^2) = (1/N^2) * (sin A / sin(A/N))^2
    calcolato con np.divide(where=...) e solo np.where (nessuna assegnazione in-place).
    Funziona per scalari o array e non produce RuntimeWarning.
    """
    A = np.asarray(A, dtype=np.float64)   # può essere 0-D
    N = float(N)

    sA = np.sin(A)
    sB = np.sin(A / N)

    # rapporto sicuro r = sA/sB dove |sB|>tol, altrove metti 1.0 (placeholder)
    r = np.empty_like(sA, dtype=np.float64)
    np.divide(sA, sB, out=r, where=(np.abs(sB) > tol))
    r = np.where(np.abs(sB) > tol, r, 1.0)

    # singolarità rimovibile: se anche sA≈0 quando sB≈0 → limite = 1
    both_zero = (np.abs(sB) <= tol) & (np.abs(sA) <= tol)
    ratio = (r**2) / (N**2)
    ratio = np.where(both_zero, 1.0, ratio)

    # pulizia: niente NaN/inf
    ratio = np.where(np.isfinite(ratio), ratio, 0.0)
    return ratio.item() if ratio.shape == () else ratio

def eta(q, N, phi_w, clip=True):
    q = np.asarray(q, dtype=np.float64)
    N = max(1, int(N))
    phi_w = float(phi_w)

    env = _sinc_sq_rad(np.pi * q / N)
    A = 0.5 * phi_w - np.pi * q
    interf = _interf_ratio_stable(A, N)
    val = env * interf
    val = np.where(np.isfinite(val), val, 0.0)
    if clip:
        val = np.clip(val, 0.0, 1.0 + 1e-12)
    return val.item() if val.shape == () else val

# def eta(q, N, phi_w):
#     """
#     Efficienza di diffrazione (0..1). Nessun warning, scalare/array OK.
#     """
#     q = np.asarray(q, dtype=np.float64)
#     N = max(1, int(N))
#     phi_w = float(phi_w)
#
#     env = _sinc_sq_rad(np.pi * q / N)
#     A = 0.5 * phi_w - np.pi * q
#     interf = _interf_ratio_stable(A, N)
#
#     eta_val = env * interf
#     eta_val = np.where(np.isfinite(eta_val), eta_val, 0.0)
#     return np.clip(eta_val, 0.0, 1.0 + 1e-12).item() if np.shape(eta_val) == () else np.clip(eta_val, 0.0, 1.0 + 1e-12)


def get_N(phase_tilt, phi_w=2*np.pi, L=256, Dpx=545*2):
    """
    Restituisce N (>=1) in modo robusto, evitando divisioni per zero e NaN.
    """
    phase_tilt = np.asarray(phase_tilt, dtype=np.float64)
    tilt_ptv = np.ptp(phase_tilt)

    # guardie: se qualcosa non torna, fallback a 1
    if not np.isfinite(tilt_ptv) or tilt_ptv <= 0:
        return 1

    L = int(L)
    Dpx = float(Dpx)
    phi_w = float(phi_w)

    if tilt_ptv < phi_w:
        # quantizzazione su L
        dphi = phi_w / float(L)  # >0
        N = int(min(L, int(tilt_ptv / dphi) + 1))
        return max(1, N)

    # ramo tilt_ptv >= phi_w
    alpha = tilt_ptv / Dpx  # può essere molto piccolo, ma Dpx>0
    if not np.isfinite(alpha) or alpha <= 0:
        return max(1, L)

    Npix_per_Lambda = phi_w / alpha
    if not np.isfinite(Npix_per_Lambda) or Npix_per_Lambda <= 0:
        return max(1, L)

    N = int(min(np.floor(Npix_per_Lambda), L))
    return max(1, N)


