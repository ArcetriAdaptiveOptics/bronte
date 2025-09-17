import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from matplotlib.ticker import LogLocator, NullFormatter
from arte.atmo.von_karman_spatial_covariance_calculator import VonKarmanSpatialCovariance


def main():
    
    #PARAMETERS OF COMMANDED TILT
    wl=633e-9
    L = 256
    Dpx = 545*2
    #SETTING PHASE STROKE
    phi_w = 2*np.pi
    
    Npoints = 1000
    tilt_c2_vector = np.linspace(1e-6, 100e-6, Npoints)
    Nq = 1000
    eta0_vector = np.zeros(Npoints)
    eta1_vector = np.zeros(Npoints)
    etam1_vector = np.zeros(Npoints)
    eta2_vector = np.zeros(Npoints)
    etam2_vector = np.zeros(Npoints)
    Nstep_vector = np.zeros(Npoints)
    Npixel_per_lambda_vector = np.zeros(Npoints)
    
    q_vect = np.arange(-0.5*Nq, 0.5*Nq+1)
    eta_2d = np.zeros((Npoints, Nq+1))
    
    for idx, amp in enumerate(tilt_c2_vector):
        
        c2 = amp
        wf_ptv = 4*c2
        phase_ptv = 2*np.pi*wf_ptv/wl
        phase_tilt = np.linspace(0, phase_ptv, Dpx)
        N = get_N(phase_tilt, phi_w, L, Dpx)
        Npxpl = get_Npix_per_Lambda(phase_tilt, phi_w, L, Dpx)
        Nstep_vector[idx] = N
        Npixel_per_lambda_vector[idx] = Npxpl
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
    #plt.plot(ptv_in_nm_vector, Npixel_per_lambda_vector)
    plt.xlabel('PtV [nm]')
    plt.ylabel('N steps per period')
    
    plt.figure()
    plt.clf()
    plt.title(fr'$\phi_w = {phi_w/np.pi:.2f}\pi, \ L = {L},  \ Dpx = {Dpx}$')
    plt.plot(tilt_c2_vector/1e-9, etam2_vector, '-', label=r'$\eta_{-2}$')
    plt.plot(tilt_c2_vector/1e-9, etam1_vector, '-', label=r'$\eta_{-1}$')
    plt.plot(tilt_c2_vector/1e-9, eta0_vector, '-', label=r'$\eta_0$')
    plt.plot(tilt_c2_vector/1e-9, eta1_vector, '-', label=r'$\eta_1$')
    plt.plot(tilt_c2_vector/1e-9, eta2_vector, '-', label=r'$\eta_2$')
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
    
    
    plt.figure()
    plt.clf()
    
    idx_q0 = np.where(q_vect==0)[0][0]
    Norder2inspect = 7
    idxq_inspect_vect = np.arange(idx_q0-Norder2inspect, idx_q0+Norder2inspect+1)
    q_insected = idxq_inspect_vect-idx_q0
    for idx in idxq_inspect_vect:
        
        qq = idx - idx_q0
        plt.plot(tilt_c2_vector/1e-9, eta_2d[:,idx], label=f'{qq}')
    
    plt.xlabel('tilt c2 nm')
    plt.ylabel('eta_q')
    plt.legend(loc='best')
    
    plt.figure()
    plt.clf()
    plt.imshow(eta_2d.T, norm=LogNorm(vmin=1e-5,vmax=1))
    #plt.imshow(np.log10(eta_2d.T), vmin=-10, vmax = 0)
    plt.colorbar(label='eta_q')
    yticks = np.arange(0, len(q_vect), 20)
    yticklabels = q_vect[yticks]
    plt.yticks(yticks, yticklabels)  
    plt.ylabel('Order q')  
    
    
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

def eta_with_pix(q, N, phi_w, a, Lambda, clip=True):
    q = np.asarray(q, dtype=np.float64)
    N = max(1, int(N))
    phi_w = float(phi_w)
    
    env_pix = _sinc_sq_rad(np.pi * q * a /Lambda)
    env = _sinc_sq_rad(np.pi * q / N)
    A = 0.5 * phi_w - np.pi * q
    interf = _interf_ratio_stable(A, N)
    val = env_pix * env * interf
    val = np.where(np.isfinite(val), val, 0.0)
    if clip:
        val = np.clip(val, 0.0, 1.0 + 1e-12)
    return val.item() if val.shape == () else val


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

def get_Npix_per_Lambda(phase_tilt, phi_w=2*np.pi, L=256, Dpx=545*2):
    
    
    phase_tilt = np.asarray(phase_tilt, dtype=np.float64)
    tilt_ptv = np.ptp(phase_tilt)


    L = int(L)
    Dpx = float(Dpx)
    phi_w = float(phi_w)

    if tilt_ptv < phi_w:
        
        return Dpx

    # ramo tilt_ptv >= phi_w
    alpha = tilt_ptv / Dpx  

    Npix_per_Lambda = phi_w / alpha
    if not np.isfinite(Npix_per_Lambda) or Npix_per_Lambda <= 0:
        return max(1, L)

    return np.floor(Npix_per_Lambda)

def get_mean_coeff(Dtel = 8.2, r0=0.15, L0=25, wl=500e-9):
    
    
    Nmodes=200
    noll_index_vector = np.arange(2,Nmodes+2)
    scale = (Dtel/r0)**(5./3)
    L0norm = L0/Dtel
    vk_var = np.zeros(Nmodes)
    
    for idx, j in enumerate(noll_index_vector):
        vk = VonKarmanSpatialCovariance(int(j),int(j), L0norm)
        vk_var[idx] = vk.get_covariance()
    
    vk_var_in_rad2 = 4*np.pi**2 * vk_var * scale
    vk_cj_in_m = np.sqrt(vk_var_in_rad2)*wl/(2*np.pi)
    return vk_cj_in_m

def main2():
    #PARAMETERS OF COMMANDED TILT
    wl=633e-9
    L = 256
    Dpx = 545*2
    a = 9e-6
    d = 9.2e-6
    #SETTING PHASE STROKE
    phi_w = 2*np.pi
    
    Npoints = 1000
    tilt_c2_vector = np.linspace(1e-6, 100e-6, Npoints)
    Nq = 1000
    Nstep_vector = np.zeros(Npoints)
    Npixel_per_lambda_vector = np.zeros(Npoints)
    
    q_vect = np.arange(-0.5*Nq, 0.5*Nq+1)
    eta_2d = np.zeros((Npoints, Nq+1))
    eta_2d_with_px = np.zeros((Npoints, Nq+1))
    env_pix = np.zeros((Npoints, Nq+1))
    for idx, amp in enumerate(tilt_c2_vector):
        
        c2 = amp
        wf_ptv = 4*c2
        phase_ptv = 2*np.pi*wf_ptv/wl
        phase_tilt = np.linspace(0, phase_ptv, Dpx)
        N = get_N(phase_tilt, phi_w, L, Dpx)
        Npxpl = get_Npix_per_Lambda(phase_tilt, phi_w, L, Dpx)
        Nstep_vector[idx] = N
        Npixel_per_lambda_vector[idx] = Npxpl
        
        env_pix[idx] = _sinc_sq_rad(np.pi * q_vect * a /(Npxpl*d))
        eta_2d[idx] = eta(q_vect, N, phi_w)
        eta_2d_with_px[idx] = eta_with_pix(q_vect, N, phi_w, a, Npxpl*d)
        
    ptv_vector = tilt_c2_vector*4
    return eta_2d, eta_2d_with_px, ptv_vector, env_pix

def pixel_envelope(q, Lambda_px, FF):
    # E_q = sinc^2(pi*FF * q / Lambda), con sinc(x)=sin(x)/x
    x = np.pi * FF * q / np.maximum(Lambda_px, 1e-12)
    # np.sinc usa sinc(pi x), quindi convertiamo: sinc(x) = np.sinc(x/np.pi)
    return np.sinc(x/np.pi)**2

def show_pixel_effect_q1(FF=0.95):
    # Parametri (come nei tuoi plot)
    wl=633e-9; L=256; Dpx=545*2; phi_w=2*np.pi
    Npoints=1200
    tilt_c2 = np.linspace(1e-6, 120e-6, Npoints)
    ptv_um = tilt_c2*4/1e-6

    eta1_raw = np.zeros(Npoints)
    eta1_pix = np.zeros(Npoints)
    Lambda_px_vec = np.zeros(Npoints)

    for i, c2 in enumerate(tilt_c2):
        wf_ptv = 4*c2
        phase_ptv = 2*np.pi*wf_ptv/wl
        phase_tilt = np.linspace(0, phase_ptv, Dpx)
        # tua funzione:
        Npx_per_Lambda = get_Npix_per_Lambda(phase_tilt, phi_w, L, Dpx)  # = Λ in pixel
        Lambda_px_vec[i] = Npx_per_Lambda

        # eta "ideale" (senza pixel) dal tuo modello:
        # qui uso N = get_N(...) se la tua eta(q,N,phi_w) lo richiede:
        N = get_N(phase_tilt, phi_w, L, Dpx)
        eta1_raw[i] = eta(1, N, phi_w)

        # applico inviluppo del pixel:
        E1 = pixel_envelope(q=1, Lambda_px=Npx_per_Lambda, FF=FF)
        eta1_pix[i] = eta1_raw[i] * E1

    fig, ax = plt.subplots(figsize=(8.8,5))
    ax.plot(ptv_um, eta1_raw, lw=2.2, label=r'$\eta_1$ (no pixel)')
    ax.plot(ptv_um, eta1_pix, lw=2.2, ls='--', label=fr'$\eta_1$ with pixel (FF={FF:.2f})')
    ax.set_xlabel(r'Tilt PtV [$\mu$m wf]')
    ax.set_ylabel(r'$\eta_1$')
    ax.set_ylim(-0.02, 1.02); ax.grid(True, ls='--', alpha=0.35)
    ax.legend(loc='best')
    ax2 = ax.twinx()
    ax2.plot(ptv_um, Lambda_px_vec, lw=1.2, color='tab:gray', alpha=0.6, label=r'$\Lambda$ [px/period]')
    ax2.set_ylabel(r'$\Lambda$ [pixel/period]')
    fig.tight_layout(); plt.show()
    


# --- Raw plots ---

def show_Nstep_plots():
    
    #PARAMETERS OF COMMANDED TILT
    wl=633e-9
    L = 256
    Dpx = 545*2
    #SETTING PHASE STROKE
    phi_w = 2*np.pi
    
    Npoints = 30000
    tilt_c2_vector = np.linspace(0, 80e-6, Npoints)


    Nstep_vector = np.zeros(Npoints)
    Npixel_per_lambda_vector = np.zeros(Npoints)

    for idx, amp in enumerate(tilt_c2_vector):
        
        c2 = amp
        wf_ptv = 4*c2
        phase_ptv = 2*np.pi*wf_ptv/wl
        phase_tilt = np.linspace(0, phase_ptv, Dpx)
        N = get_N(phase_tilt, phi_w, L, Dpx)
        Npxpl = get_Npix_per_Lambda(phase_tilt, phi_w, L, Dpx)
        Nstep_vector[idx] = N
        Npixel_per_lambda_vector[idx] = Npxpl
       
    
    ptv_in_um_vector = tilt_c2_vector*4/1e-6
    plt.figure()
    plt.clf()
    plt.plot(ptv_in_um_vector, Nstep_vector, label = r'$N$')
    plt.plot(ptv_in_um_vector, Npixel_per_lambda_vector, '--', label=r'$N_{\Lambda}$', alpha=0.8)
    plt.vlines(wl/1e-6, 300, L, colors='k', linestyles='--', lw=1.6, label=fr'$PtV \ = \ {wl/1e-6:.3f} \ \mu m$', alpha=0.5)
    plt.hlines(256, -5, ptv_in_um_vector.max(), colors='g', linestyles='--',  lw=1.6 ,label='L=256', alpha=0.5)
    plt.xlabel('Tilt PtV [um wf]')
    plt.ylabel('N steps per period ' + '$\Lambda$')
    plt.title(fr'$\phi_w = {phi_w/np.pi:.0f}\pi, \ L = {L},  \ Dpx = {Dpx}, \ \lambda = {wl/1e-6:.3f} \ \mu m$')
    plt.legend(loc='best')
    plt.grid(which='major', linestyle='--', alpha=0.35)
    plt.grid(which='minor', linestyle=':', alpha=0.25)

    plt.xlim(-5, 100)
    plt.ylim(0, 300)

def show_eta_vs_ptv():
    #PARAMETERS OF COMMANDED TILT
    wl=633e-9
    L = 256
    Dpx = 545*2
    a=9e-6
    d=9.2e-6
    #SETTING PHASE STROKE
    phi_w = 2*np.pi
    
    Npoints = 1000
    tilt_c2_vector = np.linspace(1e-6, 80e-6, Npoints)
 
    eta0_vector = np.zeros(Npoints)
    eta1_vector = np.zeros(Npoints)
    
    eta0_vect050 = np.zeros(Npoints)
    eta1_vect050 = np.zeros(Npoints)
    eta0_vect090 = np.zeros(Npoints)
    eta1_vect090 = np.zeros(Npoints)
    eta0_vect095 = np.zeros(Npoints)
    eta1_vect095 = np.zeros(Npoints)

    Nstep_vector = np.zeros(Npoints)
    Npixel_per_lambda_vector = np.zeros(Npoints)
    
    for idx, amp in enumerate(tilt_c2_vector):
        
        c2 = amp
        wf_ptv = 4*c2
        phase_ptv = 2*np.pi*wf_ptv/wl
        phase_tilt = np.linspace(0, phase_ptv, Dpx)
        N = get_N(phase_tilt, phi_w, L, Dpx)
        Npxpl = get_Npix_per_Lambda(phase_tilt, phi_w, L, Dpx)
        Nstep_vector[idx] = N
        Npixel_per_lambda_vector[idx] = Npxpl

        eta0_vector[idx] = eta(0, N, phi_w)
        eta1_vector[idx] = eta(1, N, phi_w)
        
        eta0_vect050[idx] = eta(0, N, phi_w*0.5)
        eta1_vect050[idx] = eta(1, N, phi_w*0.5)
        
        eta0_vect090[idx] = eta(0, N, phi_w*0.9)
        eta1_vect090[idx] = eta(1, N, phi_w*0.9)
        
        eta0_vect095[idx] = eta(0, N, phi_w*0.95)
        eta1_vect095[idx] = eta(1, N, phi_w*0.95)
        
    
    ptv_in_um_vector = tilt_c2_vector*4/1e-6
    plt.figure()
    plt.clf()
    plt.title(fr'$\phi_w = {phi_w/np.pi:.2f}\pi, \ L = {L},  \ Dpx = {Dpx}$')

    plt.plot(ptv_in_um_vector, eta0_vector, '--', label=r'$\eta_0(\phi_w/2\pi=1)$')
    plt.plot(ptv_in_um_vector, eta1_vector, '-', label=r'$\eta_1(\phi_w/2\pi=1)$')
    
    plt.plot(ptv_in_um_vector, eta0_vect090, '--', label=r'$\eta_0(\phi_w/2\pi=0.9)$')
    plt.plot(ptv_in_um_vector, eta1_vect090, '-', label=r'$\eta_1(\phi_w/2\pi=0.9)$')
    
    plt.plot(ptv_in_um_vector, eta0_vect050, '--', label=r'$\eta_0(\phi_w/2\pi=0.5)$')
    plt.plot(ptv_in_um_vector, eta1_vect050, '-', label=r'$\eta_1(\phi_w/2\pi=0.5)$')
    

    plt.legend(loc='best')
    plt.xlabel('Tilt PtV '+'$\mu m$' +' wf')
    plt.ylabel(r'$\eta_q$')    
    plt.grid('--', alpha=0.4)
    
def show_eta_map():
    #PARAMETERS OF COMMANDED TILT
    wl=633e-9
    L = 256
    Dpx = 545*2
    #SETTING PHASE STROKE
    phi_w = 2*np.pi
    
    Npoints = 1000
    tilt_c2_vector = np.linspace(1e-6, 80e-6, Npoints)
    Nq = 1000

    
    q_vect = np.arange(-0.5*Nq, 0.5*Nq+1)
    eta_2d = np.zeros((Npoints, Nq+1))
    
    for idx, amp in enumerate(tilt_c2_vector):
        
        c2 = amp
        wf_ptv = 4*c2
        phase_ptv = 2*np.pi*wf_ptv/wl
        phase_tilt = np.linspace(0, phase_ptv, Dpx)
        N = get_N(phase_tilt, phi_w, L, Dpx)
        eta_2d[idx] = eta(q_vect, N, phi_w)
    
    tot_en = []
    
    for idx in range(Npoints):
        
        tot_en.append(eta_2d[idx,:].sum())
    
    plt.figure()
    plt.clf()
    plt.plot(tilt_c2_vector, tot_en, '.-')
    plt.ylabel('Itot')
    plt.xlabel('tilt')
    
    
    plt.figure()
    plt.clf()
    
    idx_q0 = np.where(q_vect==0)[0][0]
    Norder2inspect = 7
    idxq_inspect_vect = np.arange(idx_q0-Norder2inspect, idx_q0+Norder2inspect+1)

    for idx in idxq_inspect_vect:
        
        qq = idx - idx_q0
        plt.plot(tilt_c2_vector/1e-9, eta_2d[:,idx], label=f'{qq}')
    
    plt.xlabel('tilt c2 nm')
    plt.ylabel('eta_q')
    plt.legend(loc='best')
    
    plt.figure()
    plt.clf()
    plt.imshow(eta_2d.T, norm=LogNorm(vmin=1e-5,vmax=1))
    #plt.imshow(np.log10(eta_2d.T), vmin=-10, vmax = 0)
    plt.colorbar(label='eta_q')
    yticks = np.arange(0, len(q_vect), 20)
    yticklabels = q_vect[yticks]
    plt.yticks(yticks, yticklabels)  
    plt.ylabel('Order q')  
    
# --- Cooler plots ---

def show_Nstep_plots2():
    
    plt.rcParams.update({
    "figure.dpi": 140,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.35,
    })
    # PARAMETERS OF COMMANDED TILT
    wl = 633e-9
    L = 256
    Dpx = 545*2
    # SETTING PHASE STROKE
    phi_w = 2*np.pi

    Npoints = 30_000
    tilt_c2_vector = np.linspace(0, 80e-6, Npoints)

    Nstep_vector = np.zeros(Npoints)
    Npixel_per_lambda_vector = np.zeros(Npoints)

    for idx, amp in enumerate(tilt_c2_vector):
        c2 = amp
        wf_ptv = 4 * c2
        phase_ptv = 2*np.pi * wf_ptv / wl
        phase_tilt = np.linspace(0, phase_ptv, Dpx)
        N = get_N(phase_tilt, phi_w, L, Dpx)
        Npxpl = get_Npix_per_Lambda(phase_tilt, phi_w, L, Dpx)
        Nstep_vector[idx] = N
        Npixel_per_lambda_vector[idx] = Npxpl

    # x in micrometri PtV (wf)
    ptv_in_um_vector = tilt_c2_vector * 4 / 1e-6

    # --- FIGURA ---
    fig, ax = plt.subplots( constrained_layout=True)

    # linee principali
    h1, = ax.plot(ptv_in_um_vector, Nstep_vector, lw=2.2, label=r'$N$')
    h2, = ax.plot(ptv_in_um_vector, Npixel_per_lambda_vector, lw=2.2, ls='--', label=r'$N_{\Lambda}$')

    # linee di riferimento
    ax.axvline(wl/1e-6, color='k', lw=1.6, ls='--', alpha=0.5,
               label=fr'$PtV = \lambda$')
    ax.axhline(L, color='tab:green', lw=1.6, ls='--', alpha=0.5,
               label=fr'$L = {L}$')

    # annotazione elegante per PtV=λ
    ax.annotate('PtV = λ',
                xy=(wl/1e-6, np.interp(wl/1e-6, ptv_in_um_vector, Nstep_vector)),
                xytext=(10, 25),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', relpos=(0.5, 0.5), alpha=0.6),
                fontsize=11)

    # etichette e titolo
    ax.set_xlabel(r'Tilt PtV [$\mu$m wf]')
    ax.set_ylabel(r'$N$ steps per period $\Lambda$')
    ax.set_title(fr'$\phi_w = {phi_w/np.pi:.0f}\pi,\ \ L = {L},\ \ D_{{px}} = {Dpx}, \ \lambda = {wl/1e-6:.3f} \ \mu m$')

    # ticks e griglia (major+minor)
    ax.xaxis.set_major_locator(MaxNLocator(8))
    ax.yaxis.set_major_locator(MaxNLocator(8))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which='major', linestyle='--', alpha=0.35)
    ax.grid(which='minor', linestyle=':', alpha=0.25)

    # limiti “intelligenti”
    x_max = min(100, ptv_in_um_vector.max())
    y_max = max(Nstep_vector.max(), Npixel_per_lambda_vector.max(), L) * 1.05
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)

    # legenda compatta e leggibile
    ax.legend(loc='best', ncol=2, frameon=True, framealpha=0.9, fancybox=True)

    plt.show()
    
def show_eta_vs_ptv2():

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Stile globale (comodo per tesi/figure)
    plt.rcParams.update({
        "figure.dpi": 140,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "axes.titlepad": 8,
    })

    # --- Parametri ---
    wl = 633e-9
    L = 256
    Dpx = 545 * 2

    # φ_w di interesse espressi come frazione di 2π
    phi_fracs = [1.00, 0.99, 0.98, 0.95, 0.90, 0.50]  # 1.00 evidenzia il caso ideale 2π
    labels = [fr'$\phi_w/(2\pi)={f:.2f}$' for f in phi_fracs]

    Npoints = 1000
    tilt_c2_vector = np.linspace(1e-6, 80e-6, Npoints)

    # x in micrometri PtV (wf)
    ptv_um = tilt_c2_vector * 4 / 1e-6

    # Preallocazioni
    Nstep = np.zeros(Npoints)
    Npx_per_lambda = np.zeros(Npoints)

    # --- Figura con 2 subplot allineati ---
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 5.3), sharex=True, sharey=False)
    fig.suptitle(fr'Diffraction efficiency vs PtV tilt  |  $L={L}$, $D_{{px}}={Dpx}$, $\lambda={wl*1e9:.0f}\,$nm')

    # Colori coerenti per coppie (η0 tratteggiata, η1 piena)
    base_cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(phi_fracs)))

    # --- Loop sui PtV e calcolo N solo una volta per il caso "di riferimento" φ_w=2π ---
    # Nota: N dipende dal tilt e dalla soglia φ_w usata nella discretizzazione dello sawtooth.
    # Se nella tua get_N il parametro φ_w influenza il risultato, mantieni la chiamata phi_ref = 2π.
    phi_ref = 2 * np.pi  # per il conteggio di N, usiamo il caso 2π come riferimento operativo
    for idx, amp in enumerate(tilt_c2_vector):
        c2 = amp
        wf_ptv = 4 * c2
        phase_ptv = 2 * np.pi * wf_ptv / wl
        phase_tilt = np.linspace(0, phase_ptv, Dpx)
        N = get_N(phase_tilt, phi_ref, L, Dpx)
        Npxpl = get_Npix_per_Lambda(phase_tilt, phi_ref, L, Dpx)
        Nstep[idx] = N
        Npx_per_lambda[idx] = Npxpl

    # --- Tracce η0 e η1 per ciascun φ_w ---
    handles_left = []
    handles_right = []

    for color, frac, lab in zip(base_cmap, phi_fracs, labels):
        phi_w = (2 * np.pi) * frac

        eta0 = np.empty(Npoints)
        eta1 = np.empty(Npoints)
        for i, N in enumerate(Nstep):
            eta0[i] = eta(0, N, phi_w)
            eta1[i] = eta(1, N, phi_w)

        # φ_w = 2π: evidenzia che η0 ≡ 0
        if np.isclose(frac, 1.0, atol=1e-12):
            # linea nera spessa a zero per η0 (più chiaro del numerico)
            ax0.plot(ptv_um, np.zeros_like(ptv_um), color='k', lw=2.6, ls='-', label=r'$\phi_w/2\pi=1$')
            # η1 per φ_w=2π in nero spesso
            line1, = ax1.plot(ptv_um, eta1, color='k', lw=2.6, ls='-', label=fr'  {lab}')
            handles_right.append(line1)
            # annotazione
            # ax0.annotate(r'$\phi_w=2\pi \Rightarrow \eta_0=0\ \forall\,\mathrm{PtV}$',
            #              xy=(ptv_um[int(0.65*len(ptv_um))], 0.0),
            #              xytext=(0, 18), textcoords='offset points',
            #              ha='center', va='bottom', fontsize=11,
            #              bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.7', alpha=0.9))
        else:
            # stesse palette per η0 (tratteggiata) e η1 (piena)
            line0, = ax0.plot(ptv_um, eta0, color=color, lw=2.0, ls='--', label=fr'{lab}')
            line1, = ax1.plot(ptv_um, eta1, color=color, lw=2.0, ls='-',  label=fr'{lab}')
            handles_left.append(line0)
            handles_right.append(line1)

    # --- layout pannello sinistro (η0) ---
    ax0.set_xlabel(r'Tilt PtV [$\mu$m wf]')
    ax0.set_ylabel(r'$\eta_0$')
    ax0.set_title(r'Order $q=0$')
    ax0.xaxis.set_major_locator(MaxNLocator(7))
    ax0.yaxis.set_major_locator(MaxNLocator(6))
    ax0.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax0.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax0.set_xlim(0, ptv_um.max())
    ax0.set_ylim(-0.02, 1.02)

    # inset zoom (utile a mostrare i residui quando φ_w≠2π)
    axins = inset_axes(ax0, width="30%", height="30%", loc="upper right", borderpad=1.2)
    for line in handles_left:
        x = line.get_xdata(); y = line.get_ydata()
        axins.plot(x, y, color=line.get_color(), lw=1.6, ls='--')
    # anche la linea nera a zero per riferimento
    axins.plot(ptv_um, np.zeros_like(ptv_um), color='k', lw=1.4)
    axins.set_xlim(0, ptv_um.max())
    axins.set_ylim(-0.01, 0.05)  # zoom sui residui piccoli
    axins.xaxis.set_major_locator(MaxNLocator(4))
    axins.yaxis.set_major_locator(MaxNLocator(4))
    axins.grid(True, linestyle=':', alpha=0.25)
    axins.set_title("zoom", fontsize=10, pad=2)

    # --- layout pannello destro (η1) ---
    ax1.set_xlabel(r'Tilt PtV [$\mu$m wf]')
    ax1.set_ylabel(r'$\eta_1$')
    ax1.set_title(r'Order $q=1$')
    ax1.xaxis.set_major_locator(MaxNLocator(7))
    ax1.yaxis.set_major_locator(MaxNLocator(6))
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.set_xlim(0, ptv_um.max())
    ax1.set_ylim(-0.02, 1.02)
    
    # inset zoom su eta_1 (0–120 µm)
    axins1 = inset_axes(ax1, width="30%", height="20%", loc="upper right", borderpad=1.2)
    for line in handles_right:
        x = line.get_xdata(); y = line.get_ydata()
        axins1.plot(x, y, color=line.get_color(), lw=1.6, ls='-')
    axins1.set_xlim(0, 120)  # PtV in µm
    axins1.set_ylim(0.9, 1.0)
    # # limiti Y calcolati sul range 0–120 µm per adattare lo zoom
    # ymins, ymaxs = [], []
    # for line in handles_right:
    #     x = line.get_xdata(); y = line.get_ydata()
    #     m = (x >= 0) & (x <= 120)
    #     if np.any(m):
    #         ymins.append(np.min(y[m])); ymaxs.append(np.max(y[m]))
    # if ymins and ymaxs:
    #     ypad = 0.05*(max(ymaxs)-min(ymins))
    #     axins1.set_ylim(min(ymins)-ypad, max(ymaxs)+ypad)
    axins1.xaxis.set_major_locator(MaxNLocator(4))
    axins1.yaxis.set_major_locator(MaxNLocator(4))
    axins1.grid(True, linestyle=':', alpha=0.25)
    axins1.set_title("zoom", fontsize=10, pad=2)
    # # --- linee guida facoltative utili ---
    # # PtV = λ (verticale)
    # x_lambda = wl / 1e-6
    # for ax in (ax0, ax1):
    #     ax.axvline(x_lambda, color='0.25', lw=1.2, ls='--', alpha=0.4)
    #     ax.text(x_lambda, ax.get_ylim()[1]*0.95, r'$PtV=\lambda$', ha='center', va='top',
    #             fontsize=10, color='0.25', rotation=90, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.8', alpha=0.9))

    # --- legende chiare ---
    # Sinistra: solo le curve η0 (escludo la duplicazione colori dalla destra)
    leg0 = ax0.legend(loc='upper left', frameon=True, framealpha=0.95, ncol=1,  fontsize=9)
    # Destra: tutte le φ_w di η1 (inclusa la nera φ_w=2π)
    leg1 = ax1.legend(loc='lower left', frameon=True, framealpha=0.95, ncol=1,  fontsize=9)

    fig.tight_layout()
    plt.show()

def show_eta_map2():

    # --- Parametri ---
    wl = 633e-9
    L = 256
    Dpx = 545*2
    phi_w = 2*np.pi

    Npoints = 1000
    tilt_c2_vector = np.linspace(1e-6, 100e-6, Npoints)
    ptv_um = tilt_c2_vector * 4 / 1e-6  # PtV in µm (wf)

    # Limita gli ordini visualizzati per una mappa leggibile (es. ±20)
    qmax_vis = 20
    q_vect = np.arange(-qmax_vis, qmax_vis+1)             # ordini mostrati
    n_q = q_vect.size

    # Se vuoi calcolare più ordini "internamente", aumenta qmax_calc
    qmax_calc = qmax_vis
    q_calc = np.arange(-qmax_calc, qmax_calc+1)

    eta_2d = np.zeros((n_q, Npoints))

    # Calcolo N una sola volta per ogni PtV (dipende dal tilt)
    for j, c2 in enumerate(tilt_c2_vector):
        wf_ptv = 4*c2
        phase_ptv = 2*np.pi*wf_ptv/wl
        phase_tilt = np.linspace(0, phase_ptv, Dpx)
        N = get_N(phase_tilt, phi_w, L, Dpx)

        # efficienze per tutti gli ordini desiderati
        eta_full = eta(q_vect, N, phi_w)   # shape (n_q,)
        eta_2d[:, j] = eta_full

    # Evita zeri in scala log
    eps = 1e-8
    eta_2d_clip = np.clip(eta_2d, eps, 1.0)

    # --- Figura: mappa + tagli ---
    fig, (ax_map, ax_cuts) = plt.subplots(1, 2, figsize=(11.5, 5.2), constrained_layout=True)
    fig.suptitle(fr'Diffraction efficiency map  |  $\phi_w=2\pi$, $L={L}$, $D_{{px}}={Dpx}$, $\lambda={wl*1e9:.0f}$ nm')

    # Mappa 2D (q vs PtV)
    im = ax_map.imshow(
        eta_2d_clip,
        origin='lower',
        aspect='auto',
        cmap='viridis',
        norm=LogNorm(vmin=1e-6, vmax=1.0),
        extent=[ptv_um.min(), ptv_um.max(), q_vect.min()-0.5, q_vect.max()+0.5],
    )
    cbar = fig.colorbar(im, ax=ax_map, pad=0.02, label=r'$\eta_q$')
    from matplotlib.ticker import LogLocator, LogFormatter
    
    cbar.locator = LogLocator(base=10, numticks=6)
    cbar.formatter = LogFormatter(base=10)
    cbar.update_ticks()
    ax_map.tick_params(axis='y', labelsize=8)

    # Ticks asse Y: assicurati che q=1 sia visibile
    step_y = 5  # ogni 2 ordini; cambia a 1 se vuoi tutte le tacche
    yticks = np.arange(q_vect.min(), q_vect.max()+1, step_y)
    if 1 not in yticks:
        yticks = np.unique(np.append(yticks, 1))
    ax_map.set_yticks(yticks)
    ax_map.set_ylabel('Diffraction order $q$')
    ax_map.set_xlabel(r'Tilt PtV [$\mu$m wf]')

    # Evidenzia q=1
    ax_map.axhline(1, color='k', lw=0.8, ls='--', alpha=0.35)
    # ax_map.text(ptv_um.min(), 1+0.5, 'q=1', color='w', va='bottom', ha='left',
    #             fontsize=10, bbox=dict(boxstyle='round,pad=0.2', fc='black', ec='none', alpha=0.25))
    
    # --- Pannello sinistro: Energy budget (conservazione) ---
    eta_2d = main()
    tot = eta_2d.sum(axis=1)                     # somma su tutti gli ordini
    #residual = np.clip(1.0 - tot, 1e-12, None)   # residuo positivo per scala log
    ax_sum = ax_cuts
    ax_sum.tick_params(axis='y', labelsize=8)
    ax_sum.plot(ptv_um, tot, lw=2.0, label=r'$\sum_q \eta_q$')
    ax_sum.axhline(1.0, color='k', lw=1.2, ls='--', alpha=0.5)
    ax_sum.set_xlabel(r'Tilt PtV [$\mu$m wf]')
    ax_sum.set_ylabel(r'$\sum_q \eta_q$')
    ax_sum.set_ylim(0.999, 1.00)  # stringi la banda attorno a 1 per far vedere deviazioni
    ax_sum.grid(True, linestyle='--', alpha=0.35)
    ax_sum.legend(loc='lower left', frameon=True, framealpha=0.95)
    
    plt.show()

