import numpy as np
import matplotlib.pyplot as plt

def main():
    
    #PARAMETERS OF COMMANDED TILT
    wl=633e-9
    L = 256*0.5
    Dpx = 545*2
    #SETTING PHASE STROKE
    phi_w = 2*np.pi*0.5
    
    Npoints = 100
    tilt_c2_vector = np.linspace(1e-6, 80e-6, Npoints)
    Nq = 20
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
    
    return eta_2d

def eta(q, N, phi_w):
    
    #q = float(q)
    arg_shift = phi_w*0.5/np.pi - q
    
    envelope = np.sinc(q/N)**2
    interf_num = np.sinc(arg_shift)**2
    interf_denom = np.sinc(arg_shift/N)**2
    eps = 1e-16
    return envelope*interf_num/np.maximum(interf_denom,eps)

def get_N(phase_tilt, phi_w=2*np.pi, L=256, Dpx=545*2):
    
    tilt_ptv = np.ptp(phase_tilt)
    
    if tilt_ptv < phi_w:
        
        dphi = phi_w/L
        N = np.min((L, np.int32(tilt_ptv/dphi)+1))
    
    if tilt_ptv >= phi_w:
        
        alpha = tilt_ptv/Dpx
        Npix_per_Lambda = phi_w/alpha
        
        N = np.min((np.int32(Npix_per_Lambda), L))
    
    return N