import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.special import j1  # Bessel J1

def main():
    # ---------- PARAMETRI ----------
    fname5ms = "D:\\phd_slm_edo\\bronte\\psf_camera_frames\\250908_173700.fits"
    fname5s = "D:\\phd_slm_edo\\bronte\\psf_camera_frames\\250908_125300.fits"
    
    texp_5s  = 5.0
    texp_5ms = 5e-3
    
    # Reticolo / ottica (come nel tuo esempio)
    fill_factor = 0.957
    d  = 9.2e-6
    a  = np.sqrt(fill_factor)*d
    wl = 633e-9
    z  = 250e-3      # <-- qui la focale della lente (piano di Fourier)
    
    # Profilo: media colonne [650:654]
    XCOL_START, XCOL_STOP = 650, 654
    
    # Pixel pitch camera (niente magnification in Fourier): 
    PIXEL_PITCH = 4.65e-6  # [m/pixel]
    
    # Finestra per stimare il core non saturo (larghezza in pixel)
    CORE_HALF_WIDTH = 1
    
    # Finestre laterali per stimare flusso nei lobes (esempi, in pixel rispetto al centro)
    # Metti intervalli dove nel 5s NON sei saturo
    LOBE_WINDOWS = [ (0,470), (600,-1) ]  # dx e sx, cambiale a seconda dei tuoi dati
    
    # ---------- MODELLO ----------
    def I1d_pixelated(x, a, d, wl, z, N):
        i0 = (a*N/(wl*z))**2
        i1 = (np.sinc(x*d*N/(wl*z)))**2
        i2 = (np.sinc(x*d/(wl*z)))**2
        i3 = (np.sinc(x*a/(wl*z)))**2
        return i0*i1*i3/i2
    
    N = 560*2
    
    # ---------- CARICA DATI E PROFILI ----------
    ima5ms = fits.open(fname5ms)[0].data
    ima5s  = fits.open(fname5s)[0].data
    
    I5ms = ima5ms[:, XCOL_START:XCOL_STOP].mean(axis=-1)
    I5s  = ima5s[:,  XCOL_START:XCOL_STOP].mean(axis=-1) - 65.25
    
    # Centro (m=0) dal frame non saturo
    i0 = int(np.argmax(I5ms))
    print(i0)
    # Asse x in metri, stessa griglia dei dati
    pix = np.arange(I5ms.size)
    x   = (pix - i0) * PIXEL_PITCH
    
    # ---------- FLUSSO PER PIXEL NEL CORE (5 ms) ----------
    core_slice = slice(i0-CORE_HALF_WIDTH, i0+CORE_HALF_WIDTH+1)
    core_ADU_5ms = np.mean(I5ms[core_slice])
    core_flux_per_s = core_ADU_5ms / texp_5ms   # ADU/s al centro
    
    print(f"Flusso core (ADU/s): {core_flux_per_s:.2f}")
    
    # ---------- FLUSSO PER PIXEL NEI LOBES (5 s) ----------
    # maschera saturazione (se sai il full-scale, usa quello)
    sat_level = np.iinfo(ima5s.dtype).max if np.issubdtype(ima5s.dtype, np.integer) else np.nanmax(I5s)
    mask_nonsat_5s = I5s < 0.98*sat_level
    
    lobe_fluxes = []
    for w in LOBE_WINDOWS:
        j1 = i0 + w[0]
        j2 = i0 + w[1]
        if j1 > j2: j1, j2 = j2, j1
        win = np.arange(max(0,j1), min(I5s.size, j2+1))
        win = win[ mask_nonsat_5s[max(0,j1):min(I5s.size, j2+1)] ]
        if win.size == 0:
            lobe_fluxes.append(np.nan)
            continue
        lobe_ADU_5s = np.mean(I5s[win])
        lobe_fluxes.append(lobe_ADU_5s / texp_5s)  # ADU/s
    print("Flusso lobi (ADU/s) per finestre:", lobe_fluxes)
    
    # ---------- SCALA E SOVRAPPONI IL MODELLO ----------
    xx = np.linspace(-2.5e-3,2.5e-3,50000)
    Ith = I1d_pixelated(xx, a, d, wl, z, N)
    # Porta il modello in ADU/s, agganciandolo al core misurato
    Ith_per_s = core_flux_per_s * (Ith / np.max(Ith))  # max(Ith) è ~Ith(0)
    
    # Predizione ADU per il 5 s (solo per confronto visivo)
    Ith_5s = Ith_per_s * texp_5s
    
    # profilo 5 s con i punti saturi mascherati a NaN (così il plot non li unisce)
    I5s_masked = I5s.astype(float).copy()
    I5s_masked[~mask_nonsat_5s] = np.nan
    
    # ---------- PLOT ----------
    plt.figure(figsize=(9,5))
    plt.plot(x*1e3, I5ms/texp_5ms, label="Misura 5 ms (ADU/s)")
    plt.plot(xx*1e3, Ith_per_s, '--', label="Modello (ADU/s, scalato sul core)")
    plt.xlabel("x [mm]")
    plt.ylabel("ADU/s per pixel")
    plt.title("Core non saturo: misura vs modello")
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend()
    
    plt.figure(figsize=(9,5))
    plt.plot(x*1e3, I5s, alpha=0.3, label="Misura 5 s (grezza)")
    plt.plot(x*1e3, I5s_masked, label="Misura 5 s (non saturo)")
    plt.plot(xx*1e3, Ith_5s, '--', label="Modello → 5 s (ADU)")
    plt.xlabel("x [mm]")
    plt.ylabel("ADU per pixel")
    plt.title("Lobes sul 5 s: confronto con il modello (centro saturo mascherato)")
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main2():
    
     # ---------- PARAMETRI ----------
    fname5ms = "D:\\phd_slm_edo\\bronte\\psf_camera_frames\\250908_173700.fits"
    fname5s = "D:\\phd_slm_edo\\bronte\\psf_camera_frames\\250908_125300.fits"
    
    texp_5s  = 5.0
    texp_5ms = 5e-3
    
    # Reticolo / ottica (come nel tuo esempio)
    fill_factor = 0.957
    d  = 9.2e-6
    a  = np.sqrt(fill_factor)*d
    wl = 633e-9
    z  = 250e-3      # <-- qui la focale della lente (piano di Fourier)
    
    # Profilo: media colonne [650:654]
    XCOL_START, XCOL_STOP = 650, 654
    # --- parametri chiave ---
    PIXEL_PITCH = 4.65e-6     # [m/pixel]
    f = 250e-3                # focale (il tuo z)
    D = 10.5e-3               # diametro pupilla SLM (scegli 10.2–10.4 mm)
    
    N = 560*2
    
    # ---------- CARICA DATI E PROFILI ----------
    ima5ms = fits.open(fname5ms)[0].data
    ima5s  = fits.open(fname5s)[0].data
    
    I5ms = ima5ms[:, XCOL_START:XCOL_STOP].mean(axis=-1)
    I5s  = ima5s[:,  XCOL_START:XCOL_STOP].mean(axis=-1) - 65.25
    # Centro (m=0) dal frame non saturo
    i0 = int(np.argmax(I5ms))
    print(i0)
    
    pix = np.arange(I5ms.size)
    x = (pix - i0) * PIXEL_PITCH  # [m]
    
    # --- flusso per pixel nel core (5 ms) ---
    core_slice = slice(i0-5, i0+5+1)
    core_ADU_5ms = np.mean(I5ms[core_slice])
    I0_per_s = core_ADU_5ms / texp_5ms   # ADU/s
    
    # --- modello Airy sugli stessi x, scalato in ADU/s ---
    Ith_rel = I_airy_1d(x, D=D, wl=wl, f=f)  # relativo (max ~ 1 al centro)
    Ith_per_s = I0_per_s * (Ith_rel / Ith_rel.max())
    
    # --- confronto con 5 s (mascherando saturi) ---
    Ith_5s = Ith_per_s * texp_5s
    sat_level = np.iinfo(ima5s.dtype).max if np.issubdtype(ima5s.dtype, np.integer) else np.nanmax(I5s)
    I5s_masked = I5s.astype(float).copy()
    I5s_masked[I5s >= 0.98*sat_level] = np.nan
    
    # --- plot essenziale ---
    
    plt.figure(figsize=(9,5))
    plt.plot(x*1e3, I5ms/texp_5ms, label="5 ms (ADU/s)")
    plt.plot(x*1e3, Ith_per_s, '--', label="Airy (ADU/s, scalata)")
    plt.xlabel("x [mm]"); plt.ylabel("ADU/s per pixel"); plt.grid(True, ls='--', alpha=0.3); plt.legend()
    
    plt.figure(figsize=(9,5))
    plt.plot(x*1e3, I5s_masked, label="5 s (non saturo)")
    plt.plot(x*1e3, Ith_5s, '--', label="Airy → 5 s (ADU)")
    plt.xlabel("x [mm]"); plt.ylabel("ADU per pixel"); plt.grid(True, ls='--', alpha=0.3); plt.legend()
    plt.show()

def I_airy_1d(x, D, wl, f):
    # x in metri sul piano di immagine (fuoco); D diametro pupilla [m]
    # ritorna intensità non scalata (unità arbitrarie)
    k = np.pi * D * np.abs(x) / (wl * f)
    # gestisci il limite k->0 per evitare 0/0
    I = np.ones_like(x)
    m = k > 0
    I[m] = (2.0 * j1(k[m]) / k[m])**2
    return I