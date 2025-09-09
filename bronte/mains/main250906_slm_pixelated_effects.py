import numpy as np
import matplotlib.pyplot as plt
from astropy.io.fits.hdu import hdulist

def main():
    
    fill_factor = 0.957
    d = 9.2e-6
    a = np.sqrt(fill_factor*d**2)
    
    wl=633e-9
    z=250e-3
    N=545*2
    D = N*d
    x = np.linspace(-3.15e-3,3.15e-3, 50000)
    
    i_slm = I1d_pixelated(x, a, d, wl, z, N)
    i_ideal = I1d_continuos(x, d*N, wl, z)
    i_ideal1 = I1d_pixelated(x, d, d, wl, z, N)
    
    i_max = i_ideal.max()
    
    plt.figure()
    plt.clf()
    plt.plot(x, i_ideal/i_max, label='continuo')
    plt.plot(x, i_ideal1/i_max, label='eta=100%')
    plt.plot(x, i_slm/i_max, label='eta=95.7%')
    plt.grid('--',alpha=0.3)
    plt.legend(loc='best')
    
    sr = i_slm.max()/i_ideal.max()
    print(f"SR={sr}")
    var_in_rad2 = -np.log(sr)
    sigma_in_nm = (wl/(2*np.pi))*np.sqrt(var_in_rad2)/1e-9
    print(f"error in nm rms wf: {sigma_in_nm}")
    return i_slm/i_max

def I1d_pixelated(x, a, d, wl, z, N):
    
    i0 = (a*N/(wl*z))**2
    i1 = (np.sinc(x*d*N/(wl*z)))**2
    i2 = (np.sinc(x*d/(wl*z)))**2
    i3 = (np.sinc(x*a/(wl*z)))**2
    
    i = i0*i1*i3/i2
    return i
    
def I1d_continuos(x, D, wl, z):
    
    i0 = (D/(wl*z))**2
    i1 = (np.sinc(x*D/(wl*z)))**2
    i=i0*i1
    return i


def mainDC_equivalent():
    
    '''
    Serve solo per calcolare lo SR visto che mi interessa solo l'ordine zero    
    2D PSF simulation (apertura circolare) con **solo zero padding FFT** per migliorare il campionamento della PSF.
    Modello DC-equivalente del fill factor: ampiezza uniforme = FF all'interno dell'apertura.
    Parametri impostati come da tuo setup; puoi cambiare 'pad' per aumentare il campionamento nel piano immagine.
    qui non vedo gli effetti della diffrazione sugli altri ordini principali che trovo a distanze che vanno con 
    l inverso del pitch del pixel
    '''
    # --- Parametri SLM / apertura ---
    Nx, Ny = 1920, 1152       # risoluzione SLM (pixel)
    R_pix = 545               # raggio apertura in pixel SLM
    Nx, Ny = R_pix, R_pix
    FF = 0.957                # fill factor (area)
    
    # --- Fattore di zero-padding FFT (lineare). Esempio: 1=nessun padding, 2=dimensione raddoppiata, 4=quadruplicata.
    pad = 4
    
    # --- Pupilla ideale e con FF "DC-equivalente" ---
    yy, xx = np.indices((Ny, Nx))
    cy, cx = Ny//2, Nx//2
    circ = ((yy - cy)**2 + (xx - cx)**2) <= (R_pix**2)
    
    pupil_ideal = np.zeros((Ny, Nx), dtype=np.complex64)
    pupil_ideal[circ] = 1.0
    
    pupil_FF_dc = np.zeros_like(pupil_ideal)
    pupil_FF_dc[circ] = FF  # ampiezza ridotta uniformemente (replica correttamente la perdita del picco = FF^2)
    
    def psf_from_pupil(pupil, pad=1):
        H, W = pupil.shape
        Hp, Wp = int(H*pad), int(W*pad)
        field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil), s=(Hp, Wp)))
        I = np.abs(field)**2
        return I
    
    # --- Calcolo PSF ---
    I_ideal = psf_from_pupil(pupil_ideal, pad=pad)
    I_FF_dc = psf_from_pupil(pupil_FF_dc, pad=pad)
    
    # --- Strehl e confronto analitico ---
    SR = I_FF_dc.max() / I_ideal.max()
    print(f"Zero-padding = ×{pad} (lineare)")
    print(f"Strehl misurato = {SR:.5f} ; atteso = FF^2 = {FF**2:.5f}")
    
    # --- Sezione 1D al centro (normalizzata) ---
    cyI, cxI = np.unravel_index(np.argmax(I_ideal), I_ideal.shape)
    cut_ideal = I_ideal[cyI, :]
    cut_ff   = I_FF_dc[cyI, :]
    mx = cut_ideal.max()
    x = np.arange(cut_ideal.size) - cxI
    
    plt.figure(figsize=(7,4))
    plt.plot(x, cut_ideal/mx, label='Ideale')
    plt.plot(x, cut_ff/mx, label=f'Con fill factor (FF={FF:.3f})')
    plt.xlim(-2000, 2000)
    plt.yscale('log')
    plt.ylim(1e-10, 1.0)
    plt.xlabel("Coordinate immagine (campioni FFT)")
    plt.ylabel("PSF (norm)")
    plt.title("Sezione 1D della PSF — solo zero-padding FFT")
    plt.legend(loc='best')
    plt.grid(True)
    
    # --- Mappe 2D (log) ---
    def show_psf(I, title):
        plt.figure(figsize=(5,5))
        plt.imshow(np.log10(I/I.max() + 1e-12), origin='lower')
        plt.title(title)
        plt.colorbar(label='log10(PSF norm)')
        plt.axis('off')
    
    show_psf(I_ideal,  f"PSF ideale (pad ×{pad})")
    show_psf(I_FF_dc, f"PSF con fill factor (pad ×{pad})")
    
    print("Done.")

def main250908():
    from astropy.io import fits
    fname5ms = "D:\\phd_slm_edo\\bronte\\psf_camera_frames\\250908_173700.fits"
    fname5s = "D:\\phd_slm_edo\\bronte\\psf_camera_frames\\250908_125300.fits"
    
    texp5s = 5
    texp5ms = 5e-3
    hdulist = fits.open(fname5ms)
    ima5ms = hdulist[0].data
    plt.figure()
    plt.clf()
    plt.imshow(ima5ms)
    
    I5ms_profile = ima5ms[:, 650:654].mean(axis=-1)
    plt.figure()
    plt.clf()
    plt.plot(I5ms_profile, '.-')
    
    hdulist = fits.open(fname5s)
    ima5s = hdulist[0].data
    plt.figure()
    plt.clf()
    plt.imshow(ima5s)
    I5s_profile = ima5s[:, 650:654].mean(axis=-1)
    plt.figure()
    plt.clf()
    plt.plot(I5s_profile, '.-')
    
    
    
    fill_factor = 0.957
    d = 9.2e-6
    a = np.sqrt(fill_factor*d**2)
    
    wl=633e-9
    z=250e-3
    N=545*2
    D = N*d
    x = np.linspace(-3.15e-3,3.15e-3, 50000)
    
    i_slm = I1d_pixelated(x, a, d, wl, z, N)
    
    # plt.figure()
    # plt.clf()
    # plt.plot(x, i_slm/i_slm.max(), label='eta=95.7%')
    # plt.grid('--',alpha=0.3)
    # plt.legend(loc='best')
    
