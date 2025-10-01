# import numpy as np 
# import matplotlib.pyplot as plt
# from tesi_slm.utils.my_tools import  open_fits_file
#
# def show_plots_seima8m_on_ccd():
#     #fpath =  "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\display_phase_screens\\kolmogorov_atmo\\class8m\\"
#     fpath = "D:\\phd_slm_edo\\old_data\\display_phase_screens\\"
#     fname_seima = fpath + "231024daos_seima_r0vis0.2__Nfr100texp10ms_bias_c2_m10umrms.fits"
#
#     head,hdrdat = open_fits_file(fname_seima) 
#     seima_cube = hdrdat[0].data
#     bias = hdrdat[1].data
#     texp = head['T_EX_MS']
#     Nframes = head['N_AV_FR']
#     r0 = head['R_0']
#
#
#     # roi on ghost and tilt
#     plt.figure()
#     plt.clf()
#     plt.imshow(seima_cube[0,400:600,650:1000], cmap = 'inferno')
#     plt.colorbar(label = 'ADU')
#
#     return bias


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from tesi_slm.utils.my_tools import open_fits_file
from mpl_toolkits.axes_grid1 import make_axes_locatable

def _robust_bg_scalar(frame, mask_outside):
    vals = frame[mask_outside]
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) + 1e-12
    sigma_rob = 1.4826 * mad
    keep = (vals >= med - 3*sigma_rob) & (vals <= med + 3*sigma_rob)
    return np.median(vals[keep]) if np.any(keep) else med

def show_plots_seima8m_on_ccd():
    fpath = "D:\\phd_slm_edo\\old_data\\display_phase_screens\\"
    fname = fpath + "231024daos_seima_r0vis0.2__Nfr100texp10ms_bias_c2_m10umrms.fits"

    head, hdrdat = open_fits_file(fname)
    seima_cube = hdrdat[0].data  # [N, Y, X]
    texp_ms = head.get('T_EX_MS', None)
    nframes = seima_cube.shape[0] if seima_cube.ndim == 3 else 1

    # ROI
    y0, y1 = 400, 600
    x0, x1 = 650, 1000

    # maschera fuori-ROI per background
    Y, X = seima_cube.shape[-2], seima_cube.shape[-1]
    mask_outside = np.ones((Y, X), dtype=bool)
    mask_outside[y0:y1, x0:x1] = False

    # sottrazione background per-frame (robusta)
    if seima_cube.ndim == 3:
        corrected = np.empty_like(seima_cube, dtype=float)
        for i in range(nframes):
            bg = _robust_bg_scalar(seima_cube[i], mask_outside)
            corrected[i] = seima_cube[i] - bg
    else:
        bg = _robust_bg_scalar(seima_cube, mask_outside)
        corrected = (seima_cube - bg)[None, ...]

    # >>> CLIP VALORI NEGATIVI A ZERO <<<
    corrected = np.clip(corrected, 0, None)

    # estrai ROI: short vs long
    short_img = corrected[0, y0:y1, x0:x1]
    long_img  = corrected.mean(axis=0)[y0:y1, x0:x1]

    # normalizzazione lineare condivisa
    vmin = float(min(short_img.min(), long_img.min()))
    vmax = float(max(short_img.max(), long_img.max()))
    norm = Normalize(vmin=vmin, vmax=vmax)

    # plotting
    plt.rcParams.update({
        "figure.dpi": 160,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0))
    plt.subplots_adjust(wspace=0.08)  # spaziatura tra i due pannelli

    im0 = axes[0].imshow(short_img, cmap="inferno", norm=norm, origin="upper", aspect="equal")
    axes[0].set_title("Short exposure (1 frame)")
    axes[0].set_xlabel("X [pix]")
    axes[0].set_ylabel("Y [pix]")

    ttl = f"Long exposure (mean of {nframes} frames)" if nframes > 1 else "Long exposure (single frame)"
    im1 = axes[1].imshow(long_img, cmap="inferno", norm=norm, origin="upper", aspect="equal")
    axes[1].set_title(ttl)
    axes[1].set_xlabel("X [pix]")
    # >>> NASCONDI TICK-LABELS Y NEL SECONDO SUBPLOT <<<
    axes[1].tick_params(labelleft=False)

    # >>> UNICO COLORBAR PIÙ PICCOLO, PROPORZIONATO ALL'IMMAGINE <<<
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="4%", pad=0.05)  # più piccolo e alto come l'immagine
    cb = fig.colorbar(im1, cax=cax)
    cb.set_label("ADU")

    for ax in axes:
        ax.tick_params(direction='out', length=4, width=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if texp_ms is not None:
        fig.suptitle(f"Seeing-limited PSF vs zero-order spot", fontsize=12, y=0.99)

    plt.show()
    
    # --- FIGURA 2: solo PSF short exposure (stessa normalizzazione) ---
    fig2, ax2 = plt.subplots(1, 1, figsize=(5.2, 5.0))
    im2 = ax2.imshow(short_img, cmap="jet", norm=norm, origin="upper", aspect="equal")
    ax2.set_title("Short exposure Seeing limited PSF + Tilt")
    ax2.set_xlabel("X [pix]")
    ax2.set_ylabel("Y [pix]")
    

    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="4%", pad=0.05)
    cb2 = fig2.colorbar(im2, cax=cax2)
    cb2.set_label("ADU")
    
    ax2.tick_params(direction='out', length=4, width=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

