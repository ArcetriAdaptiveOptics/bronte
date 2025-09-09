import numpy as np
from bronte import startup
from bronte.package_data import psf_camera_folder, other_folder
from bronte.wfs.slm_rasterizer import SlmRasterizer
from arte.types.mask import CircularMask
import matplotlib.pyplot as plt
import time
from astropy.io import fits
from bronte.utils.camera_master_bkg import CameraMasterMeasurer

def main(ftag, c2, wl_wrap, ftag_bkg):
    
    
    bkg, texp = load_bkg(ftag_bkg)
    
    slm_pupil_mask = get_circular_mask()
    sr = SlmRasterizer( slm_pupil_mask, Nmodes = 1)
    sf = startup.specula_startup()
    zc = np.array([c2])
    
    tilt = sr.zernike_coefficients_to_raster(zc).toNumpyArray()
        
    tilt_wrapped = get_wrapped_wf(tilt, wl_wrap)
    tilt_wrapped = sr.load_a_tilt_under_pupil_mask(tilt_wrapped)
    
    plt.figure()
    plt.plot(tilt[579, :], label = 'commanded tilt')
    plt.plot(tilt_wrapped[579,:],label = 'wrapped tilt')
    plt.xlabel('N pixel')
    plt.ylabel('OPD [nm] rms wf')
    
    cam = sf.psf_camera
    cam.setExposureTime(texp)
    
    flat = tilt_wrapped.copy()
    flat[flat.mask == False] = 0
    cmd0 = sr.reshape_map2vector(flat)
    sf.deformable_mirror.set_shape(cmd0)
    time.sleep(2)
    ima0 = cam.getFutureFrames(1).toNumpyArray()
    
    red0 = ima0-bkg
    red0[red0<0]=0
    
    cmd = sr.reshape_map2vector(tilt_wrapped)
    sf.deformable_mirror.set_shape(cmd)
    time.sleep(2)

    ima = cam.getFutureFrames(1).toNumpyArray()
    red1 = ima-bkg
    red1[red1<0]=0
    
    fname = psf_camera_folder() / (ftag + '.fits')
    
    hdr = fits.Header()
    hdr['TEXP_MS'] = texp
    hdr['WL_WARP'] = wl_wrap
    hdr['C2_M'] = c2
    
    fits.writeto(fname, red1, hdr)
    fits.append(fname, red0)
    
    return red1,red0

def get_circular_mask():
    
    cmask = CircularMask(frameShape=(1152,1920), maskCenter=(579, 968), maskRadius=545)
    
    return cmask

def get_wrapped_wf(wf, wl_wrap = 633e-9):
    
    return wf % wl_wrap


def get_bkg(ftag, texp=6.5):
    
    sf = startup.specula_startup()
    fname = psf_camera_folder() /(ftag + '.fits')
    texp = texp
    cam = sf.psf_camera
    cam.setExposureTime(texp)
    bkgs = cam.getFutureFrames(20).toNumpyArray()
    master_bkg = np.median(bkgs, axis = -1)
    
    plt.figure()
    plt.clf()
    plt.imshow(master_bkg)
    
    hdr = fits.Header()
    hdr['TEXP_MS'] = texp
    
    fits.writeto(fname, master_bkg, hdr)

def load_bkg(ftag):
    startup.set_data_dir()
    bkg, texp = CameraMasterMeasurer.load_master(ftag, 'psf_bkg')
    return bkg, texp
###
# mains
def main250909_154800():
    
    ftag = '250909_154800'
    c2 = 2e-6
    wl = 633e-9
    wl_wrap = 0.5 * wl
    ftag_bkg = '250909_153700'
    return main(ftag, c2, wl_wrap, ftag_bkg)

def main250909_155600():
    
    ftag = '250909_155600'
    c2 = 2e-6
    wl = 633e-9
    wl_wrap = 0.75 * wl
    ftag_bkg = '250909_153700'
    return main(ftag, c2, wl_wrap, ftag_bkg)

def main250909_155800():
    
    ftag = '250909_155800'
    c2 = 2e-6
    wl = 633e-9
    wl_wrap = 0.85 * wl
    ftag_bkg = '250909_153700'
    return main(ftag, c2, wl_wrap, ftag_bkg)

def main250909_160000():
    
    ftag = '250909_160000'
    c2 = 2e-6
    wl = 633e-9
    wl_wrap = 0.9 * wl
    ftag_bkg = '250909_153700'
    return main(ftag, c2, wl_wrap, ftag_bkg)

def main250909_160200():
    
    ftag = '250909_160200'
    c2 = 2e-6
    wl = 633e-9
    wl_wrap = 0.95 * wl
    ftag_bkg = '250909_153700'
    return main(ftag, c2, wl_wrap, ftag_bkg)

def main250909_161100():
    
    ftag = '250909_161100'
    c2 = 2e-6
    wl = 633e-9
    wl_wrap = 1 * wl
    ftag_bkg = '250909_153700'
    return main(ftag, c2, wl_wrap, ftag_bkg)

def main250909_161400():
    
    ftag = '250909_161400'
    c2 = 2e-6
    wl = 633e-9
    wl_wrap = 1.75 * wl
    ftag_bkg = '250909_153700'
    return main(ftag, c2, wl_wrap, ftag_bkg)

def main250909_161900():
    
    ftag = '250909_161900'
    c2 = 2e-6
    wl = 633e-9
    wl_wrap = 1.5 * wl
    ftag_bkg = '250909_153700'
    return main(ftag, c2, wl_wrap, ftag_bkg)

def main250909_162100():
    
    ftag = '250909_162100'
    c2 = 2e-6
    wl = 633e-9
    wl_wrap = 1.9 * wl
    ftag_bkg = '250909_153700'
    return main(ftag, c2, wl_wrap, ftag_bkg)