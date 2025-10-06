import numpy as np 
from bronte.package_data import other_folder
from bronte.startup import specula_startup, set_data_dir
from astropy.io import fits
from bronte.utils.raw_strehl_ratio_computer import StrehlRatioComputer
from bronte.utils.data_cube_cleaner import DataCubeCleaner
from bronte.plots.main251001_psf_sharpening import main_low_order, main_high_order
import matplotlib.pyplot as plt

def main(ftag, Nframes  = 10, zc2apply = np.zeros(3)):
    
    sf = specula_startup()

    texp = sf.psf_camera.exposureTime()
    fps = sf.psf_camera.getFrameRate()
    
    #zc2apply = np.zeros(3)
    
    command  = sf.slm_rasterizer.m2c(zc2apply, applyTiltUnderMask=True)
    
    sf.deformable_mirror.set_shape(command)
    
    yc = 554
    xc = 653
    size = 40
    
    #measure_flux_in_roi = np.zeros(Nframes)
    psf_cube = np.zeros((Nframes, size, size))
    
    for idx in np.arange(Nframes):
        
        psf_in_roi = get_psf_in_roi(sf, yc, xc, size)
        psf_cube[idx] = psf_in_roi
        #measure_flux_in_roi[idx] = psf_in_roi.sum()
    
    
    fname = other_folder() / (ftag + '.fits')
    hdr = fits.Header()
    hdr['TEXP_MS'] = texp
    hdr['FPS'] = fps
    
    fits.writeto(fname, psf_cube, hdr)
    
    
def main2(ftag, Nframes  = 10, zc2apply = np.zeros(3), Nframe2avarage = 10):
    
    sf = specula_startup()

    texp = sf.psf_camera.exposureTime()
    fps = sf.psf_camera.getFrameRate()
    
    #zc2apply = np.zeros(3)
    
    command  = sf.slm_rasterizer.m2c(zc2apply, applyTiltUnderMask=True)
    
    sf.deformable_mirror.set_shape(command)
    
    yc = 554
    xc = 653
    size = 40
    
    #measure_flux_in_roi = np.zeros(Nframes)
    psf_cube = np.zeros((Nframes, size, size))
    
    for idx in np.arange(Nframes):
        
        psf_in_roi = get_psf_in_roi_as_avarage(sf, yc, xc, size, Nframe2avarage)
        psf_cube[idx] = psf_in_roi
        #measure_flux_in_roi[idx] = psf_in_roi.sum()
    
    
    fname = other_folder() / (ftag + '.fits')
    hdr = fits.Header()
    hdr['TEXP_MS'] = texp
    hdr['FPS'] = fps
    
    fits.writeto(fname, psf_cube, hdr)
        
def get_psf_in_roi(sf, yc, xc, size):
    
    yc_roi = yc
    xc_roi = xc
    size = size
    bkg = sf.psf_camera_master_bkg
    
    raw_data = sf.psf_camera.getFutureFrames(1).toNumpyArray()
    master_image = raw_data - bkg
    master_image[master_image < 0] = 0
            
    hsize = int(np.round(size*0.5))
    roi_master = master_image[yc_roi-hsize:yc_roi+hsize, xc_roi-hsize:xc_roi+hsize]
    return roi_master

def get_psf_in_roi_as_avarage(sf, yc, xc, size, Nframe2avarage = 10):
    
    yc_roi = yc
    xc_roi = xc
    size = size
    bkg = sf.psf_camera_master_bkg
    
    
    raw_dataCube = sf.psf_camera.getFutureFrames(Nframe2avarage).toNumpyArray()
    master_image = DataCubeCleaner.get_master_from_rawCube(raw_dataCube, bkg)
            
    hsize = int(np.round(size*0.5))
    roi_master = master_image[yc_roi-hsize:yc_roi+hsize, xc_roi-hsize:xc_roi+hsize]
    return roi_master

# --- data acquisition ---

def main251006_115700():
    
    ftag = '251006_115700'
    Nframes = 200
    zc2apply = np.zeros(3)
    main(ftag, Nframes, zc2apply)

def main251006_121100():
    
    ftag = '251006_121100'
    Nframes = 200
    zc2apply = np.zeros(3)
    main(ftag, Nframes, zc2apply)
    
def main251006_121500():
    
    ftag = '251006_121500'
    Nframes = 200
    zc2apply = np.zeros(3)
    main(ftag, Nframes, zc2apply)

def main251006_122500():
    
    ftag = '251006_122500'
    Nframes = 1000
    zc2apply = np.zeros(3)
    main(ftag, Nframes, zc2apply)

def main251006_120300():
    
    ftag = '251006_120300'
    Nframes = 200
    
    zc2apply =  np.array([ 0.0e+00,  0.0e+00, -2.0e-08,  4.0e-08,  1.5e-08,  1.0e-08,
        1.0e-08,  5.0e-09, -5.0e-09, -5.0e-09])
  
    main(ftag, Nframes, zc2apply)
    
def main251006_121300():
    
    ftag = '251006_121300'
    Nframes = 200
    
    zc2apply =  np.array([ 0.0e+00,  0.0e+00, -2.0e-08,  4.0e-08,  1.5e-08,  1.0e-08,
        1.0e-08,  5.0e-09, -5.0e-09, -5.0e-09])
  
    main(ftag, Nframes, zc2apply)

def main251006_121600():
    
    ftag = '251006_121600'
    Nframes = 200
    
    zc2apply =  np.array([ 0.0e+00,  0.0e+00, -2.0e-08,  4.0e-08,  1.5e-08,  1.0e-08,
        1.0e-08,  5.0e-09, -5.0e-09, -5.0e-09])
  
    main(ftag, Nframes, zc2apply)
    
def main251006_123100():
    
    ftag = '251006_123100'
    Nframes = 1000
    
    zc2apply =  np.array([ 0.0e+00,  0.0e+00, -2.0e-08,  4.0e-08,  1.5e-08,  1.0e-08,
        1.0e-08,  5.0e-09, -5.0e-09, -5.0e-09])
  
    main(ftag, Nframes, zc2apply)
    
def main251006_150600():
    
    ftag = '251006_150600'
    Nframes = 200
    Nframes2average = 10
    
    zc2apply =  np.array([ 0.0e+00,  0.0e+00, -2.0e-08,  4.0e-08,  1.5e-08,  1.0e-08,
        1.0e-08,  5.0e-09, -5.0e-09, -5.0e-09])
  
    main2(ftag, Nframes, zc2apply, Nframes2average)

def main251006_154000():
    ftag = '251006_154000'
    Nframes = 1000
    
    c_hat_low, _ =  main_low_order()
    c_hat_hi, _ = main_high_order() 
    plt.close('all')
    
    zc2apply = np.zeros(30)
    
    zc2apply[0:2] = 0
    zc2apply[2:10] = c_hat_low*1e-9
    zc2apply[10:] = c_hat_hi*1e-9
    
    main(ftag, Nframes, zc2apply)
    
# --- analysis --- 
class PSFDynamicsAnalyser():
    
    def __init__(self, ftag):
        
        self._ftag = ftag
        self._psf_cube, self._texp, self._fps = self._load()
        self._Nframes = self._psf_cube.shape[0]
        self._sr_pc  = StrehlRatioComputer()
        self._flux_in_roi_vector, self._measured_sr_vector = self._compute_flux_and_sr_in_roi()
        
        self._gain_e_per_adu=3.5409 
        self._ron_adu = 2.4
    
    def _compute_flux_and_sr_in_roi(self):
        
        flux_in_roi_vector = np.zeros(self._Nframes)
        sr_vector = np.zeros(self._Nframes)
        for idx in np.arange(self._Nframes):
            flux_in_roi_vector[idx] = self._psf_cube.sum()
            sr_vector[idx] = self._sr_pc.get_SR_from_image(self._psf_cube[idx], enable_display=False)
        return flux_in_roi_vector, sr_vector
    
    def _load(self):
        set_data_dir()
        fname = other_folder() / (self._ftag + '.fits')
        hdr = fits.getheader(fname)
        texp =  hdr['TEXP_MS'] 
        fps = hdr['FPS']
        
        hdulist = fits.open(fname)
        psf_cube = hdulist[0].data
        return psf_cube, texp, fps