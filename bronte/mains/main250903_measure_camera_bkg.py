import numpy as np
from bronte import startup
from astropy.io import fits
from bronte.package_data import shframes_folder, psf_camera_folder

def main(ftag, texp, Nframes, cam, folder):
    
    cam.setExposureTime(texp)
    raw_frames = cam.getFutureFrames(Nframes).toNumpyArray()
    master_frame = np.median(raw_frames, axis=-1)
    
    startup.set_data_dir()
    file_name = folder / (ftag + '.fits')
    hdr = fits.Header()
    hdr['TEXP_MS'] = texp
    fits.writeto(file_name, master_frame, hdr)

def main250903_114900():
    '''
    computing psf camera bkg at 20ms
    '''
    ftag = '250903_114900'
    texp = 20 # ms
    Nframes = 20
   
    sf = startup.specula_startup()
    cam = sf.psf_camera
    main(ftag, texp, Nframes, cam, psf_camera_folder())