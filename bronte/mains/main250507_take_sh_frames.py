from astropy.io import fits
from bronte import startup
import numpy as np 
from bronte.package_data import shframes_folder

def main(ftag):
    
    sf = startup.specula_startup()
    
    cmd = sf.slm_rasterizer.m2c(np.array([2e-6, 0, 0]))
    #cmd = np.zeros(1920*1152)
    sf.deformable_mirror.set_shape(cmd)
    
    ftag_bkg = '250506_135400'
    fname_bkg = shframes_folder() /(ftag_bkg + '.fits')
    hdr = fits.getheader(fname_bkg)
    hdul = fits.open(fname_bkg)
    
    texp = hdr['TEXP_MS']
    sf.sh_camera.setExposureTime(texp)
    bkg = hdul[0].data
    
    Nframes = 100
    
    fr_list = []
    
    for idx in np.arange(Nframes):
        
        frame = sf.sh_camera.getFutureFrames(1).toNumpyArray() - bkg
        frame[frame < 0] = 0
        fr_list.append(frame)
    
    file_name = shframes_folder() / (ftag + '.fits')
    fits.writeto(file_name, np.array(fr_list), hdr)