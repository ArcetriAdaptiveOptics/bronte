from astropy.io import fits
from bronte import startup
import numpy as np 
from bronte.package_data import shframes_folder

def main(ftag):
    '''
    this main is meant to acquire 100 sh frames
    used to estimate the slope offset when on the slm is set 
    the zero command in circular radius of 545 pixels
    '''
    sf = startup.specula_startup()
    
    wfc = np.zeros((1152,1920))
    ma_wfc = np.ma.array(data = wfc, mask = sf.slm_pupil_mask.mask())
    ma_wfc = sf.slm_rasterizer.load_a_tilt_under_pupil_mask(ma_wfc)
    wfc_cmd = sf.slm_rasterizer.reshape_map2vector(ma_wfc)
    
    sf.deformable_mirror.set_shape(wfc_cmd.data)
    

    ftag_bkg = '250610_111300'
    fname_bkg = shframes_folder() /(ftag_bkg + '.fits')
    hdr = fits.getheader(fname_bkg)
    hdul = fits.open(fname_bkg)
    
    texp = hdr['T_EXP']
    sf.sh_camera.setExposureTime(texp)
    bkg = hdul[1].data
    
    Nframes = 100
    
    fr_list = []
    
    for idx in np.arange(Nframes):
        
        frame = sf.sh_camera.getFutureFrames(1).toNumpyArray() - bkg
        frame[frame < 0] = 0
        fr_list.append(frame)
    
    file_name = shframes_folder() / (ftag + '.fits')
    fits.writeto(file_name, np.array(fr_list), hdr)