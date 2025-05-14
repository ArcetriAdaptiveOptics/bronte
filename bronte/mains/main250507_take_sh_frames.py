from astropy.io import fits
from bronte import startup
import numpy as np 
from bronte.package_data import shframes_folder, modal_offsets_folder

def main(ftag, addOffset = False):
    
    sf = startup.specula_startup()
    
    offset_cmd = 0
    hdr_offset = 'NA'
    if addOffset is True:
        off_tag = '250509_170000'#'250509_161700'
        offset_fname = modal_offsets_folder() / (off_tag+'.fits')
        hdl = fits.open(offset_fname)
        offset = hdl[0].data
        offset_cmd = - offset#factory.slm_rasterizer.m2c(modal_offset)
        hdr_offset = off_tag
        
    sf.deformable_mirror.set_shape(offset_cmd)
    
    cmd = sf.slm_rasterizer.m2c(np.array([0, 0, 0]))
    #cmd = np.zeros(1920*1152)
    sf.deformable_mirror.set_shape(cmd + offset_cmd)
    
    ftag_bkg = '250211_135800'#'250506_135400'
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