from bronte import startup
from bronte.package_data import other_folder
import numpy as np 
from astropy.io import fits
from bronte.utils.retry_on_zmqrpc_timeout_error import retry_on_timeout
import time

def main(ftag, jnoll_mode=2):
    
    factory = startup.specula_startup()
    
    flat = np.zeros(1920*1152)
    factory.deformable_mirror.set_shape(flat)
    texp_ms = retry_on_timeout(lambda: factory.sh_camera.exposureTime())
    
    ref_frame = factory.sh_camera.getFutureFrames(1).toNumpyArray() - factory.sh_camera_master_bkg
    
    Nscans = 61
    frame_size = 2048
    frame_cube = np.zeros((Nscans, frame_size, frame_size))
    coef_scan_vector = np.linspace(-30, 30, Nscans) * 1e-6 
    
    modal_cmd = np.zeros(3)
    mode_index2be_scan = jnoll_mode-2
    
    for idx, amp in enumerate(coef_scan_vector):
        print(f"+ Scan {idx+1}/{Nscans}")
        modal_cmd[mode_index2be_scan] = amp
        print(f"applying c = {amp} m rms wf")
        cmd = factory.slm_rasterizer.m2c(modal_cmd)
        factory.deformable_mirror.set_shape(cmd)
        time.sleep(0.05)
        frame_cube[idx] = factory.sh_camera.getFutureFrames(1).toNumpyArray() - factory.sh_camera_master_bkg
    
    file_name = other_folder()/(ftag + '.fits')
    hdr = fits.Header()
    hdr['NOLL_J'] =  jnoll_mode
    hdr['TEXP'] = texp_ms
    fits.writeto(file_name, frame_cube, hdr)
    fits.append(file_name, ref_frame)
    fits.append(file_name, coef_scan_vector)