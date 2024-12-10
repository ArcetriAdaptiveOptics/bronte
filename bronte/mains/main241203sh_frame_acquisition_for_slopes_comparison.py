import numpy as np
import time
from bronte.startup import startup
from bronte import package_data
from arte.types.zernike_coefficients import ZernikeCoefficients
from astropy.io import fits


def main(ftag, amp):
    SLM_RESPONSE_TIME = 0.005
    bf = startup()
    
    flat = np.zeros(1152*1920)
    bf.deformable_mirror.set_shape(flat)
    texp = 10
    bf.sh_camera.setExposureTime(texp)
    
    mode_idex_list = np.array([2,3,4])
    nmodes = len(mode_idex_list)
    sh_frame_cube = np.zeros((nmodes+1, 2048, 2048))
    
    zc_vec = np.array([[0.,0.,0.],
                       [amp, 0., 0.],
                       [ 0., amp, 0.],
                       [ 0., 0., amp]])
    
    for idx in np.arange(nmodes+1):
        
        zc = ZernikeCoefficients.fromNumpyArray(zc_vec[idx])
        wfz = bf.slm_rasterizer.zernike_coefficients_to_raster(zc)
        #wfz_to_raster = bf.slm_rasterizer.get_recentered_phase_screen_on_slm_pupil_frame(wfz.toNumpyArray())
        command = bf.slm_rasterizer.reshape_map2vector(wfz.toNumpyArray())
        bf.deformable_mirror.set_shape(command)
        time.sleep(SLM_RESPONSE_TIME)
        sh_camera_frame = bf.sh_camera.getFutureFrames(1, 1).toNumpyArray()
        sh_frame_cube[idx] = sh_camera_frame
    
    
    fname = package_data.shframes_folder() / (ftag+'.fits')
    
    hdr = fits.Header()
    hdr['TEXP_MS']=texp
    fits.writeto(fname, sh_frame_cube, hdr)
    fits.append(fname, zc_vec)
    fits.append(fname,  mode_idex_list)

def load_data(ftag):
    file_name = package_data.shframes_folder() / (ftag+'.fits')
    header = fits.getheader(file_name)
    hduList = fits.open(file_name)
    texp = header['TEXP_MS']
    sh_cube_frames = hduList[0].data
    zc_vec= hduList[1].data
    mode_list = hduList[2].data
    return sh_cube_frames, zc_vec, mode_list, texp