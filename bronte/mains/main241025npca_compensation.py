import numpy as np 
from bronte.ncpa.sharp_psf_on_camera import SharpPsfOnCamera
import time 

def demo():
    
    list_of_zernike2compensate = [4,5,7,11] # noll indexes
    spoc = SharpPsfOnCamera(list_of_zernike2compensate)
    
    # loading zc offset
    zc_offset_array = np.array([0.5e-6, 0.5e-6])
    spoc.load_zc_offset(zc_offset_array)
    
    # selecting roi for psf sharpening
    wfz_offset = spoc._sr.zernike_coefficients_to_raster(spoc.get_zc_offset())
    command = spoc._sr.reshape_map2vector(wfz_offset.toNumpyArray())
    spoc._slm.set_shape(command)
    time.sleep(spoc.SLM_RESPONSE_TIME_SEC)
    
    ima = spoc._cam.getFutureFrames(1,20).toNumpyArray()
    # ... see the image and select the roi
    yc = 200
    xc = 200
    size = 60
    spoc.define_roi(yc, xc, size)
    
    # get dark ... turn off thelaser
    spoc.acquire_master_dark()
    
    
    amp_span = np.linspace(-2e-6, 2e-6, 5)
    
    spoc.sharp(amp_span)
    
    ncpa = spoc.get_ncpa().toNumpyArray()
    