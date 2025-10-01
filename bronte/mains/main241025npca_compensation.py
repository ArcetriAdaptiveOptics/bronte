import numpy as np 
from bronte.ncpa.sharp_psf_on_camera import SharpPsfOnCamera
import time 
from bronte.package_data import other_folder
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
    
    # get dark ... turn off the laser
    spoc.acquire_master_dark()
    
    
    amp_span = np.linspace(-2e-6, 2e-6, 5)
    
    spoc.sharp(amp_span)
    
    ncpa = spoc.get_ncpa().toNumpyArray()


def main251001_100300():
    
    list_of_zernike2compensate = [4,5,6,7,8,9,10,11] # noll indexes
    spoc = SharpPsfOnCamera(list_of_zernike2compensate)
    
    # loading zc offset
    #zc_offset_array = np.array([0.5e-6, 0.5e-6])
    #spoc.load_zc_offset(zc_offset_array)
    flat = np.zeros(1152*1920)
    spoc._factory.deformable_mirror.set_shape(flat)
    
    bkg = spoc._factory.psf_camera_master_bkg
    texp_cam =  spoc._factory._pc_texp
    spoc.load_master_dark(bkg)
    
    yc = 544
    xc = 652
    size = 50
    spoc.define_roi(yc, xc, size)
    
    spoc.sharp( amp_span = np.linspace(-2.5e-6, 2.5e-6, 51), texp_in_ms = texp_cam, Nframe2average = 10)
    
    fname = other_folder() / ('251001_100300.fits')
    
    spoc.save_ncpa(fname)
    
    
def main251001_104600():
    
    list_of_zernike2compensate = [4,5,6,7,8,9,10,11] # noll indexes
    spoc = SharpPsfOnCamera(list_of_zernike2compensate)
    
    # loading zc offset
    zc_offset_array = np.array([ 0.0e+00,  0.0e+00, -2.0e-08,  4.0e-08,  1.5e-08,  1.0e-08,
        1.0e-08,  5.0e-09, -5.0e-09, -5.0e-09])
    #zc_offset_array = np.array([0.5e-6, 0.5e-6])
    spoc.load_zc_offset(zc_offset_array)
    #flat = np.zeros(1152*1920)
    cmd_offset = spoc._sr.m2c(zc_offset_array, True)
    spoc._factory.deformable_mirror.set_shape(cmd_offset)
    
    bkg = spoc._factory.psf_camera_master_bkg
    texp_cam = 5
    spoc.load_master_dark(bkg)
    
    yc = 544
    xc = 652
    size = 50
    spoc.define_roi(yc, xc, size)
    
    spoc.sharp( amp_span = np.linspace(-100e-9, 100e-9, 21), texp_in_ms = texp_cam, Nframe2average = 10)
    
    fname = other_folder() / ('251001_104600.fits')
    
    spoc.save_ncpa(fname)
    
def main251001_114200():
    
    list_of_zernike2compensate = [4,5,6,7,8,9,10,11] # noll indexes
    spoc = SharpPsfOnCamera(list_of_zernike2compensate)
    
    # loading zc offset
    #zc_offset_array = np.array([0.5e-6, 0.5e-6])
    #spoc.load_zc_offset(zc_offset_array)
    flat = np.zeros(1152*1920)
    spoc._factory.deformable_mirror.set_shape(flat)
    
    bkg = spoc._factory.psf_camera_master_bkg
    texp_cam =  spoc._factory._pc_texp
    spoc.load_master_dark(bkg)
    
    yc = 544
    xc = 652
    size = 50
    spoc.define_roi(yc, xc, size)
    
    spoc.sharp( amp_span = np.linspace(-2.5e-6, 2.5e-6, 51), texp_in_ms = texp_cam, Nframe2average = 10, useGaussFit=True)
    
    fname = other_folder() / ('251001_114200.fits')
    
    spoc.save_ncpa(fname)
    

def main251001_123800():
    
    list_of_zernike2compensate = np.arange(12,32) # noll indexes
    spoc = SharpPsfOnCamera(list_of_zernike2compensate)
    
    # loading zc offset
    zc_offset_array = np.array([ 0.0e+00,  0.0e+00, -2.0e-08,  4.0e-08,  1.5e-08,  1.0e-08,
        1.0e-08,  5.0e-09, -5.0e-09, -5.0e-09])
    #zc_offset_array = np.array([0.5e-6, 0.5e-6])
    spoc.load_zc_offset(zc_offset_array)
    #flat = np.zeros(1152*1920)
    cmd_offset = spoc._sr.m2c(zc_offset_array, True)
    spoc._factory.deformable_mirror.set_shape(cmd_offset)
    
    bkg = spoc._factory.psf_camera_master_bkg
    texp_cam = 5
    spoc.load_master_dark(bkg)
    
    yc = 544
    xc = 652
    size = 50
    spoc.define_roi(yc, xc, size)
    
    spoc.sharp( amp_span = np.linspace(-200e-9, 200e-9, 21), texp_in_ms = texp_cam, Nframe2average = 10)
    
    fname = other_folder() / ('251001_123800.fits')
    
    spoc.save_ncpa(fname)