import numpy as np 
from astropy.io import fits
from bronte.startup import set_data_dir
from bronte.subapertures_initializer import SubapertureGridInitialiser
from bronte.package_data import shframes_folder
from bronte.package_data import subaperture_set_folder 

def main():
    
    set_data_dir()
    fname_ext_source = shframes_folder() / '250110_170300.fits'
    fname_laser_source = shframes_folder() / '250117_103000.fits'
    
    data_ext = fits.open(fname_ext_source)
    wf_ref = data_ext[0].data
    
    data_laser = fits.open(fname_laser_source)
    ima = data_laser[0].data
    
    sgi_ext = SubapertureGridInitialiser(wf_ref, 26, 50)
    sgi_ext.define_subaperture_set(230, 330)
    #coordinate of the central subaperture in shwfs frame in 250117_103000.fits
    yc = 861
    xc = 959
    # centering the grid on the central subapertures to minimise offset
    sgi_ext.shift_subaperture_grid_around_central_subap([-6, -7], yc, xc)
    
    #applying the same subaperture grid to a shwfs frame 
    sgi = SubapertureGridInitialiser(ima, 26, 50)
    sgi.define_subaperture_set(230, 330)
    sgi.shift_subaperture_grid_around_central_subap([-6, -7], yc, xc)
    
    sgi.display_flux_and_grid_maps()
    
    #removing subaps with low flux by thresholding
    
    sgi.remove_low_flux_subaperturers(threshold = 9.5e4)
    sgi.display_flux_and_grid_maps()
    
    #sgi._subaps.save(subaperture_set_folder()/ "fname.fits", None)
    
    return sgi