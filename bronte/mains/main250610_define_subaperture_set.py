from astropy.io import fits
from bronte.package_data import shframes_folder, subaperture_set_folder
from bronte.startup import set_data_dir
from bronte.subapertures_initializer import SubapertureGridInitialiser
from bronte.utils.data_cube_cleaner import DataCubeCleaner
import matplotlib.pyplot as plt

def main():
    '''
    this main is meant to define the subpaperture set grid
    when on the slm is applied a zeros on a circular pupil
    of 545 pixel and with a huge tilt in the masked region
    '''
    set_data_dir()
    fname_ext_source = shframes_folder() / '250610_103800.fits'
    fname_laser_source = shframes_folder() / '250610_111300.fits'
    
    dcc = DataCubeCleaner()
    
    data_ext = fits.open(fname_ext_source)
    ext_frame_cube  = data_ext[0].data
    ext_bkg = data_ext[1].data
    red_ext_cube = dcc.get_redCube_from_rawCube(ext_frame_cube, ext_bkg)
    ext_sorce_frame = red_ext_cube.mean(axis=-1)
    
    data_laser = fits.open(fname_laser_source)
    
    laser_frame_cube  = data_laser[0].data
    laser_bkg = data_laser[1].data
    texp = 8 
    # hdr = fits.Header()
    # hdr['TEXP_MS'] = texp
    # filename = shframes_folder() / '250610_152400.fits'
    # fits.writeto(filename, laser_bkg, hdr)
    
    
    red_laser_cube = dcc.get_redCube_from_rawCube( laser_frame_cube, laser_bkg)
    laser_souce_frame = red_laser_cube.mean(axis=-1)
    
    # plt.figure()
    # plt.clf()
    # plt.imshow(ext_sorce_frame)
    # plt.colorbar()
    #
    # plt.figure()
    # plt.clf()
    # plt.imshow(laser_souce_frame)
    # plt.colorbar()
    
    plt.figure()
    plt.clf()
    plt.imshow(ext_sorce_frame-laser_souce_frame, cmap='gray')
    plt.colorbar()
    
    sgi_ext = SubapertureGridInitialiser(ext_sorce_frame, 26, 50)
    sgi_ext.define_subaperture_set(230, 330)
    
    yc  = 887
    xc = 985
    # centering the grid on the central subapertures to minimise offset
    sgi_ext.shift_subaperture_grid_around_central_subap([-5, -7], yc, xc)
    
    #appling the subaperture grind on the working frame
    
    abs_thr = 40
    thr_frame = laser_souce_frame - abs_thr
    thr_frame[thr_frame<0] = 0
    
    sgi = SubapertureGridInitialiser(thr_frame, 26, 50)
    sgi.define_subaperture_set(230, 330)
    sgi.shift_subaperture_grid_around_central_subap([-5, -7], yc, xc)
    sgi.display_flux_and_grid_maps()
    
    sgi.show_subaperture_flux_histogram()
    
    sgi.remove_low_flux_subaperturers(threshold = 1.45e4)
    sgi.display_flux_and_grid_maps()
    
    sgi._subaps.save(subaperture_set_folder()/ "250610_140500.fits", None)