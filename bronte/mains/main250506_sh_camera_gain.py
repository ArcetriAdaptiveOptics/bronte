from bronte import startup
from bronte.package_data import shframes_folder
from astropy.io import fits
import numpy as np 
from bronte.utils.data_cube_cleaner import DataCubeCleaner
from bronte.utils.retry_on_zmqrpc_timeout_error import retry_on_timeout
import matplotlib.pyplot as plt


def acquire_master_bkgs_vs_texp(ftag):
    
    N = 60
    Nframes2average = 20
    texp_in_ms_vector = np.linspace(0.5, 30, N)
    sf = startup.specula_startup()
    bkgs_list = []
    for idx, texp in enumerate(texp_in_ms_vector):
        
        print(f"Step{idx+1}/{N}: acquiring bkg at {texp} ms")
        sf.sh_camera.setExposureTime(texp)
        raw_cube = sf.sh_camera.getFutureFrames(Nframes2average).toNumpyArray()
        master_bkg = np.median(raw_cube, axis=-1)
        bkgs_list.append(master_bkg)
    
    file_name = shframes_folder() /(ftag + '.fits')
    hdr = fits.Header()
    hdr['AV_FR'] = Nframes2average
    fits.writeto(file_name, np.array(bkgs_list), hdr)
    fits.append(file_name, texp_in_ms_vector)


def acquire_flat_fields(ftag):
    
    N = 60
    Nframes2average = 50#100
    frame_size = 2048
    sf = startup.specula_startup()
    
    fname_bkgs = shframes_folder() / ('250506_122300.fits')
    hdulist = fits.open(fname_bkgs)
    
    bkgs = hdulist[0].data
    texp_in_ms_vector = hdulist[1].data
    flat_field_list = []
    std_flat_field_list = []
    dcc = DataCubeCleaner()
    #texp_in_ms_vector = np.linspace(0.5, 30, N)
    
    for idx, texp in enumerate(texp_in_ms_vector):
        
        print(f"Step{idx+1}/{N}: acquiring flat fields at {texp} ms")
        sf.sh_camera.setExposureTime(texp)
        raw_cube = np.zeros((frame_size,frame_size,Nframes2average))
        for k in np.arange(Nframes2average):
            #print(f"frame {k+1}/{Nframes2average}")
            raw_cube[:,:,k] = retry_on_timeout(lambda:sf.sh_camera.getFutureFrames(1).toNumpyArray())
        red_cube = dcc.get_redCube_from_rawCube(raw_cube, bkgs[idx])
        flat_field = np.mean(red_cube, axis=-1)
        std_flat_field = np.std(red_cube, axis=-1)
        flat_field_list.append(flat_field)
        std_flat_field_list.append(std_flat_field)
    
    file_name = shframes_folder() /(ftag + '.fits')
    hdr = fits.Header()
    hdr['AV_FR'] = Nframes2average
    fits.writeto(file_name, np.array(flat_field_list), hdr)
    fits.append(file_name, np.array(std_flat_field_list))
    fits.append(file_name, texp_in_ms_vector)
    

def display_gain():
    
    
    ftag = '250506_152500'
    file_name = shframes_folder() /(ftag + '.fits')
    hdulist = fits.open(file_name)
    ffs = hdulist[0].data
    std_ffs = hdulist[1].data
    #texp_vect = hdulist[2].data
    
    y_coords = np.array([])
    x_coords = np.array([])
    Npoints = len(y_coords)
    plt.figure()
    plt.clf()
    for idx in np.arange(Npoints):
        
        signal = ffs[y_coords[idx],x_coords[idx]]
        noise = std_ffs[y_coords[idx],x_coords[idx]]**2
    
    plt.plot(signal, noise, '.-', label = f"#{idx}")
    plt.xlabel('Signal [ADU]')
    plt.ylabel('Noise [ADU^2]')
    plt.grid('--', alpha = 0.3)