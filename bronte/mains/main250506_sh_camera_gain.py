from bronte import startup
from bronte.package_data import shframes_folder
from astropy.io import fits
import numpy as np 
from bronte.utils.data_cube_cleaner import DataCubeCleaner
from bronte.utils.retry_on_zmqrpc_timeout_error import retry_on_timeout
import matplotlib.pyplot as plt


def acquire_master_bkgs_vs_texp(ftag):
    
    N = 20
    Nframes2average = 20
    #texp_in_ms_vector = np.linspace(0.5, 30, N)
    texp_in_ms_vector = np.linspace(0.05, 1, N)
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
    
    #N = 60
    Nframes2average = 20
    frame_size = 2048
    sf = startup.specula_startup()
    #ftag_bkgs = '250506_122300' # from 0.5 to 30 ms
    ftag_bkgs = '250506_163400' # from 0.05 to 1 ms
    fname_bkgs = shframes_folder() / (ftag_bkgs + '.fits')
    hdulist = fits.open(fname_bkgs)
    
    bkgs = hdulist[0].data
    texp_in_ms_vector = hdulist[1].data
    flat_field_list = []
    std_flat_field_list = []
    dcc = DataCubeCleaner()
    N = len(texp_in_ms_vector)
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
    
    
    #ftag = '250506_154800'# from 0.5 to 30ms
    ftag = '250506_164500' # from 0.05 to 1 ms
    file_name = shframes_folder() /(ftag + '.fits')
    hdulist = fits.open(file_name)
    ffs = hdulist[0].data
    std_ffs = hdulist[1].data
    #texp_vect = hdulist[2].data
    #Nframes = ffs.shape[0]
    y_coords = np.array([1070, 1070, 1070, 1070])
    x_coords = np.array([828, 828+26, 828+26*2, 828+26*3])
    Npoints = len(y_coords)
    plt.figure()
    plt.clf()
    a = 6
    b = a+1
    #Npt = (a+b)**2#9 
     
    for idx in np.arange(Npoints):
        yc = y_coords[idx]
        xc = x_coords[idx]
        signal = ffs[:,yc-a:yc+b ,xc-a:xc+b].mean(axis=(1,2))#sum(axis=(1,2))/Npt
        noise = (std_ffs[:, yc-a:yc+b, xc-a:xc+b]**2).mean(axis=(1,2))#sum(axis=(1,2))/Npt
        #signal = ffs[:, y_coords[idx], x_coords[idx]]
        #noise = std_ffs[:, y_coords[idx], x_coords[idx]]**2
        plt.plot(signal, noise, '.-', label = f"#{idx}")
    plt.xlabel('Signal [ADU]')
    plt.ylabel('Noise [ADU^2]')
    plt.grid('--', alpha = 0.3)
    plt.legend(loc = 'best')
    
    
def display_gain_all():
    
    ftag1 = '250506_164500' # from 0.05 to 1 ms
    ftag2 = '250506_154800'# from 0.5 to 30ms
    
    file_name1 = shframes_folder() /(ftag1 + '.fits')
    file_name2 = shframes_folder() /(ftag2 + '.fits')
    hdulist1 = fits.open(file_name1)
    ffs1 = hdulist1[0].data
    std_ffs1 = hdulist1[1].data
    hdulist2 = fits.open(file_name2)
    ffs2 = hdulist2[0].data
    std_ffs2 = hdulist2[1].data
    #texp_vect = hdulist[2].data
    #Nframes = ffs.shape[0]
    y_coords = np.array([1070, 1070, 1070, 1070])
    x_coords = np.array([828, 828+26, 828+26*2, 828+26*3])
    Npoints = len(y_coords)
    plt.figure()
    plt.clf()
    a = 3
    b = a+1
    N1 = ffs1.shape[0]
    N2 = ffs2.shape[0]
    Ndata = N1+N2
    for idx in np.arange(Npoints):
        
        signal = np.zeros(Ndata-2)
        noise = np.zeros(Ndata-2)
        yc = y_coords[idx]
        xc = x_coords[idx]

        signal1 = ffs1[:,yc-a:yc+b ,xc-a:xc+b].mean(axis=(1,2))
        noise1 = (std_ffs1[:, yc-a:yc+b, xc-a:xc+b]**2).mean(axis=(1,2))
        
        signal2 = ffs2[2:,yc-a:yc+b ,xc-a:xc+b].mean(axis=(1,2))
        noise2 = (std_ffs2[2:, yc-a:yc+b, xc-a:xc+b]**2).mean(axis=(1,2))
        
        signal[:N1] = signal1 
        signal[N1:] = signal2
        noise[:N1] = noise1
        noise[N1:] = noise2
        
        plt.plot(signal, noise, '.-', label = f"#{idx}")
    plt.xlabel('Signal [ADU]')
    plt.ylabel('Noise [ADU^2]')
    plt.grid('--', alpha = 0.3)
    plt.legend(loc = 'best')
    return ffs1, ffs2




def display_gain_all2():
    
    ftag1 = '250506_164500' # from 0.05 to 1 ms
    ftag2 = '250506_154800'# from 0.5 to 30ms
    
    file_name1 = shframes_folder() /(ftag1 + '.fits')
    file_name2 = shframes_folder() /(ftag2 + '.fits')
    hdulist1 = fits.open(file_name1)
    ffs1 = hdulist1[0].data
    std_ffs1 = hdulist1[1].data
    hdulist2 = fits.open(file_name2)
    ffs2 = hdulist2[0].data
    std_ffs2 = hdulist2[1].data
    #texp_vect = hdulist[2].data
    #Nframes = ffs.shape[0]
    Nsuby = 23
    Nsubx = 32
    Npixpersub = 26
    yc = 1071
    xc = 1037
    
    y_rel_coords = np.arange(0,Nsuby)*Npixpersub 
    offset_coordy = np.int32(np.round(np.linspace(0, 6, len(y_rel_coords))))
    y_coords = y_rel_coords + offset_coordy + yc
    x_rel_coords = np.arange(0,Nsubx)*Npixpersub 
    offset_coordx = np.int32(np.round(np.linspace(0, 9, len(x_rel_coords))))
    x_coords = x_rel_coords + offset_coordx + xc
    
    Npoints = len(y_coords)

    a = 5
    b = a+1
    N1 = ffs1.shape[0]
    N2 = ffs2.shape[0]
    Ndata = N1+N2
    
    s = np.zeros(Ndata-2)
    n = np.zeros(Ndata-2)
    
    for k in np.arange(len(s)):
        
        signal_in_subap_list = []
        noise_in_subap_list = []
        
        for y_idx in np.arange(Nsuby):
        
            for x_idx in np.arange(Nsubx):
                
                y = y_coords[y_idx]
                x = x_coords[x_idx]
                
                if k < N1:
                    signal_in_subap = ffs1[k , y - a: y + b , x - a : x +b]
                    noise_in_subap = std_ffs1[k ,y - a : y + b , x - a: x +b]**2
                else:
                    kk = k-N1+2
                    signal_in_subap = ffs2[kk , y - a: y + b , x - a : x +b]
                    noise_in_subap = std_ffs2[kk ,y - a : y + b , x - a: x +b]**2
                
                signal_in_subap_list.append(signal_in_subap)
                noise_in_subap_list.append(noise_in_subap)
        
                
        s[k] = np.mean(np.array(signal_in_subap_list).flatten()) 
        n[k] = np.mean(np.array(noise_in_subap_list).flatten())      
        
    plt.figure()
    plt.clf()
    plt.plot(s, n, '.-')
    plt.xlabel('Signal [ADU]')
    plt.ylabel('Noise [ADU^2]')
    plt.grid('--', alpha = 0.3)
    
    
    xdata = s[19:70]
    ydata = n[19:70]
    
    p = np.polyfit(xdata,ydata,1)
    g = 1/p[0]
    print(f"Gain = {g} e-/ADU in t[1-30ms]")
    
    xdata = s[2:18]
    ydata = n[2:18]
    
    p = np.polyfit(xdata,ydata,1)
    g = 1/p[0]
    print(f"Gain = {g} e-/ADU in t[0.05-1ms]")
    

   