from bronte.startup import set_data_dir
from bronte.startup import specula_startup
from bronte.package_data import shframes_folder, other_folder
from astropy.io import fits
from bronte.utils.retry_on_zmqrpc_timeout_error import retry_on_timeout
import numpy as np

def main(ftag):
    
    N = 25
    texp_vect_ms = np.linspace(1, N, 25)
    frames_list =[]
    factory = specula_startup()
    factory.deformable_mirror.set_shape(np.zeros(1920*1152))
    step = 0
    for texp in texp_vect_ms:
        step+=1
        print(f"+ Step {step}/{N}")
        factory.sh_camera.setExposureTime(int(texp))
        frame =  factory.sh_camera.getFutureFrames(1,20).toNumpyArray()
        frames_list.append(frame)
    
    file_name = shframes_folder() / (ftag + '.fits')
   
    fits.writeto(file_name, np.array(frames_list), None)
    fits.append(file_name, texp_vect_ms)


def analyse(ftag):
    
    set_data_dir()
    file_name = shframes_folder() / (ftag + '.fits')
    hduList = fits.open(file_name)
    shframes_cube = hduList[0].data
    texp_vector = hduList[1].data
    return shframes_cube, texp_vector


def measure_detector_linearity(ftag):
    
    Nsubap = 4
    N = 30
    subap_coord_x = np.array([1011, 1114, 957, 879])
    subap_coord_y = np.array([600, 993, 967, 1097])
    i_mean_per_sub = np.zeros((Nsubap, N))
    err_i_per_sub = np.zeros((Nsubap, N))
    
    
    Nframes = 100
    texp_vect_ms = np.linspace(1, N, N)
    
    factory = specula_startup()
    factory.deformable_mirror.set_shape(np.zeros(1920*1152))
    step = 0
    
    for idx, texp in enumerate(texp_vect_ms):
        step+=1
        print(f"+ Step {step}/{N}")
        factory.sh_camera.setExposureTime(int(texp), 50)
        frame =  retry_on_timeout(lambda:factory.sh_camera.getFutureFrames(Nframes))
        frame = frame.toNumpyArray()
        mean_frame = frame.mean(axis=-1)
        std_frame = frame.std(axis=-1)
        for k in range(Nsubap):
            i_mean_per_sub[idx,k] = mean_frame[subap_coord_y[k], subap_coord_x[k]]
            err_i_per_sub[idx,k] = std_frame[subap_coord_y[k], subap_coord_x[k]]
    
    file_name = other_folder() / (ftag + '.fits')
   
    fits.writeto(file_name, i_mean_per_sub, None)
    fits.append(file_name, err_i_per_sub)
    fits.append(file_name, texp_vect_ms)
    fits.append(file_name, subap_coord_y)
    fits.append(file_name, subap_coord_x)
