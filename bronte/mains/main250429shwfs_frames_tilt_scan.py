from bronte import startup
from bronte.package_data import shframes_folder, modal_offsets_folder
import numpy as np 
from astropy.io import fits
from bronte.utils.retry_on_zmqrpc_timeout_error import retry_on_timeout
import time
from bronte.utils.data_cube_cleaner import DataCubeCleaner



def main(ftag, jnoll_mode=2, addOffset = False):
    '''
    Tilt scan:
    Acquires sh frames relative to tilts of differnet amplitude
    '''
    
    SLEEP_TIME_IN_SEC = 0.005
    
    factory = startup.specula_startup()
    factory.SH_FRAMES2AVERAGE = 10
    Nframes = factory.SH_FRAMES2AVERAGE
    flat = np.zeros(1920*1152)
    
    offset_cmd = 0
    hdr_offset = 'NA'
    if addOffset is True:
        off_tag = '250509_170000'#'250509_161700'
        offset_fname = modal_offsets_folder() / (off_tag+'.fits')
        hdl = fits.open(offset_fname)
        offset = hdl[0].data
        offset_cmd = - offset#factory.slm_rasterizer.m2c(modal_offset)
        hdr_offset = off_tag
        
    factory.deformable_mirror.set_shape(flat + offset_cmd)
    texp_ms = retry_on_timeout(lambda: factory.sh_camera.exposureTime())
    time.sleep(SLEEP_TIME_IN_SEC)
    ref_frame = factory.sh_camera.getFutureFrames(1).toNumpyArray() - factory.sh_camera_master_bkg
    
    Nscans = 61
    frame_size = 2048
    frame_cube = np.zeros((Nscans, frame_size, frame_size))
    coef_scan_vector = np.linspace(-30, 30, Nscans) * 1e-6 
    
    modal_cmd = np.zeros(3)
    mode_index2be_scan = jnoll_mode-2
    dcc = DataCubeCleaner()
    for idx, amp in enumerate(coef_scan_vector):
        print(f"+ Scan {idx+1}/{Nscans}")
        modal_cmd[mode_index2be_scan] = amp
        print(f"applying c = {amp} m rms wf")
        cmd = factory.slm_rasterizer.m2c(modal_cmd)
        factory.deformable_mirror.set_shape(cmd + offset_cmd)
        time.sleep(SLEEP_TIME_IN_SEC)
        raw_cube = factory.sh_camera.getFutureFrames(Nframes).toNumpyArray()
        frame_cube[idx] = dcc.get_mean_from_rawCube(raw_cube[:,:, 2:], factory.sh_camera_master_bkg)
    
    file_name = shframes_folder()/(ftag + '.fits')
    hdr = fits.Header()
    hdr['NOLL_J'] =  jnoll_mode
    hdr['TEXP'] = texp_ms
    hdr['TSLEEP'] = SLEEP_TIME_IN_SEC
    hdr['OFFSET'] = hdr_offset
    fits.writeto(file_name, frame_cube, hdr)
    fits.append(file_name, ref_frame)
    fits.append(file_name, coef_scan_vector)
    

def load(ftag):
    
    startup.set_data_dir()
    file_name = shframes_folder() / (ftag + '.fits')
    hdr = fits.getheader(file_name)
    hudlist = fits.open(file_name)
    frame_cube =  hudlist[0].data
    ref_frame =  hudlist[1].data
    coef_vector =  hudlist[2].data
    return hdr, frame_cube, ref_frame, coef_vector



def main250430(ftag):
    '''
    Measurements for Visual check of the actuation of the slm command:
    for each applied command on the slm it takes 10 frames
    few of these acquires the frame when the slm is still 
    applying the command 
    '''
    
    SLM_TIME_RISE_SEC = 0.005
    factory = startup.specula_startup()

    texp_ms = retry_on_timeout(lambda: factory.sh_camera.exposureTime())
    fps = retry_on_timeout(lambda: factory.sh_camera.getFrameRate())
    
    cmd3 = factory.slm_rasterizer.m2c(np.array([-30e-6, 0, 0]))
    cmd2 = factory.slm_rasterizer.m2c(np.array([-10e-6, 0, 0]))
    cmd1 = factory.slm_rasterizer.m2c(np.array([-1e-6, 0, 0]))
    cmd0 = np.zeros(1920*1152)
    
    factory.deformable_mirror.set_shape(cmd3)
    time.sleep(SLM_TIME_RISE_SEC)
    fr3 = factory.sh_camera.getFutureFrames(10).toNumpyArray()
    
    factory.deformable_mirror.set_shape(cmd2)
    time.sleep(SLM_TIME_RISE_SEC)
    fr2 = factory.sh_camera.getFutureFrames(10).toNumpyArray()
    
    factory.deformable_mirror.set_shape(cmd1)
    time.sleep(SLM_TIME_RISE_SEC)
    fr1 = factory.sh_camera.getFutureFrames(10).toNumpyArray()
    
    factory.deformable_mirror.set_shape(cmd0)
    time.sleep(SLM_TIME_RISE_SEC)
    fr0 = factory.sh_camera.getFutureFrames(10).toNumpyArray()
    
    file_name = shframes_folder()/(ftag + '.fits')
    hdr = fits.Header()
    hdr['NOLL_J'] =  2
    hdr['TEXP'] = texp_ms
    hdr['FPS'] = fps
    hdr['TSLEEP'] = SLM_TIME_RISE_SEC
    fits.writeto(file_name, fr3, hdr)
    fits.append(file_name, fr2)
    fits.append(file_name, fr1)
    fits.append(file_name, fr0)
    fits.append(file_name, np.array([-30e-6, -10e-6, -1e-6, 0.]))
    
    
def load_main250430(ftag):

    startup.set_data_dir()
    file_name = shframes_folder() / (ftag + '.fits')
    hdr = fits.getheader(file_name)
    hudlist = fits.open(file_name)
    fr3 = hudlist[0].data
    fr2 = hudlist[1].data
    fr1 = hudlist[2].data
    fr0 = hudlist[3].data
    modal_cmd = hudlist[4].data
    return hdr, fr3, fr2, fr1, fr0, modal_cmd

def display_main250430():
    '''
    Visual check of the actuation of the slm command:
    for each applied command on the slm it takes 10 frames
    few of these acquires the frame when the slm is still 
    applying the command 
    '''
    
    hdr, fr3, fr2, fr1, fr0, modal_cmd = load_main250430('250430_122700')
    
    ntime = 5
    ncmds = 4 
    
    frames3 = fr3[230:1460,300:1670,:5]
    frames2 = fr2[230:1460,300:1670,:5]
    frames1 = fr1[230:1460,300:1670,:5]
    frames0 = fr0[230:1460,300:1670,:5]
    
    frs = [frames3, frames2, frames1, frames0]
    
    disp_shape = (4*1230, 5*1370)
    
    full_map = np.zeros(disp_shape)
    
    for j in np.arange(ntime):
        for i in np.arange(ncmds):
            full_map[i*1230:(i+1)*1230, j*1370:(j+1)*1370] = frs[i][:,:,j]
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.clf()
    plt.imshow(full_map)
    plt.colorbar()
        
    
    
    
    
    