from astropy.io import fits
from bronte.startup import set_data_dir
from bronte.package_data import shframes_folder, modal_offsets_folder
import numpy as np 
from bronte.utils.slopes_vector_analyser import SlopesVectorAnalyser
import matplotlib.pyplot as plt


def check_frame_after_thr():
    '''
    this main is meant to check the who the frame looks like
    after thresholding
    '''
    set_data_dir()
    # loading TT scan data
    ftag ='250512_102100'# Z2 data
    file_name = shframes_folder() / (ftag + '.fits')
    hdr = fits.getheader(file_name)
    #jnoll_index = hdr['NOLL_J']
    hdl = fits.open(file_name)
    frame_cube = hdl[0].data 
    #Nframes = frame_cube.shape[0]
    ref_frame = hdl[1].data
    ref_frame[ref_frame<0] = 0
    c_vector = hdl[2].data
    
    subap_tag = '250120_122000'
    sva = SlopesVectorAnalyser(subap_tag)
    sva.reload_slope_pc(pix_thr_ratio = 0.18, abs_pix_thr = 0)
    #Nsubap = sva._subapertures_set.n_subaps
    #NpixperSub = sva._subapertures_set.np_sub
    
    
    plt.figure()
    plt.clf()
    plt.title('Reference frame before thr')
    fr2disp = ref_frame + 1000*sva._subaperture_grid_map
    plt.imshow(fr2disp[840:955, 940:1054])
    plt.colorbar()
    
    thr_abs = 200
    sva.reload_slope_pc(pix_thr_ratio = 0, abs_pix_thr = thr_abs)
    s = sva.get_slopes_from_frame(ref_frame)
    fr_abs_thr = sva.get_frame_after_thresholding()
    plt.figure()
    plt.clf()
    plt.title(f'Reference frame after abs thr = {thr_abs}')
    fr2disp = fr_abs_thr + 1000*sva._subaperture_grid_map
    plt.imshow(fr2disp[840:955, 940:1054])
    plt.colorbar()
    
    thr_ratio = 0.18
    sva.reload_slope_pc(pix_thr_ratio = thr_ratio, abs_pix_thr = 0)
    s = sva.get_slopes_from_frame(ref_frame)
    fr_thr_ratio = sva.get_frame_after_thresholding()
    plt.figure()
    plt.clf()
    plt.title(f'Reference frame after thr_ratio = {thr_ratio}')
    fr2disp = fr_thr_ratio + 1000*sva._subaperture_grid_map
    plt.imshow(fr2disp[840:955, 940:1054])
    plt.colorbar()
    
    return sva
    

def check_spot_displacement():
    
    set_data_dir()
    # loading TT scan data
    ftag ='250512_102100'# Z2 data
    file_name = shframes_folder() / (ftag + '.fits')
    hdr = fits.getheader(file_name)
    #jnoll_index = hdr['NOLL_J']
    hdl = fits.open(file_name)
    frame_cube = hdl[0].data 
    #Nframes = frame_cube.shape[0]
    ref_frame = hdl[1].data
    ref_frame[ref_frame<0] = 0
    c_vector = hdl[2].data
    
    subap_tag = '250120_122000'
    sva = SlopesVectorAnalyser(subap_tag)
    sva.reload_slope_pc(pix_thr_ratio = 0.18, abs_pix_thr = 0)
    
    