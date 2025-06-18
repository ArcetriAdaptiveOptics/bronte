import numpy as np
from arte.types.mask import CircularMask
from bronte.startup import set_data_dir
from bronte.package_data import reconstructor_folder
from astropy.io import fits
from arte.utils.modal_decomposer import ModalDecomposer
#from arte.utils.zernike_decomposer import ZernikeModalDecomposer
from arte.utils.zernike_generator import ZernikeGenerator
from arte.utils.rebin import rebin
from arte.types.slopes import Slopes

def get_synth_intmat():
    set_data_dir()
    #slm_shape = (1152, 1920)
    #yc = 579
    #xc =  968
    slm_pix_size = 9.2e-6
    slm_radius = 545*slm_pix_size
    #Nmodes = 200
    intmat_tag = '250616_103300'
    calib_tag = '_bronte_calib_config'
    file_name = reconstructor_folder() / (intmat_tag + calib_tag + '.fits')
    config_data = fits.open(file_name)
    pp_vect_in_nm = config_data[0].data
    Nmodes = len(pp_vect_in_nm)
    j_noll_vector =np.arange(Nmodes)+2
    
    frame_size = 460
    sh_frame_shape = (frame_size, frame_size)
    yc = 230
    xc = 230
    radius =  210 
    fla = 8.31477e-3
    subap_size = 144e-6
    # Npix_per_sub = 26
    
    cmask = CircularMask(
        frameShape = sh_frame_shape,
        maskRadius = radius,
        maskCenter = (yc, xc))
    
    zg = ZernikeGenerator(cmask)
    
    Nsubap = len(cmask.mask()[cmask.mask() == False])
    #synth_intmat = np.zeros((Nsubap*2, Nmodes))
    synth_intmat_list = []
    cc  = CircularMask(
        frameShape = (46,46),
        maskRadius = 21,
        maskCenter = (23, 23))
    
    pp_norm = (pp_vect_in_nm - pp_vect_in_nm.min())/(pp_vect_in_nm.max() - pp_vect_in_nm.min())
    
    for idx, j in enumerate(j_noll_vector):
        
        
        slope_x = zg.getDerivativeX(np.int16(j))*pp_norm[idx]*fla/(subap_size*0.5)#/pp_norm[idx]#*0.5#*1e-9*2/subap_size
        slope_y = zg.getDerivativeY(np.int16(j))*pp_norm[idx]*fla/(subap_size*0.5)#/pp_norm[idx]#*0.5#*1e-9*2/subap_size
        
        slope_x_reb = rebin(slope_x.data,(46,46))
        slope_y_reb = rebin(slope_y.data,(46,46))
        
        #print(f"{slope_x_reb.shape}")
        #print(f"{slope_y_reb.shape}")
        
        slope_x_vector = np.reshape(np.ma.array(data = slope_x_reb, mask = cc.mask()), (46*46))
        slope_y_vector = np.reshape(np.ma.array(data = slope_y_reb, mask = cc.mask()), (46*46))

        #print(f"{slope_x_vector.shape}")
        #print(f"{slope_y_vector.shape}")
        
        sx = slope_x_vector[slope_x_vector.mask == False]
        sy = slope_y_vector[slope_y_vector.mask == False]
        nsub_reb = len(sx)
        slope_vector = np.zeros(nsub_reb*2)
        slope_vector[:len(sx)] = sx
        slope_vector[len(sx):] = sy
        
        synth_intmat_list.append(slope_vector) 
        
    return np.array(synth_intmat_list).T

def check_shwfs_sampling(j):
    
    import matplotlib.pyplot as plt
    
    set_data_dir()
    intmat_tag = '250616_103300'
    calib_tag = '_bronte_calib_config'
    file_name = reconstructor_folder() / (intmat_tag + calib_tag + '.fits')
    config_data = fits.open(file_name)
    pp_vect_in_nm = config_data[0].data
    Nmodes = len(pp_vect_in_nm)

    frame_size = 460
    frame_shape = (frame_size, frame_size)
    yc = 230
    xc = 230
    radius =  210 
    subap_size = 10
    
    cmask = CircularMask(
    frameShape = frame_shape,
    maskRadius = radius,
    maskCenter = (yc, xc))
    
    zg = ZernikeGenerator(cmask)
    
    wf = zg.getZernike(j)#* pp_vect_in_nm[j-2]
    
    plt.figure()
    plt.clf()
    plt.imshow(wf, cmap='RdBu_r')
    plt.colorbar(label='au')
    
    for y in range(0, frame_size, subap_size):
        plt.hlines(y, 0, frame_size, colors='k', linewidth=0.5)
    for x in range(0, frame_size, subap_size):
        plt.vlines(x, 0, frame_size, colors='k', linewidth=0.5)
    
    plt.title(f'Zernike mode {j} with subap grid (size = {subap_size}px)')
    plt.xlabel('X [pixel]')
    plt.ylabel('Y [pixel]')
    

    gx = zg.getDerivativeX(j)
    gy = zg.getDerivativeY(j)
    
    # simulating WFS sampling
    #slope downsample for each subap
    #slope mean for each subap
    slope_x = gx.reshape((frame_size // subap_size, subap_size,
                          frame_size // subap_size, subap_size)).mean(axis=(1, 3))
    
    slope_y = gy.reshape((frame_size // subap_size, subap_size,
                          frame_size // subap_size, subap_size)).mean(axis=(1, 3))
    
    
    plt.subplots(1,2,sharex=True, sharey=True)
    plt.subplot(1,2,1)
    v_min = np.min((slope_x.min(),slope_y.min()))
    v_max = np.max((slope_x.max(),slope_y.max()))
    plt.imshow(slope_x, vmin = v_min, vmax = v_max, cmap='RdBu_r')
    plt.title('Slope X')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(slope_y, vmin = v_min, vmax = v_max, cmap='RdBu_r')
    plt.title('Slope Y')
    plt.colorbar()
    
    md = ModalDecomposer(Nmodes)

    arte_slopes = Slopes.fromNumpyArray(slope_x, slope_y)
    sh_mask = CircularMask(
        frameShape = (46,46),
        maskRadius = 21,
        maskCenter = (23, 23))              
    zc = md.measureZernikeCoefficientsFromSlopes(arte_slopes, sh_mask)
    wf_rec = md.recomposeWavefrontFromModalCoefficients(zc, cmask)
    wf_recon = wf_rec.toNumpyArray()
    
    wf_diff = np.ma.array(wf - wf_recon, mask = cmask.mask())

    

    plt.subplots(1,3, sharex = True, sharey = True)
    plt.subplot(1,3,1) 
    plt.imshow(wf, cmap='RdBu_r')
    plt.title('Original WF')
    plt.colorbar()
    
    plt.subplot(1,3,2)
    plt.imshow(np.ma.array(data =wf_recon, mask = cmask.mask()), cmap='RdBu_r')
    plt.title('Reconstructed from slopes')
    plt.colorbar()
    
    plt.subplot(1,3,3)
    plt.imshow(wf_diff, cmap='RdBu_r')
    plt.title('Difference')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
  
    return wf_diff, md, zc
 



    