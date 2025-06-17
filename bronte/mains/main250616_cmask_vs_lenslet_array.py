from bronte.startup import specula_startup
import numpy as np


def main():
    
    subap_tag = '250612_143100'
    sf = specula_startup()
    sf.SUBAPS_TAG = subap_tag
    
    c_vector = np.array([5e-6, 0])
    cmd = sf.slm_rasterizer.m2c(c_vector, applyTiltUnderMask=True)
    wf = sf.slm_rasterizer.reshape_vector2map(cmd)
    
    Npix_per_subap = 26
    subap_size = 5.5e-6 * Npix_per_subap
    
    mag = 250/150
    
    subap_size_on_slm = mag*subap_size
    subap_size_on_slm_in_pix = np.int16(subap_size_on_slm/9.2e-6)
    
    #SLM cmask center
    yc = 579
    xc =  968
    slm_radius = 545
    hh = np.int16(subap_size_on_slm_in_pix / 2)
    wf.mask[:yc-hh,:]= True
    wf.mask[yc+hh:,:] = True
    wf.mask[:,xc-slm_radius+subap_size_on_slm_in_pix:xc+slm_radius-subap_size_on_slm_in_pix] = True
    
    wf = sf.slm_rasterizer.load_a_tilt_under_pupil_mask(wf)
    
    cmd = sf.slm_rasterizer.reshape_map2vector(wf)
    sf.deformable_mirror.set_shape(cmd.data)
    
    sh_fr = sf.sh_camera.getFutureFrames(10).toNumpyArray()
    
    
    return sh_fr, wf