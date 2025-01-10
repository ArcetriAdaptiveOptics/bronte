import numpy as np
from arte.types.mask import CircularMask
from bronte.wfs.slm_rasterizer import SlmRasterizer

def display_mode_on_pupil(dm, radius = 576, centerYX = (579, 968)):
    
    amp = 1e-6
    frame_shape = (1152, 1920)
    cmask = CircularMask(
        frameShape = frame_shape,
        maskRadius = radius,
        maskCenter = centerYX
        )
    
    sr = SlmRasterizer(slm_pupil_mask = cmask)
    
    wf = sr.zernike_coefficients_to_raster(np.array([0,0,amp])).toNumpyArray()
    cmd = sr.reshape_map2vector(wf)
    
    dm.set_shape(cmd)