import numpy as np
from astropy.io import fits

def main(fname_tag = "241211_122600"):
    
    fpath = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\bronte\\temp_trash\\"
    
    steps = np.arange(0,6)
    
    out_pixels_list = []
    
    for step in steps:
        file_name = fpath + fname_tag + f"_frame{step}.fits"
        print("reading file %s frame %d" %(fname_tag, step))
        hduList = fits.open(file_name)
        out_pixels_list.append(hduList[0].data)
        
    return out_pixels_list