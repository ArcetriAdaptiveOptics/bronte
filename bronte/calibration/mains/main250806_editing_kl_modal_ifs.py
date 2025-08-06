from bronte.calibration.utils.influence_functions_editor import InfluenceFucntionEditor
from bronte.calibration.utils.display_ifs_map import DisplayInfluenceFunctionsMap
import numpy as np
import matplotlib.pyplot as plt

def main(ifs_ftag, Nmodes, new_frame_size, ftag):
    
    ife = InfluenceFucntionEditor(ifs_ftag)
    
    ife.remove_modes(Nmodes)
    ife.rescale_ifs(new_frame_size)
    ife.save_filtered_ifs(ftag)
    
    disp_rescaled_ifs = DisplayInfluenceFunctionsMap(ftag)
    disp_original_ifs = DisplayInfluenceFunctionsMap(ifs_ftag)
    
    return disp_original_ifs, disp_rescaled_ifs
    

def main250806_165000():
    
    ifs_ftag = '250806_110800'
    Nmodes = 10
    new_frame_size = 2*545
    ftag = '250806_165000'
    
    return main(ifs_ftag, Nmodes, new_frame_size, ftag)

def main250806_170800():
    
    ifs_ftag = '250806_110800'
    Nmodes = 200
    new_frame_size = 2*545
    ftag = '250806_170800'
    
    return main(ifs_ftag, Nmodes, new_frame_size, ftag)

def main250806_171900():
    
    ifs_ftag = '250806_115800'
    Nmodes = 200
    new_frame_size = 2*545
    ftag = '250806_171900'
    
    return main(ifs_ftag, Nmodes, new_frame_size, ftag)

def main_250806_173200():
    
    rescaled_ifs_tag = '250806_171900'
    original_ifs_tag = '250806_115800'
    
    disp_rescaled_ifs = DisplayInfluenceFunctionsMap(rescaled_ifs_tag)
    disp_original_ifs = DisplayInfluenceFunctionsMap(original_ifs_tag)
    Nmodes  = 200
    ptv_original = np.zeros(Nmodes)
    ptv_rescaled = np.zeros(Nmodes)
    amp_original = np.zeros(Nmodes)
    amp_rescaled = np.zeros(Nmodes)
    
    for idx in range(Nmodes):
        
        original_mode = disp_original_ifs.get_if_2Dmap(idx)
        rescaled_mode = disp_rescaled_ifs.get_if_2Dmap(idx) 
        
        ptv_original[idx] = np.ptp(original_mode)
        ptv_rescaled[idx] = np.ptp(rescaled_mode)
        amp_original[idx] = original_mode[original_mode.mask==False].std()
        amp_rescaled[idx] = rescaled_mode[rescaled_mode.mask==False].std()
        
    plt.figure()
    plt.clf()
    plt.plot(ptv_original, '.-', label = 'Original')
    plt.plot(ptv_rescaled, '.-', label = 'Rescaled')
    plt.grid('--', alpha = 0.3)
    plt.xlabel('KL mode index')
    plt.ylabel('PTV')
    plt.legend(loc='best')
    
    plt.figure()
    plt.clf()
    plt.plot(amp_original, '.-', label = 'Original')
    plt.plot(amp_rescaled, '.-', label = 'Rescaled')
    plt.grid('--', alpha = 0.3)
    plt.xlabel('KL mode index')
    plt.ylabel('mode std')
    plt.legend(loc='best')
    
    print('PTV relative error std:')
    print(((ptv_original-ptv_rescaled)/ptv_original).std())
    print('amp relative error std')
    print(((amp_original - amp_rescaled)/amp_original).std())
    
    
    