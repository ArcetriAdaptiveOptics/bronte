from bronte.calibration.utils.influence_functions_editor import InfluenceFucntionEditor
from bronte.calibration.utils.display_ifs_map import DisplayInfluenceFunctionsMap

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