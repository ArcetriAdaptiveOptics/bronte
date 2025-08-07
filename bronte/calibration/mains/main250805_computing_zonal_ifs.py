from bronte.calibration.utils.zonal_influence_function_computer import ZonalInfluenceFunctionComputer
from bronte.package_data import elt_pupil_folder
from bronte.startup import set_data_dir
from bronte.types.slm_pupil_mask_generator import SlmPupilMaskGenerator
def main(
        pupil_diameter_in_pixel,
        Nact_on_diameter,
        ftag,
        custom_mask = None):

    zifc = ZonalInfluenceFunctionComputer(
        pupil_diameter_in_pixel, Nact_on_diameter)
    
    zifc.set_pupil_geometry(obstraction_ratio = 0, diameter_ratio = 1)
    zifc.set_actuators_slaving(slaving_thr = 0.1)
    if custom_mask is not None:
        zifc.load_custom_pupil_mask(custom_mask)
    
    zifc.compute_zonal_ifs()
    zifc.save_ifs(ftag)
    
    return zifc


def main250805_103600():
    
    pupil_diameter_in_pixel = 2*136
    Nact_on_diameter  = 41
    ftag  = '250805_103600'

    zifc = main(pupil_diameter_in_pixel, Nact_on_diameter, ftag)
    
    return zifc

def main_slm_frame_size():
    
    pupil_diameter_in_pixel = 2*545
    Nact_on_diameter  = 41
    ftag  = '250806_000000'
    
    zifc = main(pupil_diameter_in_pixel, Nact_on_diameter, ftag)
    
    return zifc


def main250807_084700():
    '''
    defining zonal ifs of 2x2 (but actualy 6 why) in a elt-like complex pupil
    '''
    puptag = 'EELT480pp0.0803m_obs0.283_spider2023'
    fname = elt_pupil_folder()/(puptag + '.fits')
    
    elt_pupil_mask_idl = 1 - SlmPupilMaskGenerator._get_elt_pupil_from_idl_file_data(fname)
    
    #return elt_pupil_mask_idl
    #pup_size480
    pupil_diameter_in_pixel = elt_pupil_mask_idl.shape[0]
    Nact_on_diameter  = 2
    ftag  = '250807_090200'
    
    zifc = main(pupil_diameter_in_pixel, Nact_on_diameter, ftag, elt_pupil_mask_idl)
    return zifc


def main250807_114200():
    '''
    defining zonal ifs of 41x41 acts in a elt-like complex pupil
    '''
    puptag = 'EELT480pp0.0803m_obs0.283_spider2023'
    fname = elt_pupil_folder()/(puptag + '.fits')
    
    elt_pupil_mask_idl = 1 - SlmPupilMaskGenerator._get_elt_pupil_from_idl_file_data(fname)
    
    #return elt_pupil_mask_idl
    #pup_size480
    pupil_diameter_in_pixel = elt_pupil_mask_idl.shape[0]
    Nact_on_diameter  = 41
    ftag  = '250807_114200'
    
    zifc = main(pupil_diameter_in_pixel, Nact_on_diameter, ftag, elt_pupil_mask_idl)
    return zifc
