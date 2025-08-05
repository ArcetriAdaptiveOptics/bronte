from bronte.calibration.utils.zonal_influence_function_computer import ZonalInfluenceFunctionComputer

def main250805_103600():
    
    pupil_diameter_in_pixel = 2*136#2*545
    Nact_on_diameter  = 41
    
    zifc = ZonalInfluenceFunctionComputer(
        pupil_diameter_in_pixel, Nact_on_diameter)
    
    zifc.set_pupil_geometry(obstraction_ratio = 0, diameter_ratio = 1)
    zifc.set_actuators_slaving(slaving_thr = 0.1)
    
    zifc.compute_zonal_ifs()
    ftag  = '250805_103600'
    zifc.save_ifs(ftag)
    
    return zifc