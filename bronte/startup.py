
#TODO: convert as a StartupManager class

def set_data_dir():
    LB_ROOT = '/Users/lbusoni/Library/CloudStorage/GoogleDrive-lorenzo.busoni@inaf.it/.shortcut-targets-by-id/1SPpwbxlHyuuXmzaajup9lXpg_qSHjBX4/phd_slm_edo/misure_tesi_slm/bronte'
    EB_ROOT = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\bronte"
    from bronte.package_data import set_data_root_dir
    set_data_root_dir(EB_ROOT)
    #set_data_root_dir(LB_ROOT)
    
def startup():
    set_data_dir()
    from bronte.factories import bronte_factory
    return bronte_factory.BronteFactory()

def flattening_startup():
    set_data_dir()
    from bronte.factories import flattening_factory
    return flattening_factory.SpeculaFlatteningFactory()

def specula_startup():
    set_data_dir()
    from bronte.factories import specula_scao_factory
    return specula_scao_factory.SpeculaScaoFactory()

def synthetic_calibration_startup():
    set_data_dir()
    from bronte.factories import synthetic_calibration_factory
    return synthetic_calibration_factory.SyntheticCalibrationFactory()

def measured_calibration_startup():
    set_data_dir()
    from bronte.factories import measured_calibration_factory
    return measured_calibration_factory.MeasuredCalibrationFactory()