

def startup():
    LB_ROOT = '/Users/lbusoni/Library/CloudStorage/GoogleDrive-lorenzo.busoni@inaf.it/.shortcut-targets-by-id/1SPpwbxlHyuuXmzaajup9lXpg_qSHjBX4/phd_slm_edo/misure_tesi_slm/bronte'
    EB_ROOT = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\bronte"

    from bronte.package_data import set_data_root_dir
    set_data_root_dir(EB_ROOT)
    #set_data_root_dir(LB_ROOT)
    from bronte.factories import bronte_factory
    return bronte_factory.BronteFactory()

def specula_startup():
    
    LB_ROOT = '/Users/lbusoni/Library/CloudStorage/GoogleDrive-lorenzo.busoni@inaf.it/.shortcut-targets-by-id/1SPpwbxlHyuuXmzaajup9lXpg_qSHjBX4/phd_slm_edo/misure_tesi_slm/bronte'
    EB_ROOT = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\bronte"

    from bronte.package_data import set_data_root_dir
    set_data_root_dir(EB_ROOT)
    #set_data_root_dir(LB_ROOT)
    from bronte.factories import specula_scao_factory
    return specula_scao_factory.SpeculaScaoFactory()