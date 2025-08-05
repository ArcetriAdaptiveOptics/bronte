from bronte.calibration.utils.kl_modal_base_generator import KarhunenLoeveGenerator

def main250805_123100():
    
    ifs_tag = '250805_103600'
    
    klg = KarhunenLoeveGenerator(ifs_tag)
    
    Dtel = 8.2
    r0 = 0.15
    L0 = 25
    
    klg.set_atmo_parameters(Dtel, r0, L0)
    
    z2beIncluded  = 5 
    klg.compute_modal_basis(zern_modes = z2beIncluded, oversampling = 1)
    
    klg.save_kl_modes_as_modal_ifs('250805_123100')
    
    return klg