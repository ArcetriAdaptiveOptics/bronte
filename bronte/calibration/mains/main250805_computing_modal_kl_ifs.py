from bronte.calibration.utils.kl_modal_base_computer import KarhunenLoeveGenerator
import numpy as np


def main(ifs_tag, Dtel, r0, L0, ftag):
    
    klg = KarhunenLoeveGenerator(ifs_tag)
    klg.set_atmo_parameters(Dtel, r0, L0)
    z2beIncluded  = 5 
    klg.compute_modal_basis(zern_modes = z2beIncluded, oversampling = 1)
    klg.save_kl_modes_as_modal_ifs(ftag)
    
    return klg

def main250805_123100():
    
    ifs_tag = '250805_103600'
    Dtel = 8.2
    r0 = 0.15
    L0 = 25
    ftag = '250805_123100'
    
    klg = main(ifs_tag, Dtel, r0, L0, ftag)
    
    return klg

def main250805_154500():
    
    ifs_tag = '250805_103600'
    Dtel = 8
    seeing = 0.5 # arcsec
    wl  = 633e-9
    r0 = (wl/seeing)*(180/np.pi)*60*60
    L0 = 40
    ftag = '250805_154500'
    
    klg = main(ifs_tag, Dtel, r0, L0, ftag)
    
    return klg

def main250806_110800():
    
    ifs_tag = '250805_103600'
    Dtel = 8.2
    r0 = 0.15
    L0 = 25
    ftag = '250806_110800'
    
    klg = main(ifs_tag, Dtel, r0, L0, ftag)
    
    return klg

def main250806_115800():
    
    ifs_tag = '250805_103600'
    Dtel = 8
    seeing = 0.5 # arcsec
    wl  = 633e-9
    r0 = (wl/seeing)*(180/np.pi)*60*60
    L0 = 40
    ftag = '250806_115800'
    
    klg = main(ifs_tag, Dtel, r0, L0, ftag)
    
    return klg


def main250807_100300():
    '''
    computing kl modal ifs on complex elt pupil example
    '''
    
    ifs_tag = '250807_090200'
    Dtel = 8
    seeing = 0.5 # arcsec
    wl  = 633e-9
    r0 = (wl/seeing)*(180/np.pi)*60*60
    L0 = 40
    ftag = '250807_100300'
    
    klg = main(ifs_tag, Dtel, r0, L0, ftag)
    
    return klg


def main250807_121900():
    '''
    computing kl modal ifs on complex elt pupil with 41x41 acts
    '''
    
    ifs_tag = '250807_114200'
    Dtel = 8
    seeing = 0.5 # arcsec
    wl  = 633e-9
    r0 = (wl/seeing)*(180/np.pi)*60*60
    L0 = 40
    ftag = '250807_121900'
    
    klg = main(ifs_tag, Dtel, r0, L0, ftag)
    
    return klg