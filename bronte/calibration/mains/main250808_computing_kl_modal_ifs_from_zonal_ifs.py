from bronte.calibration.mains import main250805_computing_zonal_ifs
from bronte.calibration.mains import main250805_computing_modal_kl_ifs
from bronte.calibration.mains import main250806_editing_kl_modal_ifs
import numpy as np

def main(
    pupil_diameter_in_pixel,
    Nact_on_diameter,
    zifs_ftag,
    custom_mask,
    Dtel,
    r0,
    L0,
    mifs_ftag,
    Nmodes,
    new_frame_size,
    edited_mifs_ftag):
    '''
    computes the kl modal ifs scaled to slm pupil frame
    from the definition of zonal ifs defined in a smaller
    pupil frame, then comuputes the correspondi kl modal ifs,
    finaly selects the wanted modes for the wanted modal bases 
    and scales it to the slm pupil frame 
    '''
    zifc = main250805_computing_zonal_ifs.main(
        pupil_diameter_in_pixel,
        Nact_on_diameter,
        zifs_ftag,
        custom_mask)
    
    klc = main250805_computing_modal_kl_ifs.main(
        zifs_ftag, Dtel, r0, L0, mifs_ftag)
    
    _,_ = main250806_editing_kl_modal_ifs.main(
        mifs_ftag, Nmodes, new_frame_size, edited_mifs_ftag)
    
def main25808_092600():
    '''
    computing 200 kl on the slm pupil frame
    starting from 41x41 actuators on a circular pupil 
    of 480x480 pixel,L0=40, 8m of diameter, seeing 0.5arcsec, wl=633nm
    '''
    pupil_diameter_in_pixel = 480
    Nact_on_diameter  = 41
    zifs_ftag  = '250808_092600'
    custom_mask = None
    
    Dtel = 8
    seeing = 0.5 # arcsec
    wl  = 633e-9
    r0 = (wl/seeing)*(180/np.pi)*60*60
    L0 = 40
    mifs_ftag = '250808_092601'
    
    Nmodes = 200
    new_frame_size = 2*545
    edited_mifs_ftag = '250808_092602'
    
    main(pupil_diameter_in_pixel, Nact_on_diameter, zifs_ftag, custom_mask,
          Dtel, r0, L0, mifs_ftag,
           Nmodes, new_frame_size, edited_mifs_ftag)
    
def main250808_100000():
    '''
    computing 200 kl on the slm pupil frame
    starting zifs(250808_092600) on a circular pupil 
    of 480x480 pixel, 8m.2 of diameter, r0=0.15m,L0=25m
    '''
    zifs_ftag  = '250808_092600'
    Dtel = 8.2
    r0 = 0.15
    L0 = 25
    mifs_ftag = '250808_1000XX'
    
    Nmodes = 200
    new_frame_size = 2*545
    edited_mifs_ftag = '250808_1000XX'
    
    klc = main250805_computing_modal_kl_ifs(
        zifs_ftag, Dtel, r0, L0, mifs_ftag)
    
    _,_ = main250806_editing_kl_modal_ifs.main(
        mifs_ftag, Nmodes, new_frame_size, edited_mifs_ftag)