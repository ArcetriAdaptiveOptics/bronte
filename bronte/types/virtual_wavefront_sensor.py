 

class VirtualWavefrontSensor():
    
    def __init__(self, 
                 ID = 1,
                 name = 'pippo',
                 GS = 'NGS',
                 wfs_type = 'SHWFS',
                 gs_position_in_arcsec = 0):
        
        self._id = ID
        self._name = name
        self._gs = GS
        self._wfs_type =  wfs_type
        self._gs_on_sky_position = gs_position_in_arcsec
        self._slopes = None
        self._slope_offset = None
        self._reconstructor = None
        self._interaction_matrix = None