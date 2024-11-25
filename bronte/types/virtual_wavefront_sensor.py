from specula.data_objects.source import Source 

class VirtualWavefrontSensor():
    
    def __init__(self, 
                 ID = 1,
                 name = 'on_axis_source',
                 wfs_type = 'SHWFS',
                 polar_coordinate=[0.0, 0.0],
                 height = 90000,
                 magnitude = 5,
                 wavelengthInNm=589):
        
        self._id = ID
        self._name = name
        self._wfs_type =  wfs_type
        self._gs_polar_coordinate = polar_coordinate
        self._gs_height = height
        
        if height is not float('inf'):
            self._gs_type = 'LGS'
        else:
            self._gs_type = 'NGS'
        self._gs_magnitude = magnitude
        self._wl = wavelengthInNm
        self._gs_source = None
        self._setup_guide_star_source()
        
        self._slopes = None
        self._slope_offset = None
        self._reconstructor = None
        self._interaction_matrix = None
        self._projected_phase_screen = None
        
    def _setup_guide_star_source(self):
        self._gs_source = Source(
            self._gs_polar_coordinate,
            self._gs_height,
            self._gs_magnitude,
            self._wl)
        
    def load_slopes(self, slopes):
        self._slopes = slopes
        
    def get_slopes(self):
        return self._slopes
    
    def load_slope_offset(self, slope_offset):
        self._slope_offset = slope_offset
    
    def get_projected_phase_screen(self):
        return self._projected_phase_screen
    
    def load_projected_phase_screen(self, phase_screen):
        self._projected_phase_screen = phase_screen
        
    def load_reconstructor_matrix(self, reconstructor):
        self._reconstructor = reconstructor
        
    def get_reconstructor_matrix(self):
        return self._reconstructor
    