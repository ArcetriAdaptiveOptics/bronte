import numpy as np
from bronte.types.virtual_wavefront_sensor import VirtualWavefrontSensor
from bronte.types.virtual_deformable_mirror import VirtualDeformableMirror
from bronte.utils.reconstructor_manager import ReconstructorManager
from bronte.utils.slopes_manager import SlopesManager


class AdaptiveOpticsLoopSimulator():
    
    
    TURB_LAYER_ALTITUDES = [0, 2000, 5500, 20000]
    GS_ON_SKY_POS = [0, 5, 10] # in arcsec
    DM_CONJ_ALTITUDES = [6000, 17000]
    DM_NUM_OF_ACTS = [918, 1027]
    CORRECTION_DELAY = 0.005
    
    def __init__(self, factory):
        
        self._factory = factory
        
        self._num_of_dm = None
        self._num_of_wfs = None
        
        self._turbulent_layers = None
        self._dm_conjugated_layers = None
        
        self._num_of_turb_layers = len(self.TURB_LAYER_ALTITUDES)
        self._num_of_dm_layers = len(self.DM_CONJ_ALTITUDES)
        
        self._generate_virtual_wfs(self.GS_ON_SKY_POS)
        self._generate_virtual_dm(self.DM_CONJ_ALTITUDES)
        
    def execute_loop(self, loop_steps = 3):
        
        self._generate_turbulent_layers()
        self._generate_conjugated_dm_layers()
        
        rm = ReconstructorManager(self._virtual_wfs_list)
        reconstructor = rm.get_combined_reconstructor()
        
        for step in np.arange(loop_steps):
            
            print("+ AO loop step: %d \n" %step)
            
            for wfs in self._virtual_wfs_list:
                
                print("\t - Projecting layers on WFS#%d at %f arcsec"% \
                      (wfs._wfs_id, wfs._gs_on_sky_position))
                
                projected_phase_screen = self._project_layers_on_wfs(wfs)
                
                self._load_phase_screen_on_slm(projected_phase_screen)
                
                wfs._slopes = self._get_slopes()
            
            sm = SlopesManager(self._virtual_wfs_list)
            measured_slopes = sm.get_combined_slopes()
            self._compure_dm_commands(
                measured_slopes, reconstructor, self._virtual_dm_list)
            
            self._update_dm_conjugated_layers()
            self._update_turbulent_layers(self.CORRECTION_DELAY)
    
    def _generate_virtual_wfs(self, gs_positions):
        
        self._virtual_wfs_list = []
        self._num_of_wfs  = len(gs_positions)
        
        for idx, gs_pos in enumerate(gs_positions):
            wfs = VirtualWavefrontSensor(
                ID = idx,
                gs_position_in_arcsec = gs_pos
                )
            self._virtual_wfs_list.append(wfs)
    
    def _generate_virtual_dm(self, conj_altitudes_in_meters):
        
        self._virtual_dm_list = []
        self._num_of_dm = len(conj_altitudes_in_meters)
        
        for idx, dm_altitude in enumerate(conj_altitudes_in_meters):
            dm  =VirtualDeformableMirror(
                ID  = idx,
                Nact = self.DM_NUM_OF_ACTS[idx],
                conj_altitude_in_meters= dm_altitude)
            
            self._virtual_dm_list.append(dm)
            
    
    def _generate_turbulent_layers(self):
        '''
        Generates the turbulent layers at different altitudes from a
        phase screen generator
        '''
        phase_screen_shape = (100, 100)
        self._turbulent_layers = np.zeros((self._num_of_turb_layers,phase_screen_shape))
        
        
    def _generate_conjugated_dm_layers(self, dm_conj_altitudes_in_meters):
        '''
        Generates the DMs conj layers from their flat shape
        '''
        pass
    
    def _project_layers_on_wfs(self, wfs):
        '''
        Computes the projected phase screen in the direction of the GS-WFS:
        integrates the turbulent and dm conj layers on the  direction of the wfs
        '''
        pass
    
    def _load_phase_screen_on_slm(self):
        '''
        Applies the projected phase screen on the SLM in the test-bench
        '''
        pass
    
    def _get_slopes(self):
        '''
        Returns the measured slopes from the SH-WFS in the test-bench
        '''
        pass
    def _compure_dm_commands(self, slopes, reconstructor, dm_list):
        pass
    
    def _update_turbulent_layers(self, correction_delay, wind_speed):
        pass
    
    def _update_dm_conjugated_layers(self):
        pass
    