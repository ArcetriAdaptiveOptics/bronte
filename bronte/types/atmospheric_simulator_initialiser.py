import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np, cpuArray
from arte.utils.decorator import logEnterAndExit
from specula.processing_objects.func_generator import FuncGenerator
from specula.data_objects.source import Source
from specula.processing_objects.atmo_evolution import AtmoEvolution
from specula.processing_objects.atmo_propagation import AtmoPropagation

from bronte.utils.set_basic_logging import set_basic_logging

class AtmosphericSimulatorInitialiser():
    
    def __init__(self):
        
        self._logger = set_basic_logging("AtmosphericSimulatorInitialiser")
        self._atmo_params_are_set = False
        self._sources_are_set = False
        self._atmo_evol_and_prop_params_are_set = False
        
    @logEnterAndExit("Setting Atmo parameters...",
                      "Atmo parameters set.", level='debug')
    def set_atmospheric_parameters(self,
                                   seeing_in_arcsec,
                                   wind_speed_list,
                                   wind_direction_list,
                                   target_device_idx=-1, xp=np):
        
        self._check_list_lengths([wind_speed_list, wind_direction_list])
        self._num_of_tubulent_layers = len(wind_speed_list)
        
        self._seeing = FuncGenerator(
            constant = seeing_in_arcsec,
            target_device_idx = target_device_idx )
        
        self._wind_speed = FuncGenerator(
             constant = wind_speed_list,
             target_device_idx = target_device_idx)
        
        self._wind_direction = FuncGenerator(
            constant = wind_direction_list,
            target_device_idx = target_device_idx)
        
        self._atmo_params_are_set = True
    
    @logEnterAndExit("Setting GS Sources...",
                      "GS Sources set.", level='debug')
    def set_guide_star_sources(self,
                               gs_source_name_list,
                               gs_polar_coords_list,
                               gs_magnitude_list,
                               gs_heights_list,
                               gs_wl_in_nm_list):
        
        input_list = [gs_source_name_list,
                      gs_polar_coords_list,
                      gs_magnitude_list,
                      gs_heights_list,
                      gs_wl_in_nm_list]
        
        self._check_list_lengths(input_list)
        self._num_of_gs = len(gs_magnitude_list)
        if self._num_of_gs > 2:
            raise ValueError(f"Only 2 GS are supported for SCAO Simulations")
        
        self._on_axis_source = Source(
            polar_coordinate=gs_polar_coords_list[0],
            height = gs_heights_list[0],
            magnitude = gs_magnitude_list[0],
            wavelengthInNm = gs_wl_in_nm_list[0])
        
        self._lgs1_source = Source(
            polar_coordinate=gs_polar_coords_list[1],
            height = gs_heights_list[1],
            magnitude = gs_magnitude_list[1],
            wavelengthInNm = gs_wl_in_nm_list[1])
        
        self._source_dict = {gs_source_name_list[0]: self._on_axis_source,
                            gs_source_name_list[1]: self._lgs1_source}
        self._sources_are_set = True 
        
    @logEnterAndExit("Setting Atmo evolution/propagation parameters...",
                      "Atmo evolution/propagation parameters set.", level='debug')
    def set_atmospheric_evolution_and_propagation_parameters(self,
                                             pupil_diameter_in_meters,
                                             pupil_diameter_in_pixel,
                                             outer_scale_in_m,
                                             height_list,
                                             Cn2_list,
                                             data_dir = 'calib/bronte',
                                             target_device_idx = -1):
        
        if self._sources_are_set is False or self._atmo_params_are_set is False:
            raise RuntimeError("set_atmospheric_parameters and "\
                               "set_guide_star_sources must be called first.")
        
        self._check_list_lengths([height_list, Cn2_list])
        if len(height_list) != self._num_of_tubulent_layers:
            raise ValueError(f"Input list lengths must be equal to the number"\
                             f"of turbulent layers {self._num_of_tubulent_layers}")
        if sum(Cn2_list) != 1.:
            raise ValueError(f"Cn2 weights sum must be equal to 1, while is {Cn2_list.sum()}")
        
        pupil_pixel_pitch = round(pupil_diameter_in_meters/pupil_diameter_in_pixel, 3)
        
        self._atmo = AtmoEvolution(
            pixel_pupil = pupil_diameter_in_pixel, # Linear dimension of pupil phase array
            pixel_pitch= pupil_pixel_pitch,         # Linear dimension of pupil phase array
            data_dir = data_dir,      # Data directory for phasescreens
            L0 = outer_scale_in_m,                        # [m] Outer scale
            heights = height_list, # [m] layer heights at 0 zenith angle
            Cn2 = Cn2_list, # Cn2 weights (total must be eq 1)
            source_dict = self._source_dict,
            target_device_idx = target_device_idx)
    
        self._prop = AtmoPropagation(
            pixel_pupil = pupil_diameter_in_pixel,             
            pixel_pitch = pupil_pixel_pitch,         
            source_dict = self._source_dict,
            target_device_idx = target_device_idx)
        
        self._set_atmosphericevolution_and_propagation_inputs()
        self._atmo_evol_and_prop_params_are_set = True
        
    @logEnterAndExit("Setting Atmo evolution/propagation inputs...",
                     "Atmo evolution/propagation inputs set.", level='debug')
    def _set_atmosphericevolution_and_propagation_inputs(self):
        
        self._atmo.inputs['seeing'].set(self._seeing.output)
        self._atmo.inputs['wind_direction'].set(self._wind_direction.output)
        self._atmo.inputs['wind_speed'].set(self._wind_speed.output)
        self._prop.inputs['layer_list'].set(self._atmo.layer_list)
    
    def _check_list_lengths(self, lists):
        lengths = [len(lst) for lst in lists]
        if len(set(lengths)) != 1:  # Check if all lengths are the same
            raise ValueError(f"Input Lists do not have the same length. Lengths: {lengths}")
        