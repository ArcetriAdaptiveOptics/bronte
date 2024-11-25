from specula import np, cpuArray
import specula
from arte.utils.decorator import logEnterAndExit
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula.processing_objects.func_generator import FuncGenerator

from specula.data_objects.source import Source
from specula.processing_objects.atmo_evolution import AtmoEvolution
from specula.processing_objects.atmo_propagation import AtmoPropagation
from bronte.types.virtual_wavefront_sensor import VirtualWavefrontSensor
import matplotlib.pyplot as plt

class DumbScaoSimulator():
    
    TIME_STEP = 0.01    #in seconds
    
    # GS_WFS_NAME_LIST = ['on_axis_source',
    #                     'lgs1_source']
    # GS_HEIGHT_LIST = [float('inf'),
    #                    90000]
    # GS_COORD_LIST = [[0.0, 0.0],
    #                  [0.0, 0.0]]
    # GS_MAG_LIST = [8,
    #                5]
    # GS_WL_LIST = [750,
    #               589]
    
    def __init__(self, factory):
        
        self._logger = None
        self._set_up_basic_logging()
        
        self._factory = factory
        
        self._seeing = None
        self._wind_speed = None 
        self._wind_direction = None
        self._setup_atmospheric_parameters()
        
        self._source_dict = None
        self._setup_sources()
        #self._generate_virtual_gs_wfs()
        
        self._atmo = None 
        self._prop = None
        self._initialize_atmosphere()
        
        self._set_atmospheric_evolution_inputs()
        self._set_atmospheric_propagation_inputs()
        
        self._group_list = None 
        self._initialize_groups()
        
        self._display_in_loop = False
    
    def _set_up_basic_logging(self):
        import importlib
        import logging
        importlib.reload(logging)
        FORMAT = '%(asctime)s:%(levelname)s:%(name)s  %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=FORMAT)
        self._logger = logging.getLogger("DumbScaoSimulator")
    
    @logEnterAndExit("Setting Atmo parameters...", "Atmo parameters set.", level='debug')
    def _setup_atmospheric_parameters(self, target_device_idx=-1, xp=np):
    
        self._seeing = FuncGenerator(
            constant = 0.65,
            target_device_idx = target_device_idx )
        
        self._wind_speed = FuncGenerator(
             constant=[5.5, 5.5],
             target_device_idx = target_device_idx)
        
        self._wind_direction = FuncGenerator(
            constant=[0, 0],
            target_device_idx = target_device_idx)
    
    @logEnterAndExit("Setting Sources...", "Sources set.", level='debug')
    def _setup_sources(self):
    
        self._on_axis_source = Source(polar_coordinate=[0.0, 0.0], magnitude=8, wavelengthInNm=750)
        self._lgs1_source = Source(polar_coordinate=[45.0, 0.0], height=90000, magnitude=5, wavelengthInNm=589)
        self._source_dict = {'on_axis_source': self._on_axis_source,
                            'lgs1_source': self._lgs1_source}
    
    # @logEnterAndExit("Generating Virtual GS-WFS...", "Virtual GS-WFS generated..", level='debug')    
    # def _generate_virtual_gs_wfs(self):
    #
    #     self._virtual_wfs_list = []
    #     num_of_wfs  = len(self.GS_WFS_NAME_LIST)
    #
    #     for idx in np.arange(num_of_wfs):
    #         wfs = VirtualWavefrontSensor(
    #             ID = idx,
    #             name = self.GS_WFS_NAME_LIST[idx],
    #             wfs_type ='SHWFS', 
    #             polar_coordinate = self.GS_COORD_LIST[idx],
    #             height = self.GS_HEIGHT_LIST[idx],
    #             magnitude = self.GS_MAG_LIST[idx],
    #             wavelengthInNm = self.GS_WL_LIST[idx]
    #             )
    #         self._virtual_wfs_list.append(wfs)
    #
    #     self._source_dict = { wfs._name: wfs._gs_source for wfs in self._virtual_wfs_list}
    
    @logEnterAndExit("Initialising Atmosphere...", "Atmosphere initialised.", level='debug')
    def _initialize_atmosphere(self, target_device_idx=-1):
        
        pupil_diameter_in_pixel  = 2 * self._factory.slm_pupil_mask.radius()
        pupil_pixel_pitch = round(40/pupil_diameter_in_pixel, 3)
        
        self._atmo = AtmoEvolution(
            pixel_pupil = pupil_diameter_in_pixel, # Linear dimension of pupil phase array
            pixel_pitch= pupil_pixel_pitch,         # Linear dimension of pupil phase array
            data_dir = 'calib/ELT',      # Data directory for phasescreens
            L0=23,                        # [m] Outer scale
            heights = [600, 20000], # [m] layer heights at 0 zenith angle
            Cn2 = [1 - 0.119977, 0.119977], # Cn2 weights (total must be eq 1)
            source_dict = self._source_dict,
            target_device_idx = target_device_idx
            )
        
        self._prop = AtmoPropagation(
            pixel_pupil = pupil_diameter_in_pixel,             
            pixel_pitch = pupil_pixel_pitch,         
            source_dict = self._source_dict,
            target_device_idx = target_device_idx
            )
    
    @logEnterAndExit("Setting Atmo evolution inputs...", "Atmo evolution inputs set.", level='debug')
    def _set_atmospheric_evolution_inputs(self):
        
        self._atmo.inputs['seeing'].set(self._seeing.output)
        self._atmo.inputs['wind_direction'].set(self._wind_direction.output)
        self._atmo.inputs['wind_speed'].set(self._wind_speed.output)
        
    @logEnterAndExit("Setting Atmo propagation inputs...", "Atmo propagation inputs set.", level='debug')    
    def _set_atmospheric_propagation_inputs(self, dm = None):
        
        if dm is not None:
            self._prop.inputs['layer_list'].set([self._atmo.layer_list] + dm.out_layer[:-1])
        else:
            self._prop.inputs['layer_list'].set(self._atmo.layer_list)
            
    @logEnterAndExit("Initialising Groups...", "Groups initialised.", level='debug')
    def _initialize_groups(self):
        
        group1 = [self._seeing, self._wind_speed, self._wind_direction]
        group2 = [self._atmo]
        group3 = [self._prop]
        self._group_list = [group1, group2, group3]
        
        for group in self._group_list:
            for obj in group:
                obj.run_check(self.TIME_STEP)
    
    @logEnterAndExit("Checking and triggering Groups ...", "Groups Triggered.", level='debug')
    def _check_and_trigger_groups(self, t):
        
        for group in self._group_list:
            for obj in group:
                obj.check_ready(t*1e9)
                obj.trigger()
                obj.post_trigger()
    
    @logEnterAndExit("Getting Projected phase screens on GSs directions...", "Phase screens get.", level='debug')            
    def _get_projected_phase_screens(self):
        
        ef = self._prop.outputs['out_on_axis_source_ef']
        phase_on_axis = cpuArray(ef.phaseInNm)
        phase_on_axis = self._factory.slm_rasterizer.get_recentered_phase_screen_on_slm_pupil_frame(
            phase_on_axis)
        ef = self._prop.outputs['out_lgs1_source_ef']
        phase_on_lgs1 = cpuArray(ef.phaseInNm)
        
        phase_on_lgs1 = self._factory.slm_rasterizer.get_recentered_phase_screen_on_slm_pupil_frame(
            phase_on_lgs1)
        
        return phase_on_axis, phase_on_lgs1
    

    def execute_simulation(self, Nsteps = 10):
        
        tf = Nsteps * self.TIME_STEP
        self._logger.info("Executing simulation")
        
        for step in range(Nsteps):
            
            t = self.TIME_STEP + step * self.TIME_STEP
            self._logger.info("\n+ Phase screens propagation @ time: %f/%f s\t steps: %d/%d" % (t, tf, step+1, Nsteps))
            self._check_and_trigger_groups(t)
            
            self._on_axis_phase, self._lgs1_phase = self._get_projected_phase_screens()
            
            if self._display_in_loop:
                self.display()
    
    def enable_display_in_loop(self, true_or_false):
        self._display_in_loop = true_or_false
    
    @logEnterAndExit("Refreshing display...", "Display refreshed", level='debug')
    def display(self):
        plt.figure(1)
        plt.clf()
        plt.title('On axis phase screen')
        plt.imshow(self._on_axis_phase)
        plt.colorbar()
        plt.show(block = False)
        plt.pause(0.2)
        
        plt.figure(2)
        plt.clf()
        plt.title('LGS direction phase screen')
        plt.imshow(self._lgs1_phase)
        plt.colorbar()
        plt.show(block = False)
        plt.pause(0.2)              