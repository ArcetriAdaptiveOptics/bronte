import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np, cpuArray
from arte.utils.decorator import logEnterAndExit
from bronte.utils.set_basic_logging import set_basic_logging
import matplotlib.pyplot as plt

class ScaoLoopSimulator():
    
    TIME_STEP = 0.01    #in seconds
    
    def __init__(self, atmospheric_simulator):
        
        self._logger = set_basic_logging("ScaoLoopSimulator")
        self._display_in_loop = False
        self._factory = None
        #TODO: rise error if is not BaseAtmosphericSimulator
        self._atmo_sim = atmospheric_simulator
        

    @logEnterAndExit("Initialising Groups...",
                     "Groups initialised.", level='debug')
    def initialize_groups(self):
        
        group1 = [self._atmo_sim._seeing,
                  self._atmo_sim._wind_speed,
                  self._atmo_sim._wind_direction]
        group2 = [self._atmo_sim._atmo]
        group3 = [self._atmo_sim._prop]
        self._group_list = [group1, group2, group3]
        
        for group in self._group_list:
            for obj in group:
                obj.run_check(self.TIME_STEP)
                
    def run_simulation(self, factory, Nsteps = 10):
        
        tf = Nsteps * self.TIME_STEP
        self._logger.info("Executing simulation")
        self._factory = factory
        for step in range(Nsteps):
            
            t = self.TIME_STEP + step * self.TIME_STEP
            self._logger.info("\n+ Phase screens propagation @ time: %f/%f s\t steps: %d/%d" % (t, tf, step+1, Nsteps))
            self._check_and_trigger_groups(t)
            
            self._on_axis_phase, self._lgs1_phase = self._get_projected_phase_screens()
            
            if self._display_in_loop:
                self.display()
        
        self._finalise_groups()
    
    @logEnterAndExit("Checking and triggering Groups ...",
                      "Groups Triggered.", level='debug')
    def _check_and_trigger_groups(self, t):
        
        for group in self._group_list:
            for obj in group:
                obj.check_ready(t*1e9)
                obj.trigger()
                obj.post_trigger()
    
    #TODO: move to bronte obj processor
    @logEnterAndExit("Getting Projected phase screens on GSs directions...",
                     "Phase screens get.", level='debug')            
    def _get_projected_phase_screens(self):
        
        ef = self._atmo_sim._prop.outputs['out_on_axis_source_ef']
        phase_on_axis = cpuArray(ef.phaseInNm)
        phase_on_axis = self._factory.slm_rasterizer.get_recentered_phase_screen_on_slm_pupil_frame(
            phase_on_axis)
        ef = self._atmo_sim._prop.outputs['out_lgs1_source_ef']
        phase_on_lgs1 = cpuArray(ef.phaseInNm)
        
        phase_on_lgs1 = self._factory.slm_rasterizer.get_recentered_phase_screen_on_slm_pupil_frame(
            phase_on_lgs1)
        
        return phase_on_axis, phase_on_lgs1

    @logEnterAndExit("Finalising Groups...", "Groups finalised.", level='debug')        
    def _finalise_groups(self):
        
        for group in self._group_list:
            for obj in group:
                obj.finalize()
    
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