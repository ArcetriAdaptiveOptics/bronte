import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np
from specula.display.slopec_display import SlopecDisplay
from bronte.startup import measured_calibration_startup

class MeasuredControlMatrixCalibrator():
    
    def __init__(self, ftag, pp_amp_in_nm = None, xp=np):
        self._factory = measured_calibration_startup()
        self._Nmodes = self._factory.N_MODES_TO_CORRECT 
        if pp_amp_in_nm is not None:
            self._factory.PP_AMP_IN_NM = pp_amp_in_nm
            
        self._prop = self._factory.disturb_propagation
        self._slopec = self._factory.slope_computer
        self._dm = self._factory.virtual_deformable_mirror
        self._bench_devices = self._factory.testbench_devices
        self._pp = self._factory.push_pull
        self._im_calibrator = self._factory.interaction_matrix_calibrator(ftag)
        self._empty_layer = self._factory.empty_layer
        self._empty_layer.generation_time = 0
        
        self._set_inputs()
        self._define_groups()
        
    def _set_inputs(self):
        
        self._im_calibrator.inputs['in_slopes'].set(self._slopec.outputs['out_slopes'])
        self._im_calibrator.inputs['in_commands'].set(self._pp.output)
        self._bench_devices.inputs['ef'].set(self._prop.outputs['out_on_axis_source_ef'])
        self._slopec.inputs['in_pixels'].set(self._bench_devices.outputs['out_pixels'])
        self._dm.inputs['in_command'].set(self._pp.output)
        self._prop.inputs['layer_list'].set([self._empty_layer, self._dm.outputs['out_layer']])
    
        self._slopes_disp = SlopecDisplay()
        self._slopes_disp.inputs['slopes'].set(self._slopec.outputs['out_slopes'])
        self._slopes_disp.inputs['subapdata'].set(self._factory.subapertures_set)
    
    def _define_groups(self):
        
        group1 = [self._pp]
        group2 = [self._dm]
        group3 = [self._prop]
        group4 = [self._bench_devices]
        group5 = [self._slopec]
        group6 = [self._im_calibrator]
        group7 = [self._slopes_disp]
        
        self._groups = [group1, group2, group3, group4, group5, group6, group7]
    
    def run(self):
        
        time_step = self._factory.TIME_STEP_IN_SEC
        n_steps = 2*self._Nmodes
        for group in self._groups:
            for obj in group:
                obj.loop_dt = time_step * 1e9
                obj.run_check(time_step)
        
        for step in range(n_steps):
            t = 0 + step * time_step
            print('T=',t)
            for group in self._groups:
                for obj in group:
                    obj.check_ready(t*1e9)
                    print('trigger', obj)
                    obj.trigger()
                    obj.post_trigger()
        
        for group in self._groups:
            for obj in group:
                obj.finalize()