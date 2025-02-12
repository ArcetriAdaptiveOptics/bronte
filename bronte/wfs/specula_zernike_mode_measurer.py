import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np
from specula.base_value import BaseValue
from bronte.types.testbench_device_manager import TestbenchDeviceManager
from specula.data_objects.subap_data import SubapData
from specula.processing_objects.sh_slopec import ShSlopec
from specula.processing_objects.dm import DM
from specula.data_objects.recmat import Recmat
from specula.processing_objects.modalrec import Modalrec
from bronte.package_data import subaperture_set_folder, reconstructor_folder


class ZernikeModesMeasurer():
    '''
    The ZernikeModesMeasurer class applies a disturbance in the testbench, expressed as a
    Zernike mode combination, and measures the reconstructed modes.
    
    '''
    def __init__(self, factory, zc_vect_in_nm = np.array([0, 1000]), rec_tag ='250207_150800', target_device_idx=-1, xp=np):
        
        '''
        Parameters:
        - factory: Factory object containing testbench configurations.
        - zc_vect_in_nm: Vector of Zernike coefficients in nm RMS wavefront.
        - rec_tag: File tag for the reconstructor.
        '''
        
        self._factory = factory
        telescope_pupil_diameter = 40
        pupil_diameter_in_pixel  = 2 * self._factory.slm_pupil_mask.radius()
        pupil_pixel_pitch = round(telescope_pupil_diameter/pupil_diameter_in_pixel, 3)

        self._n_steps = 1

        subapdata = SubapData.restore_from_bronte(
            subaperture_set_folder() / (self._factory.SUBAPS_TAG + ".fits"))
        
        self._slopec = ShSlopec(subapdata= subapdata)
        
        nModes =self._factory.N_MODES_TO_CORRECT
        
        self._dm = DM(type_str='zernike',
                pixel_pitch=pupil_pixel_pitch,
                nmodes=nModes,
                npixels= pupil_diameter_in_pixel,                    # linear dimension of DM phase array
                obsratio= 0,                    # obstruction dimension ratio w.r.t. diameter
                height=  0)     # DM height [m]
        
        self._bench_devices = TestbenchDeviceManager(self._factory, 
                                        do_plots=True,
                                        target_device_idx=target_device_idx)
        
        recmat = Recmat.restore(reconstructor_folder() / (rec_tag + "_bronte_rec.fits"))
        self._rec = Modalrec(nModes, recmat=recmat)
        self._cmd = BaseValue(value=np.zeros(nModes))
        self._cmd.generation_time = 0
        self._cmd.value = zc_vect_in_nm
        self._set_inputs()
        self._define_groups()
    
    def _set_inputs(self):
        
        '''
        Configures input connections between components:
        - Connects the deformable mirror (DM) output to the testbench devices.
        - Routes pixel data through the ShSlopec processor.
        - Feeds computed slopes into the modal reconstructor.
        - Assigns the Zernike coefficient vector as the deformable mirror command.
        '''
        
        self._bench_devices.inputs['ef'].set(self._dm.outputs['out_layer'])
        self._slopec.inputs['in_pixels'].set(self._bench_devices.outputs['out_pixels'])
        self._rec.inputs['in_slopes'].set(self._slopec.outputs['out_slopes'])
        self._dm.inputs['in_command'].set(self._cmd)
        
    def _define_groups(self):
        '''
        Defines execution sequence by grouping objects into stages:
        1. Group 1: Deformable mirror.
        2. Group 2: Testbench device manager.
        3. Group 3: Slope computation.
        4. Group 4: Modal reconstruction.
        '''
        group1 = [self._dm]
        group2 = [self._bench_devices]
        group3 = [self._slopec]
        group4 = [self._rec]
        
        self._groups = [group1, group2, group3, group4]
    
    def run(self):
        
        '''
        Executes the measurement process in a stepwise manner:
        1. Assigns a time step (0.01s) for simulation.
        2. Iterates through object groups to initialize execution.
        3. Runs a loop to trigger objects sequentially and process data.
        4. Finalizes all objects at the end of execution.
        '''
        time_step = 0.01
        
        for group in self._groups:
            for obj in group:
                obj.loop_dt = time_step * 1e9
                obj.run_check(time_step)
    
        for step in range(self._n_steps):
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