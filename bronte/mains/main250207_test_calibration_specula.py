import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np
from specula.base_value import BaseValue
from specula.processing_objects.im_rec_calibrator import ImRecCalibrator
from specula.processing_objects.func_generator import FuncGenerator
from bronte.types.testbench_device_manager import TestbenchDeviceManager
from specula.data_objects.source import Source
from specula.processing_objects.atmo_propagation import AtmoPropagation
from specula.data_objects.layer import Layer
from specula.data_objects.subap_data import SubapData
from specula.processing_objects.sh_slopec import ShSlopec
from specula.processing_objects.dm import DM
from specula.data_objects.recmat import Recmat
from specula.processing_objects.modalrec import Modalrec
from bronte.startup import startup
from bronte.utils.noll_to_radial_order import from_noll_to_radial_order
from bronte.package_data import subaperture_set_folder, reconstructor_folder

class TestCalibration():
    
    def __init__(self, amp_in_nm = 1000, rec_tag ='250207_150800', target_device_idx=-1, xp=np):
        
        self._factory = startup()
        self._factory.sh_camera.setExposureTime(8)
        
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
        self._cmd.value[1] = amp_in_nm
        self._set_inputs()
        self._define_groups()
        
        
    
    def _set_inputs(self):
        
        self._bench_devices.inputs['ef'].set(self._dm.outputs['out_layer'])
        self._slopec.inputs['in_pixels'].set(self._bench_devices.outputs['out_pixels'])
        self._rec.inputs['in_slopes'].set(self._slopec.outputs['out_slopes'])
        self._dm.inputs['in_command'].set(self._cmd)
        
    def _define_groups(self):
        
        group1 = [self._dm]
        group2 = [self._bench_devices]
        group3 = [self._slopec]
        group4 = [self._rec]
        
        self._groups = [group1, group2, group3, group4]
    
    def run(self):
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
                
def main():
    
    # remain to choose the proper reconstructor in the factory
    rec_tag = '250207_150800' #pp=3um/n*n
    amp = 0 # flat of the calibration not the WFC
    tc = TestCalibration(amp, rec_tag)
    tc.run()
    modes_zero = tc._rec.outputs['out_modes'].value
    
    amp = 100 # 100 nm rms of tilt
    tc = TestCalibration(amp, rec_tag)
    tc.run()
    modes_100 = tc._rec.outputs['out_modes'].value
    
    amp = 1000 # 1000 nm rms of tilt
    tc = TestCalibration(amp, rec_tag)
    tc.run()
    modes_1000 = tc._rec.outputs['out_modes'].value
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.clf()
    plt.title('Calibration pp=3um/n^2')
    plt.plot(modes_100, 'o-', label = 'tilt 100 nm rms wf')
    plt.plot(modes_zero, 'o-', label = 'zero')
    plt.plot(modes_1000, 'o-', label = 'tilt 100 nm rms wf')
    plt.legend(loc='best')
    
    
    rec_tag = '250207_124200' #pp=1um/n*n
    amp = 0 # flat of the calibration not the WFC
    tc = TestCalibration(amp, rec_tag)
    tc.run()
    modes_zero = tc._rec.outputs['out_modes'].value
    
    amp = 100 # 100 nm rms of tilt
    tc = TestCalibration(amp, rec_tag)
    tc.run()
    modes_100 = tc._rec.outputs['out_modes'].value
    
    amp = 1000 # 1000 nm rms of tilt
    tc = TestCalibration(amp, rec_tag)
    tc.run()
    modes_1000 = tc._rec.outputs['out_modes'].value
    
    plt.figure()
    plt.clf()
    plt.title('Calibration pp=1um/n^2')
    plt.plot(modes_100, 'o-', label = 'tilt 100 nm rms wf')
    plt.plot(modes_zero, 'o-', label = 'zero')
    plt.plot(modes_1000, 'o-', label = 'tilt 100 nm rms wf')
    plt.legend(loc='best')