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
import matplotlib.pyplot as plt
from bronte.utils.noll_to_radial_order import from_noll_to_radial_order

class TestCalibration():
    
    def __init__(self, amp_vect_in_nm = np.array([0, 1000]), rec_tag ='250207_150800', target_device_idx=-1, xp=np):
        
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
        self._cmd.value = amp_vect_in_nm
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


def get_modes_from_test_calib(amp, rec_tag):
    tc = TestCalibration(amp, rec_tag)
    tc.run()
    return tc._rec.outputs['out_modes'].value
    
def do_plot(modes_zero, modes_100, modes_1000, str_title):
       
    plt.figure()
    plt.clf()
    plt.title(str_title)
    plt.plot(modes_100, 'o-', label = 'tilt 100 nm rms wf')
    plt.plot(modes_zero, 'o-', label = 'zero')
    plt.plot(modes_1000, 'o-', label = 'tilt 1000 nm rms wf')
    plt.ylabel('modal coefficient [nm rms wf]')
    plt.xlabel('index')
    plt.legend(loc='best')
    plt.grid('--',alpha=0.3)
           

def main250207():
    
    
    rec_tag = '250210_101700' #pp=8um/n*n
    amp = np.zeros(200) # flat of the calibration not the WFC
    modes_zero_pp8 = get_modes_from_test_calib(amp, rec_tag)
    amp = np.zeros(200) 
    amp[1] = 100 # 100 nm rms of tilt
    modes_100_pp8 = get_modes_from_test_calib(amp, rec_tag)
    amp = np.zeros(200) 
    amp[1] = 1000 # 1000 nm rms of tilt
    modes_1000_pp8 = get_modes_from_test_calib(amp, rec_tag)
    do_plot(modes_zero_pp8, modes_100_pp8, modes_1000_pp8, 'Calibration pp=8um/n^2')
    
    
    rec_tag = '250207_150800' #pp=3um/n*n
    amp = np.zeros(200) # flat of the calibration not the WFC
    modes_zero_pp3 = get_modes_from_test_calib(amp, rec_tag)
    amp = np.zeros(200) 
    amp[1] = 100 # 100 nm rms of tilt
    modes_100_pp3 = get_modes_from_test_calib(amp, rec_tag)
    amp = np.zeros(200) 
    amp[1] = 1000 # 1000 nm rms of tilt
    modes_1000_pp3 = get_modes_from_test_calib(amp, rec_tag)
    do_plot(modes_zero_pp3, modes_100_pp3, modes_1000_pp3, 'Calibration pp=3um/n^2')
    
    rec_tag = '250207_120300' #pp=1um/n*n
    amp = np.zeros(200) # flat of the calibration not the WFC
    modes_zero_pp1 = get_modes_from_test_calib(amp, rec_tag)
    amp = np.zeros(200) 
    amp[1] = 100 # 100 nm rms of tilt# 100 nm rms of tilt
    modes_100_pp1 = get_modes_from_test_calib(amp, rec_tag)
    amp = np.zeros(200) 
    amp[1] = 1000 # 100 nm rms of tilt# 1000 nm rms of tilt
    modes_1000_pp1 = get_modes_from_test_calib(amp, rec_tag)
    do_plot(modes_zero_pp1, modes_100_pp1, modes_1000_pp1, 'Calibration pp=1um/n^2')
    
    print('\n + Calibration 8um/n^2:')
    print(modes_100_pp8[:3]-modes_zero_pp8[:3])
    print(modes_1000_pp8[:3]-modes_zero_pp8[:3])
    print('\n + Calibration 3um/n^2:')
    print(modes_100_pp3[:3]-modes_zero_pp3[:3])
    print(modes_1000_pp3[:3]-modes_zero_pp3[:3])
    print('\n + Calibration 1um/n^2:')
    print(modes_100_pp1[:3]-modes_zero_pp1[:3])
    print(modes_1000_pp1[:3]-modes_zero_pp1[:3])
   
def main250210_tt():
    
    rec_tag = '250210_115200' # pp=8 um rms for tip-tilt
    amp = np.zeros(2)
    modes_zero_tt = get_modes_from_test_calib(amp, rec_tag)
    amp[1]= 8000 # 8000 nm rms of tilt
    modes_8000_tt = get_modes_from_test_calib(amp, rec_tag)
   

    plt.figure()
    plt.clf()
    plt.title('TT Calibratio 8um rms/n^2')
    plt.plot(modes_8000_tt, 'o-', label = 'tilt 8 um rms wf')
    plt.plot(modes_zero_tt, 'o-', label = 'zero')
    plt.ylabel('modal coefficient [nm rms wf]')
    plt.xlabel('index')
    plt.legend(loc='best')
    plt.grid('--',alpha=0.3)
    
    print('\n + TT Calibration 8um/n^2:')
    print(modes_8000_tt-modes_zero_tt)
    
    
def main250210_z11():
    
    rec_tag = '250210_121400' # 10 modes pp=8um rms/n^2
    
    j_vect  = np.arange(2,12)
    n_vect = from_noll_to_radial_order(j_vect)
    pp_per_mode = 8000/n_vect**2
    
    amp = np.zeros(10)
    modes_zero = get_modes_from_test_calib(amp, rec_tag)
    
    amp = np.zeros(10) 
    amp[0] = pp_per_mode[0] # 1000 nm rms of tilt
    modes_tip = get_modes_from_test_calib(amp, rec_tag)
    
    amp = np.zeros(10) 
    amp[2] = pp_per_mode[2]
    modes_fuoco = get_modes_from_test_calib(amp, rec_tag)
    
    amp = np.zeros(10) 
    amp[4] = pp_per_mode[4]
    modes_astig = get_modes_from_test_calib(amp, rec_tag)
    
    amp = np.zeros(10) 
    amp[6] = pp_per_mode[6]
    modes_coma = get_modes_from_test_calib(amp, rec_tag)
    
    amp = np.zeros(10) 
    amp[-1] = pp_per_mode[-1]
    modes_sphere = get_modes_from_test_calib(amp, rec_tag)
    
    plt.figure()
    plt.clf()
    plt.title('Calibration up to Z11 (pp=8um rms/n^2)')
    plt.plot(j_vect, modes_tip - modes_zero, 'o-', label = 'c2 = %g nm rms'%pp_per_mode[0])
    plt.plot(j_vect, modes_fuoco - modes_zero, 'o-', label = 'c4 = %g nm rms'%pp_per_mode[2])
    plt.plot(j_vect, modes_astig - modes_zero, 'o-', label = 'c6 = %g nm rms'%pp_per_mode[4])
    plt.plot(j_vect, modes_coma - modes_zero, 'o-', label = 'c8 = %g nm rms'%pp_per_mode[6])
    plt.plot(j_vect, modes_sphere - modes_zero, 'o-', label = 'c11 = %f nm rms'%pp_per_mode[-1])
    plt.ylabel('modal coefficient difference wrt zero [nm rms wf]')
    plt.xlabel('Noll index')
    plt.legend(loc='best')
    plt.grid('--',alpha=0.3)
    
    print('\n + Calibration up to Z11 (pp=8um/n^2):')
    print('Z1-Tip')
    print(modes_tip[0] - modes_zero[0])
    print('Z4-Tip')
    print(modes_fuoco[2] - modes_zero[2])
    print('Z6-Astig')
    print(modes_astig[4] - modes_zero[4])
    print('Z8-Coma')
    print(modes_coma[6] - modes_zero[6])
    print('Z11-Sphere')
    print(modes_sphere[-1] - modes_zero[-1])  