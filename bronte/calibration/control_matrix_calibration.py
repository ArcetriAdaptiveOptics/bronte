import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np
from specula.processing_objects.im_rec_calibrator import ImRecCalibrator
from specula.processing_objects.func_generator import FuncGenerator
from bronte.types.testbench_device_manager import TestbenchDeviceManager
from specula.data_objects.source import Source
from specula.processing_objects.atmo_propagation import AtmoPropagation
from specula.data_objects.layer import Layer
from specula.data_objects.subap_data import SubapData
from specula.processing_objects.sh_slopec import ShSlopec
from specula.processing_objects.dm import DM
from bronte.startup import startup
from bronte.utils.noll_to_radial_order import from_noll_to_radial_order
from bronte.package_data import subaperture_set_folder, reconstructor_folder

class ControlMatrixCalibrator():
    
    def __init__(self, ftag, target_device_idx=-1, xp=np):
        
        self._factory = startup()
        self._factory.sh_camera.setExposureTime(8)
        thr_shwfs_pix = self._factory.SH_PIX_THR
        telescope_pupil_diameter = 40
        pupil_diameter_in_pixel  = 2 * self._factory.slm_pupil_mask.radius()
        pupil_pixel_pitch = round(telescope_pupil_diameter/pupil_diameter_in_pixel, 3)

        
        on_axis_source = Source(polar_coordinate=[0.0, 0.0], magnitude=8, wavelengthInNm=750,)


        self._prop = AtmoPropagation(pixel_pupil=pupil_diameter_in_pixel,              # Linear dimension of pupil phase array
                               pixel_pitch= pupil_pixel_pitch,         # Linear dimension of pupil phase array
                               source_dict = {'on_axis_source': on_axis_source,
                                            },
                               target_device_idx=target_device_idx)
        
        subapdata = SubapData.restore_from_bronte(
            subaperture_set_folder() / (self._factory.SUBAPS_TAG + ".fits"))
        
        self._slopec = ShSlopec(subapdata= subapdata, thr_value =  thr_shwfs_pix)
        
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
        
        #ampla = np.ones(nModes)
        # ampla[0:3]=1
        # ampla[3:5]=0.2
        # ampla[5:14]=0.15
        # ampla[14:54]=0.1
        # ampla[54:]=0.08
        # ampl_vect = ampla[0:nModes]*1000
        j_noll_vector = np.arange(nModes) + 2
        radial_order = from_noll_to_radial_order(j_noll_vector)
        self._ampl_vect = 8000 /(radial_order**2) # in nm
        
        self._n_steps = nModes * 2
        self._pp = FuncGenerator(func_type= 'PUSHPULL',
                           nmodes=nModes,
                           vect_amplitude = self._ampl_vect,#in nm
                           target_device_idx=target_device_idx)
    
        self._im_calibrator = ImRecCalibrator(
                            data_dir = reconstructor_folder(),
                            nmodes=nModes,
                            rec_tag= ftag + '_bronte_rec',
                            im_tag= ftag + '_bronte_im',
                            target_device_idx=target_device_idx)
        
        self._empty_layer = Layer(pupil_diameter_in_pixel, pupil_diameter_in_pixel, pupil_pixel_pitch, height=0)
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
    
    def _define_groups(self):
        
        group1 = [self._pp]
        group2 = [self._dm]
        group3 = [self._prop]
        group4 = [self._bench_devices]
        group5 = [self._slopec]
        group6 = [self._im_calibrator]
        
        self._groups = [group1, group2, group3, group4, group5, group6]
    
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