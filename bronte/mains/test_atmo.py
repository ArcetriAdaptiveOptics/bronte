import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np, cpuArray, default_target_device
from specula.processing_objects.im_rec_calibrator import ImRecCalibrator

from specula.processing_objects.sh_subap_calibrator import ShSubapCalibrator
from specula import np, cpuArray, default_target_device
#from specula.data_objects.layer import Layer
from specula.data_objects.source import Source
from specula.processing_objects.atmo_propagation import AtmoPropagation
from specula.processing_objects.atmo_evolution import AtmoEvolution
from specula.processing_objects.func_generator import FuncGenerator
from specula.processing_objects.int_control import IntControl
from specula.processing_objects.sh_slopec import ShSlopec
from specula.processing_objects.dm import DM
from specula.processing_objects.modalrec import Modalrec
from specula.data_objects.subap_data import SubapData
from specula.data_objects.ef import ElectricField
from specula.data_objects.recmat import Recmat
from specula.data_objects.pixels import Pixels
from specula.data_objects.layer import Layer

from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputList, InputValue
from bronte.startup import startup
import time
import matplotlib.pyplot as plt
from bronte.types.testbench_device_manager import TestbenchDeviceManager

# import os
# print(os.getcwd())


def test_atmo(target_device_idx=-1, xp=np):
    
    factory = startup()
    telescope_pupil_diameter = 40
    pupil_diameter_in_pixel  = 2 * factory.slm_pupil_mask.radius()
    pupil_pixel_pitch = round(telescope_pupil_diameter/pupil_diameter_in_pixel, 3)
    
    seeing = FuncGenerator(constant=0.65,
                           target_device_idx=target_device_idx)
    wind_speed = FuncGenerator(constant=[5.5, 5.5],
                               target_device_idx=target_device_idx) 
    wind_direction = FuncGenerator(constant=[0, 0],
                                   target_device_idx=target_device_idx)

    on_axis_source = Source(polar_coordinate=[0.0, 0.0], magnitude=8, wavelengthInNm=750,)
    lgs1_source = Source(polar_coordinate=[45.0, 0.0], height=90000, magnitude=5, wavelengthInNm=589)
    lgs2_source = Source(polar_coordinate=[45.0, 60.0], height=90000, magnitude=5, wavelengthInNm=589)
    lgs3_source = Source(polar_coordinate=[45.0, 120.0], height=90000, magnitude=5, wavelengthInNm=589)
    lgs4_source = Source(polar_coordinate=[45.0, 180.0], height=90000, magnitude=5, wavelengthInNm=589)
    lgs5_source = Source(polar_coordinate=[45.0, 240.0], height=90000, magnitude=5, wavelengthInNm=589)
    lgs6_source = Source(polar_coordinate=[45.0, 300.0], height=90000, magnitude=5, wavelengthInNm=589)

    atmo = AtmoEvolution(pixel_pupil=pupil_diameter_in_pixel,              # Linear dimension of pupil phase array
                         pixel_pitch= pupil_pixel_pitch,         # Linear dimension of pupil phase array
                         data_dir = 'calib/bronte',      # Data directory for phasescreens
                         L0=23,                        # [m] Outer scale
                         heights = [300.000,  20500.0], # [m] layer heights at 0 zenith angle
                         Cn2 = [1 - 0.119977, 0.119977], # Cn2 weights (total must be eq 1)
                        source_dict = {'on_axis_source': on_axis_source,
                                        'lgs1_source': lgs1_source,
                                        },
                        target_device_idx=target_device_idx,
                        )

    prop = AtmoPropagation(pixel_pupil=pupil_diameter_in_pixel,              # Linear dimension of pupil phase array
                           pixel_pitch= pupil_pixel_pitch,         # Linear dimension of pupil phase array
                           source_dict = {'on_axis_source': on_axis_source,
                                        'lgs1_source': lgs1_source,
                                        },
                           target_device_idx=target_device_idx)

    subapdata = SubapData.restore_from_bronte('C:\\Users\\labot\\Desktop\\misure_tesi_slm\\bronte\\subaperture_set\\240807_152700.fits')
    
    slopec = ShSlopec(subapdata= subapdata)

    nslopes = 1510*2
    nModes = 10
    recmat = Recmat(recmat=np.zeros((nModes, nslopes)))
#    recmat = Recmat.restore('hiresA_ps512p0.076_pyr90x90_wl798_fv2.1_ft3.0_ma4_bn1_th0.30a0.30b_mn4094')

    rec = Modalrec(nModes, recmat=recmat)

    control = IntControl(delay=2, int_gain=np.ones(nModes)*0.5)
    dm = DM(type_str='zernike',
            pixel_pitch=pupil_diameter_in_pixel,
            nmodes=nModes,
            npixels= pupil_diameter_in_pixel,                    # linear dimension of DM phase array
            obsratio= 0,                    # obstruction dimension ratio w.r.t. diameter
            height=  0)     # DM height [m]
  
    factory = startup()
    
    bronte = TestbenchDeviceManager(factory, target_device_idx=target_device_idx)

    ######################
    # CALIBRATION objects
        
    sh_subap_calibrator= ShSubapCalibrator(
        data_dir = 'calib/bronte',
        subap_on_diameter= 46,
        output_tag='bronte_subaps_n46_th0.5',
        energy_th=         0.5,
        target_device_idx=target_device_idx)
    
    pp = FuncGenerator(func_type= 'PUSHPULL',
                       nmodes=10,
                       vect_amplitude = np.ones(10)*500,
                       target_device_idx=target_device_idx)

    im_calibrator = ImRecCalibrator(
                        data_dir = 'calib/bronte',
                        nmodes=10,
                        output_tag='bronte_rec',
                        target_device_idx=target_device_idx)


    im_calibrator.inputs['in_slopes'].set(slopec.outputs['out_slopes'])
    im_calibrator.inputs['in_commands'].set(pp.output)
    sh_subap_calibrator.inputs['in_pixels'].set(bronte.outputs['out_pixels'])

    ########################
    # MAIN

    empty_layer = Layer(pupil_diameter_in_pixel, pupil_diameter_in_pixel, pupil_pixel_pitch, height=0)
    empty_layer.generation_time = 0
    
    calib_rec = False
    atmo.inputs['seeing'].set(seeing.output)
    atmo.inputs['wind_direction'].set(wind_direction.output)
    atmo.inputs['wind_speed'].set(wind_speed.output)
    prop.inputs['layer_list'].set(atmo.layer_list + [dm.outputs['out_layer']])

    bronte.inputs['ef'].set(prop.outputs['out_on_axis_source_ef'])
    slopec.inputs['in_pixels'].set(bronte.outputs['out_pixels'])
    rec.inputs['in_slopes'].set(slopec.outputs['out_slopes'])
    control.inputs['delta_comm'].set(rec.out_modes)
    dm.inputs['in_command'].set(control.out_comm)

    if calib_rec:
        dm.inputs['in_command'].set(pp.output)
        prop.inputs['layer_list'].set([empty_layer, dm.outputs['out_layer']])
        n_steps = nModes * 2
    else:
        n_steps = 1000

    group1 = [seeing, wind_speed, wind_direction, pp]
    if calib_rec:
        group2 = [dm, atmo]
    else:
        group2 = [atmo]
    group3 = [prop]
    group4 = [bronte]
    group5 = [slopec]
    group6 = [rec, im_calibrator]
    group7 = [control]
    if calib_rec:
        group8 = []
    else:
        group8 = [dm]

    time_step = 0.01
    
    for group in [group1, group2, group3, group4, group5, group6, group7, group8]:
        for obj in group:
            obj.loop_dt = time_step * 1e9
            obj.run_check(time_step)

    for step in range(n_steps):
        t = 0 + step * time_step
        print('T=',t)
        for group in [group1, group2, group3, group4, group5, group6, group7, group8]:
            for obj in group:
                obj.check_ready(t*1e9)
                print('trigger', obj)
                obj.trigger()
                obj.post_trigger()

    for group in [group1, group2, group3, group4, group5, group6, group7, group8]:
        for obj in group:
            obj.finalize()

        # ef = prop.outputs['out_on_axis_source_ef']
        # phase_on_axis_source = cpuArray(ef.phaseInNm)
        # applyPhaseOnSlm(slm, phase_on_axis_source)
        # frame_on_axis_source = cam.getFrame()

        # ef = prop.outputs['out_lgs1_source_ef']
        # phase_lgs1_source = cpuArray(ef.phaseInNm)
        # applyPhaseOnSlm(slm, phase_lgs1_source)
        # frame_lgs1_source = cam.getFrame()

        # rtc.step(frame_lgs1_source, frame_lgs2_source)
        # dm_layer.set_value(1, rtc.dm_command_in_nm)


# class Bronte(BaseProcessingObj):
#
#     SLM_RESPONSE_TIME = 0.005
#
#     def __init__(self, factory, target_device_idx=None, precision=None):
#         super().__init__(target_device_idx, precision)
#         self._slm = factory.deformable_mirror
#         self._sh_camera = factory.sh_camera
#         self.slm_raster = factory.slm_rasterizer
#         self.output_frame = Pixels(*self._sh_camera.shape())
#         self.outputs['out_pixels'] = self.output_frame
#         self.inputs['ef'] = InputValue(type=ElectricField)
#
#         self.fig, self.axs = plt.subplots(2)
#         self.first = True
#
#     def trigger_code(self):
#         ef = self.local_inputs['ef']
#         phase_screen = cpuArray(ef.phaseInNm) * 1e-9
#
#         phase_screen_to_raster = self.slm_raster.get_recentered_phase_screen_on_slm_pupil_frame(phase_screen)
#         command = self.slm_raster.reshape_map2vector(phase_screen_to_raster)
#         self._slm.set_shape(command)
#         time.sleep(self.SLM_RESPONSE_TIME)
#         #TODO: manage the different integration times for the each wfs group
#         # how to reproduce faint source? shall we play with the texp of the hardware?
#         camera_frame = self._sh_camera.getFutureFrames(1, 1).toNumpyArray()
#
#         if self.first:
#             self.img0 = self.axs[0].imshow(camera_frame)
#             self.img1 = self.axs[1].imshow(phase_screen_to_raster)
#             self.first = False
#         else:
#             self.img0.set_data(camera_frame)
#             self.img1.set_data(phase_screen_to_raster)
# #            self.img.set_clim(frame.min(), frame.max())
#         self.fig.canvas.draw()
#         plt.pause(0.001)       
#
#         self.output_frame.pixels = camera_frame
#         self.output_frame.generation_time = self.current_time
#
#     def run_check(self, time_step):
#         return True

if __name__ == '__main__':
    test_atmo()
