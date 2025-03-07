import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np
from specula.processing_objects.im_rec_calibrator import ImRecCalibrator

from specula.data_objects.source import Source
from specula.processing_objects.atmo_propagation import AtmoPropagation
from specula.processing_objects.atmo_evolution import AtmoEvolution
from specula.processing_objects.func_generator import FuncGenerator
from specula.processing_objects.int_control import IntControl
from specula.processing_objects.sh_slopec import ShSlopec
from specula.processing_objects.dm import DM
from specula.processing_objects.modalrec import Modalrec
from specula.data_objects.subap_data import SubapData
from specula.data_objects.recmat import Recmat
from specula.data_objects.layer import Layer
from specula.display.modes_display import ModesDisplay

from bronte.startup import startup
from bronte.types.testbench_device_manager import TestbenchDeviceManager
from bronte.package_data import subaperture_set_folder, reconstructor_folder,\
    phase_screen_folder, temp_folder
from astropy.io import fits

class TestSpecula():
    
    def __init__(self, calib_rec = False, target_device_idx=-1, xp=np):
        #calib_rec = False
        self._calib_rec = calib_rec
    
        self._factory = startup()
        telescope_pupil_diameter = 40
        pupil_diameter_in_pixel  = 2 * self._factory.slm_pupil_mask.radius()
        pupil_pixel_pitch = round(telescope_pupil_diameter/pupil_diameter_in_pixel, 3)
        
        seeing = FuncGenerator(constant=0.3,
                               target_device_idx=target_device_idx)
        wind_speed = FuncGenerator(constant=[25.5, 5.5],
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
                             data_dir = phase_screen_folder(),      # Data directory for phasescreens
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
    
        subapdata = SubapData.restore_from_bronte(
            subaperture_set_folder() / (self._factory.SUBAPS_TAG + ".fits"))#"241202_172000.fits" #"241129_162300.fits"  #240807_152700.fits
        
        slopec = ShSlopec(subapdata= subapdata)
    
        nslopes = subapdata.n_subaps * 2 #1510*2
        nModes = self._factory.N_MODES_TO_CORRECT # 3#21
        
        if self._calib_rec:
            recmat = Recmat(recmat=np.zeros((nModes, nslopes)))
            rec = Modalrec(nModes, recmat=recmat)
        else:
            #recmat = Recmat.restore(reconstructor_folder() / "241129_183600_bronte_rec.fits")
    #        recmat = Recmat.restore(reconstructor_folder() / "241202_174000_bronte_rec.fits")
            recmat = Recmat.restore(reconstructor_folder() / "250127_155400_bronte_rec.fits")
            modal_offset= np.zeros(nModes)
            #num_of_modal_offset = len(self._factory.modal_offset)
            if self._factory.MODAL_OFFSET_TAG is not None:
                modal_offset = self._factory.modal_offset[:nModes]
            rec = Modalrec(nModes, recmat=recmat, modal_offset=modal_offset)
    
        int_gains = np.ones(nModes)*-0.3
        #int_gains = np.zeros(nModes); int_gains[0:3]=-0.5  
        control = IntControl(delay=2, int_gain=int_gains)
        dm = DM(type_str='zernike',
                pixel_pitch=pupil_pixel_pitch,
                nmodes=nModes,
                npixels= pupil_diameter_in_pixel,                    # linear dimension of DM phase array
                obsratio= 0,                    # obstruction dimension ratio w.r.t. diameter
                height=  0)     # DM height [m]
      
        
        self._factory.sh_camera.setExposureTime(8)
       
        bronte = TestbenchDeviceManager(self._factory, 
                                        do_plots=True,
                                        target_device_idx=target_device_idx)
    
        ######################
        # CALIBRATION objects
            
        ## Not Used. We use SubapData.restore_from_bronte()    
        # sh_subap_calibrator= ShSubapCalibrator(
        #     data_dir = 'calib/bronte',
        #     subap_on_diameter= 46,
        #     output_tag='bronte_subaps_n46_th0.5',
        #     energy_th=         0.5,
        #     target_device_idx=target_device_idx)
        # sh_subap_calibrator.inputs['in_pixels'].set(bronte.outputs['out_pixels'])
        
        
        ampla = np.ones(nModes+10)
        ampla[0:3]=1
        #ampla[2]= 0 # Focus
        ampla[3:5]=0.2
        ampla[5:14]=0.15
        ampla[14:54]=0.1
        ampla[54:]=0.08
        amplv = ampla[0:nModes]*1000 
        pp = FuncGenerator(func_type= 'PUSHPULL',
                           nmodes=nModes,
                           vect_amplitude = amplv,#in nm
                           target_device_idx=target_device_idx)
    
        im_calibrator = ImRecCalibrator(
                            data_dir = reconstructor_folder(),
                            nmodes=nModes,
                            rec_tag='250123_144400_bronte_rec',
                            im_tag='250123_144400_bronte_im',
                            target_device_idx=target_device_idx)
    
    
        im_calibrator.inputs['in_slopes'].set(slopec.outputs['out_slopes'])
        im_calibrator.inputs['in_commands'].set(pp.output)
    
        ########################
        # MAIN
    
        empty_layer = Layer(pupil_diameter_in_pixel, pupil_diameter_in_pixel, pupil_pixel_pitch, height=0)
        empty_layer.generation_time = 0
        
        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)
        prop.inputs['layer_list'].set(atmo.layer_list + [dm.outputs['out_layer']])
    
        bronte.inputs['ef'].set(prop.outputs['out_on_axis_source_ef'])
        slopec.inputs['in_pixels'].set(bronte.outputs['out_pixels'])
        rec.inputs['in_slopes'].set(slopec.outputs['out_slopes'])
        control.inputs['delta_comm'].set(rec.out_modes)
        dm.inputs['in_command'].set(control.out_comm)
    
        if self._calib_rec:
            dm.inputs['in_command'].set(pp.output)
            prop.inputs['layer_list'].set([empty_layer, dm.outputs['out_layer']])
            self._n_steps = nModes * 2
        else:
            self._n_steps = 30
    
        disp = ModesDisplay()
        disp.inputs['modes'].set(rec.out_modes)

        if self._calib_rec:
            group1 = [seeing, wind_speed, wind_direction, pp]
            group2 = [dm, atmo]
            group3 = [prop]
            group4 = [bronte]
            group5 = [slopec]
            group6 = [rec, im_calibrator]
            group7 = [control]
            group8 = []
        else:
            group1 = [seeing, wind_speed, wind_direction]
            group2 = [atmo]
            group3 = [prop]
            group4 = [bronte]
            group5 = [slopec]
            group6 = [rec]
            group7 = [control, disp]
            group8 = [dm]

        self._groups = [group1, group2, group3, group4, group5, group6, group7, group8]

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
            # if self._calib_rec:
            #     fits.writeto( 
            #         temp_folder() / f'250123_111600_frame{step}.fits',
            #         self._groups[3][0].outputs['out_pixels'].pixels
            #         )
            
        for group in self._groups:
            for obj in group:
                obj.finalize()


if __name__ == '__main__':
    tst = TestSpecula()
    tst.run()
