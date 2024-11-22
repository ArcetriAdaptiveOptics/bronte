
from specula.data_objects.layer import Layer
from specula.data_objects.source import Source
from specula.processing_objects.atmo_propagation import AtmoPropagation
from specula.processing_objects.atmo_evolution import AtmoEvolution
from specula.processing_objects.func_generator import FuncGenerator
from specula.processing_objects.int_control import IntControl
from specula.processing_objects.sh_slopec import ShSlopec
from specula.processing_objects.dm import DM
import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1

from specula import np, cpuArray





def test_atmo(target_device_idx=-1, xp=np):
    
    seeing = FuncGenerator(constant=0.65,
                           target_device_idx=target_device_idx)
    wind_speed = FuncGenerator(constant=[5.5, 5.5, 5.1, 5.5, 5.6, 5.7, 5.8, 6.0, 6.5, 7.0, # m/s
                    7.5, 8.5, 9.5, 11.5, 17.5, 23.0, 26.0, 29.0, 32.0, 27.0,
                    22.0, 14.5, 9.5, 6.3, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0,
                    8.5, 9.0, 9.5, 10.0, 10.0],
                               target_device_idx=target_device_idx) 
    wind_direction = FuncGenerator(constant=[0, -180, 0, 0, 90, 180, 0, 0, 0, -180,    # deg
                    0, 0, -90, 0, 90, -180, 90, 0, -90, -90,
                    0, -90, 0, 0, 180, 180, 0, -180, 90, 0,
                    0, 180, -90, 90, -90],
                                   target_device_idx=target_device_idx)

    on_axis_source = Source(polar_coordinate=[0.0, 0.0], magnitude=8, wavelengthInNm=750,)
    lgs1_source = Source(polar_coordinate=[45.0, 0.0], height=90000, magnitude=5, wavelengthInNm=589)
    lgs2_source = Source(polar_coordinate=[45.0, 60.0], height=90000, magnitude=5, wavelengthInNm=589)
    lgs3_source = Source(polar_coordinate=[45.0, 120.0], height=90000, magnitude=5, wavelengthInNm=589)
    lgs4_source = Source(polar_coordinate=[45.0, 180.0], height=90000, magnitude=5, wavelengthInNm=589)
    lgs5_source = Source(polar_coordinate=[45.0, 240.0], height=90000, magnitude=5, wavelengthInNm=589)
    lgs6_source = Source(polar_coordinate=[45.0, 300.0], height=90000, magnitude=5, wavelengthInNm=589)

    atmo = AtmoEvolution(pixel_pupil=480,              # Linear dimension of pupil phase array
                         pixel_pitch= 0.0802,         # Linear dimension of pupil phase array
                         data_dir = 'calib/ELT',      # Data directory for phasescreens
                              L0=23,                        # [m] Outer scale
                         heights = [30.0000, 90.0000, 150.000, 200.000, 245.000, 300.000, 390.000, 600.000, 1130.00, 1880.00,
                        2630.00, 3500.00, 4500.00, 5500.00, 6500.00, 7500.00, 8500.00, 9500.00, 10500.0, 11500.0,
                        12500.0, 13500.0, 14500.0, 15500.0, 16500.0, 17500.0, 18500.0, 19500.0, 20500.0, 21500.0,
                        22500.0, 23500.0, 24500.0, 25500.0, 26500.0], # [m] layer heights at 0 zenith angle
                         Cn2 = [0.241954, 0.119977, 0.0968817, 0.0589889, 0.0472911, 0.0472911, 0.0472911, 0.0472911, 0.0398925, 0.0323939,
                        0.0161969, 0.0260951, 0.0155971, 0.0103980, 0.00999811, 0.0119977, 0.00400924, 0.0139974, 0.0129975, 0.00700868,
                        0.0159970, 0.0258951, 0.0190964, 0.00986813, 0.00616883, 0.00400924, 0.00246953, 0.00215959, 0.00184965, 0.00135974,
                        0.00110979, 0.000616883, 0.000925825, 0.000493907, 0.000431918], # Cn2 weights (total must be eq 1)
                        source_dict = {'on_axis_source': on_axis_source,
                                        'lgs1_source': lgs1_source,
                                        'lgs2_source': lgs2_source,
                                        'lgs3_source': lgs3_source,
                                        'lgs4_source': lgs4_source,
                                        'lgs5_source': lgs5_source,
                                        'lgs6_source': lgs6_source},
                        target_device_idx=target_device_idx,
                        )

    prop = AtmoPropagation(pixel_pupil=480,              # Linear dimension of pupil phase array
                           pixel_pitch= 0.0802,         # Linear dimension of pupil phase array
                           source_dict = {'on_axis_source': on_axis_source,
                                        'lgs1_source': lgs1_source,
                                        'lgs2_source': lgs2_source,
                                        'lgs3_source': lgs3_source,
                                        'lgs4_source': lgs4_source,
                                        'lgs5_source': lgs5_source,
                                        'lgs6_source': lgs6_source},
                           target_device_idx=target_device_idx)

    bronte = Bronte(bronte_factory.deformable_mirror, bronte_factory.sh_camera)

    slopec = ShSlopec(subapdata: SubapData)

    rec = ModalRec('scao_recmat')

    nModes = 54
    control = IntControl(delay=2, int_gain=np.ones(nModes)*0.5)
    dm = DM('zernike', nmodes=nModes)
    # dm_layer = Layer(480, 480, 0.0802, height=0, target_device_idx=target_device_idx)

    atmo.inputs['seeing'].set(seeing.output)
    atmo.inputs['wind_direction'].set(wind_direction.output)
    atmo.inputs['wind_speed'].set(wind_speed.output)
    prop.inputs['layer_list'].set([atmo.layer_list] + dm.out_layer[:-1])
    bronte.inputs['ef_in_pupil'].set(prop.outputs)
    slopec.inputs['in_pixels'].set(bronte.out_pixels)
    rec.inputs['in_slopes'].set(slopec.out_slopes)
    control.inputs['delta_comm'].set(rec.out_modes)
    dm.inputs['in_command'].set(control.out_comm)

    group1 = [seeing, wind_speed, wind_direction]
    group2 = [atmo, dm_layer]
    group3 = [prop]
    group4 = [bronte]
    group5 = [slopec]
    group6 = [rec]
    group7 = [control]
    group8 = [dm]

    time_step = 0.01
    
    for group in [group1, group2, group3]:
        for obj in group:
            obj.run_check(time_step)
    
    for step in range(1000):
        t = time_step + step * time_step
        print(t)
        for group in [group1, group2, group3]:
            for obj in group:
                obj.check_ready(t*1e9)
                obj.trigger()
                obj.post_trigger()
            
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


class Bronte(BaseProcessingObj):

    def __init__(self, slm_client, cam_client):
        self._slm = slm_client
        self._cam = cam_client

    def trigger_code(self):
        input_efs = self.local_inputs['ef_in_pupil']
        output_frames = []
        for ef in input_efs:
            self._slm.set_shape(ef)
            output_frames.append(self._cam.getFrame())
        self.outputs

        
if __name__ == '__main__':
    test_atmo()
