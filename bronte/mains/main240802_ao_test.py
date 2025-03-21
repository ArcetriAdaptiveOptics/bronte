import numpy as np
import matplotlib.pyplot as plt
from arte.types.zernike_coefficients import ZernikeCoefficients
import time
import logging
from bronte import subapertures_initializer
from astropy.io import fits
from bronte import package_data
from arte.utils.decorator import logEnterAndExit


def define_subap_set(shwfs, slm, corner_xy=(0, 0), nsubaps=50, flux_threshold=100000):
    '''
    used to save subapertures 240802_122800
    '''
    slm.set_shape(np.zeros(1152*1920))
    wf_ref = shwfs.getFutureFrames(1, 20).toNumpyArray()
    sgi = subapertures_initializer.main(
        wf_ref, corner_xy=corner_xy, nsubaps=nsubaps, flux_threshold=flux_threshold)
    return sgi


class TestAoLoop:
    
    SLM_RESPONSE_TIME = 0.005

    def __init__(self, factory):
        self._factory = factory
        self._logger = logging.getLogger("TestAoLoop")
        self._display_in_loop = False
        self._t = 0
        self._wavefront_disturb = None
        self.setup_disturb()
        self._long_exp = 0
        self._initialize_telemetry()
        plt.ion()

    def enable_display_in_loop(self, true_or_false):
        self._display_in_loop = true_or_false

    def setup_disturb(self):
        ps = self._factory.phase_screen_generator.get_in_meters()[0, 0:1152, :]
        self.load_wavefront_disturb(ps)


    def loop(self, how_many=10):
        # if self._display_in_loop:
        #     fig, axs = plt.subplots(2, 4, layout = "constrained")
            
        for i in range(how_many):
            self._logger.info("loop %d/%d" % (i+1, how_many))
            self.step()
        self.display()

    @logEnterAndExit("Stepping...", "Stepped", level='debug')
    def step(self):
        self._t += 1
        self.set_wavefront_disturb(self._t)
        self._factory.rtc.step()
        time.sleep(self.SLM_RESPONSE_TIME)
        self.integrate_long_exposure()
        self._update_telemetry()
        if self._display_in_loop:
            self.display()
            # self.display2(fig, axs)

    def set_wavefront_disturb(self, temporal_step, wind_speed = 4):
        if self._wavefront_disturb is None:
            return
        self._wind_speed = wind_speed  # in phase screen/step
        roll_by = temporal_step * self._wind_speed
        self._factory.rtc.set_wavefront_disturb(
            np.roll(self._wavefront_disturb, (roll_by, 0))
        )

    def load_wavefront_disturb(self, wavefront_disturb):
        self._wavefront_disturb = wavefront_disturb

    def reset_wavefront_disturb(self):
        self._factory.rtc.reset_wavefront_disturb()
        self._wavefront_disturb = None

    @logEnterAndExit("Integrating long exposure...", "Long exposure integrated", level='debug')
    def integrate_long_exposure(self):
        self._short_exp = self._factory.psf_camera.getFutureFrames(
            1, 1).toNumpyArray()
        self._long_exp += self._short_exp.astype(float)

    def reset_long_exposure(self):
        self._long_exp = 0
    
    def reset_telemetry(self):
        self._initialize_telemetry()
        
    def _initialize_telemetry(self):
        self._slopes_x_maps_list = []
        self._slopes_y_maps_list = []
        self._short_exp_psf_list = []
        self._zc_delta_modal_command_list = []
        self._zc_integrated_modal_command_list = []
    
    @logEnterAndExit("Updating telemetry...", "Telemetry updated", level='debug')
    def _update_telemetry(self):
        # TODO just save slope vector instead of map
        self._slopes_x_maps_list.append(
            self._factory.slope_computer.slopes_x_map())
        self._slopes_y_maps_list.append(self._factory.slope_computer.slopes_y_map())
        self._short_exp_psf_list.append(self._short_exp)
        self._zc_delta_modal_command_list.append(
            self._factory.rtc._delta_modal_command_buffer.get(self._t)
            )
        self._zc_integrated_modal_command_list.append(
            self._factory.pure_integrator_controller.command()
            )
        
    def display_sh_ima(self):
        sh_ima = self._factory.sh_camera.getFutureFrames(1, 1).toNumpyArray()
        plt.figure(2)
        plt.clf()
        plt.imshow(self._factory.slope_computer.subapertures_map()*1000+sh_ima)
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.2)
 
    @logEnterAndExit("Refreshing display...", "Display refreshed", level='debug')
    def display(self):
        
        plt.figure(1)
        plt.clf()
        plt.imshow(self._short_exp)#[290:390, 580:680])
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.2)

        self.display_sh_ima()

        plt.figure(3)
        self.show_slopes_x_maps()
        plt.figure(4)
        self.show_slopes_y_maps()

        plt.figure(5)
        plt.clf()
        plt.plot(self._factory.slope_computer.slopes()[:, 0])
        plt.plot(self._factory.slope_computer.slopes()[:, 1])
        plt.show(block=False)
        plt.pause(0.2)

        plt.figure(6)
        plt.clf()
        plt.imshow(self._factory.slm_rasterizer.reshape_vector2map(
            self._factory.deformable_mirror.get_shape()))
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.2)

        plt.figure(7)
        plt.clf()
        zc = ZernikeCoefficients.fromNumpyArray(
            self._factory.pure_integrator_controller.command())
        plt.plot(zc.modeIndexes(), zc.toNumpyArray(), '.-')
        plt.grid(True)
        plt.ylabel('integrated modal coefficient')
        plt.xlim(0, 20)
        plt.show(block=False)
        plt.pause(0.2)

        plt.figure(8)
        plt.clf()
        zc = self._factory.rtc._compute_zernike_coefficients()
        plt.plot(zc.modeIndexes(), zc.toNumpyArray(), '.-')
        plt.grid(True)
        plt.ylabel('delta modal coefficient')
        plt.xlim(0, 20)
        plt.show(block=False)
        plt.pause(0.2)


    def show_slopes_x_maps(self):
        sc = self._factory.slope_computer
        plt.clf()
        plt.title("Slope X")
        plt.imshow(sc.slopes_x_map())
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.2)

    def show_slopes_y_maps(self):
        sc = self._factory.slope_computer
        plt.clf()
        plt.title("Slope Y")
        plt.imshow(sc.slopes_y_map())
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.2)
        
    def _delete_short_exp_psf(self):
        self._short_exp_psf_list = np.zeros(3)
        
    def save_telemetry(self, fname):
        
        psf_camera_texp = self._factory.psf_camera.exposureTime()
        psf_camera_fps = self._factory.psf_camera.getFrameRate()
        shwfs_texp = self._factory.sh_camera.exposureTime()
        shwfs_fps = self._factory.sh_camera.getFrameRate()
        
        file_name = package_data.telemetry_folder() / (fname + '.fits')
        hdr = fits.Header()
        
        
        #FILE TAG DEPENDENCY
        hdr['SUB_TAG'] = self._factory.SUBAPS_TAG
        hdr['PS_TAG'] = self._factory.PHASE_SCREEN_TAG
        hdr['MD_TAG'] = self._factory.MODAL_DEC_TAG
        
        #ATMO PARAMETERS
        if self._wavefront_disturb is None:
            self._wind_speed = 0
            self._factory._r0 = 0
        hdr['R0_IN_M'] = self._factory._r0
        hdr['WIND_SP'] = self._wind_speed # in phase screen/step
        
        #LOOP PARAMETERS
        #hdr['AO_STAT'] = self._ao_status # 'open' or 'closed'
        hdr['INT_TYPE'] = self._factory.pure_integrator_controller._integrator_type
        hdr['INT_GAIN'] = self._factory.pure_integrator_controller._gain
        hdr['N_STEPS'] = self._t
        
        #HARDWARE PARAMETERS

        hdr['PC_TEXP'] = psf_camera_texp # in ms
        hdr['PC_FPS'] = psf_camera_fps
        hdr['SH_TEXP'] = shwfs_texp # in ms
        hdr['SH_FPS'] = shwfs_fps
        hdr['SLM_RT'] = self.SLM_RESPONSE_TIME # in sec
        
        fits.writeto(file_name, self._long_exp, hdr)
        fits.append(file_name, np.array(self._short_exp_psf_list))
        fits.append(file_name, np.array(self._slopes_x_maps_list))
        fits.append(file_name, np.array(self._slopes_y_maps_list))
        
        #CONTROL MATRICES
        fits.append(file_name, self._factory.modal_decomposer._lastIM)
        fits.append(file_name, self._factory.modal_decomposer._lastReconstructor)
        
        #MODAL COMMANDS
        fits.append(file_name, np.array(self._zc_delta_modal_command_list))
        fits.append(file_name, np.array(self._zc_integrated_modal_command_list))
        
        #OFFSETS
        fits.append(file_name, self._factory.rtc.modal_offset.toNumpyArray())
        
    @staticmethod
    def load_telemetry(fname):
        
        file_name = package_data.telemetry_folder() / (fname + '.fits')
        header = fits.getheader(file_name)
        hduList = fits.open(file_name)
        
        #FILE TAG DEPENDENCY
        SUBAPS_TAG = header['SUB_TAG']
        PHASE_SCREEN_TAG = header['PS_TAG']
        MODAL_DEC_TAG = header['MD_TAG']
        tag_list = [SUBAPS_TAG, PHASE_SCREEN_TAG, MODAL_DEC_TAG]
        
        #ATMO PARAMETERS
        r0 = header['R0_IN_M']
        wind_speed = header['WIND_SP'] # in phase screen/step
        atmospheric_param_list = [r0, wind_speed]
        
        #LOOP PARAMETERS
        #ao_status = header['AO_STAT']
        integrator_type = header['INT_TYPE']
        int_gain = header['INT_GAIN']
        loop_steps = header['N_STEPS']
        loop_param_list = [integrator_type, int_gain, loop_steps]
        
        #HARDWARE PARAMETERS
        psf_camera_texp = header['PC_TEXP']
        psf_camera_fps = header['PC_FPS']
        shwfs_texp = header['SH_TEXP']
        shwfs_fps = header['SH_FPS']
        slm_response_time = header['SLM_RT']
        hardware_param_list = [psf_camera_texp, psf_camera_fps,\
                                shwfs_texp, shwfs_fps, slm_response_time]
        
        #FRAMES
        long_exp_psf = hduList[0].data
        short_exp_psfs = hduList[1].data
        slopes_x_maps = hduList[2].data
        slopes_y_maps = hduList[3].data
        
        #CONTROL MATRICES
        #TO DO: to fix cached prorerties are not picklable
        # save the slope2modal_coeff matrix (rec_mat)
        # and the modal_coeff2command/wf matrix (int_mat)
        interaction_matrix = hduList[4].data
        reconstructor = hduList[5].data
        
        #MODAL COMMANDS
        zc_delta_modal_commands = hduList[6].data
        zc_integrated_modal_commands  = hduList[7].data
        
        #OFFSETS
        zc_modal_offset = hduList[8].data
        
        return tag_list,\
             atmospheric_param_list,\
              loop_param_list,\
               hardware_param_list,\
                long_exp_psf,\
                 short_exp_psfs,\
                  slopes_x_maps, slopes_y_maps,\
                  interaction_matrix, reconstructor,\
                    zc_delta_modal_commands, zc_integrated_modal_commands,\
                        zc_modal_offset