import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np, cpuArray
from bronte.startup import set_data_dir
from bronte.package_data import phase_screen_folder
from arte.types.wavefront import Wavefront
from arte.utils.circular_buffer import NumpyCircularBuffer
from astropy.io import fits 
import h5py

class PhaseScreenGenerator():
    
    PROPAGATION_DIR = 'on_axis' #'lgs1'
    STORE_PHASE_SCREENS = True
    BUFFER_SIZE = 100
    
    def __init__(self, scao_factory):
        
        self._factory = scao_factory
        self._slm_raster = self._factory.slm_rasterizer
        self._slm_mask = self._factory.slm_pupil_mask
        self._slm_mask_diameter = 2*self._slm_mask.radius()
        self._setup_atmosphere()
        self._set_inputs()
        self._define_groups()
        self._initialise_telemetry_buffers()
    

    def _setup_atmosphere(self):
        
        self._seeing = self._factory.seeing
        self._wind_speed = self._factory.wind_speed 
        self._wind_direction = self._factory.wind_direction
        self._atmo = self._factory.atmo_evolution
        self._prop = self._factory.atmo_propagation
        
    def _set_inputs(self):
        self._atmo.inputs['seeing'].set(self._seeing.output)
        self._atmo.inputs['wind_direction'].set(self._wind_direction.output)
        self._atmo.inputs['wind_speed'].set(self._wind_speed.output)
        self._prop.inputs['layer_list'].set(self._atmo.layer_list)
    
        #self._bench_devices.inputs['ef'].set(self._prop.outputs['out_on_axis_source_ef'])
    
    def _initialise_telemetry_buffers(self):

        frame_shape = (self._slm_mask_diameter, self._slm_mask_diameter)
        self._phase_screen_buffer = NumpyCircularBuffer(self.BUFFER_SIZE, frame_shape, dtype=np.float16)
        self._modal_coefficients_list = []
    
    def _define_groups(self):
        
        group1 = [self._seeing, self._wind_speed, self._wind_direction]
        group2 = [self._atmo]
        group3 = [self._prop]
        self._groups = [group1, group2, group3]
        
    def _update_telemetry_buffers(self):
        
        ef_output = 'out_'+self.PROPAGATION_DIR+'_source_ef'
        ef = self._groups[2][0].outputs[ef_output]
        phase_screen = cpuArray(ef.phaseInNm)
        
        if self.STORE_PHASE_SCREENS is True:
            self._phase_screen_buffer.store(phase_screen)
            if self._phase_screen_buffer.counter() == self._phase_screen_buffer._len:
                self._save_phase_screens_hdf5(self._ftag_hdf5, self._phase_screen_buffer._buf )
        
        phase_screen_on_slm_pupil = self._slm_raster.get_recentered_phase_screen_on_slm_pupil_frame(phase_screen* 1e-9)
        wfz = Wavefront.fromNumpyArray(phase_screen_on_slm_pupil)
        modal_coefficents = self._slm_raster._zernike_modal_decomposer.measureModalCoefficientsFromWavefront(
            wfz,
            self._slm_mask,
            self._slm_mask)
        
        self._modal_coefficients_list.append(modal_coefficents.toNumpyArray())
        
    def run(self, Nsteps = 30, ftag_pahse_screen_hdf5 = None):
        
        if ftag_pahse_screen_hdf5 is not None:
            self.STORE_PHASE_SCREENS = True
            self._ftag_hdf5 = ftag_pahse_screen_hdf5
            
        self._n_steps = Nsteps
        # time step of the simulated loop isnt it QM
        self.time_step = self._factory.TIME_STEP_IN_SEC
        
        for group in self._groups:
            for obj in group:
                obj.loop_dt = self.time_step * 1e9
                obj.run_check(self.time_step)
    
        for step in range(self._n_steps):
            t = 0 + step * self.time_step
            print('T=',t)
            for group in self._groups:
                for obj in group:
                    obj.check_ready(t*1e9)
                    print('trigger', obj)
                    obj.trigger()
                    obj.post_trigger()
                    
            self._update_telemetry_buffers()
                
        for group in self._groups:
            for obj in group:
                obj.finalize()
    
    def save(self, ftag):
        
        file_name = phase_screen_folder() / (ftag + '.fits')
        hdr = fits.Header()
        
        # ATMO PARAMETERS
        hdr['PROP_DIR'] = self.PROPAGATION_DIR
        hdr['SEEING'] = self._factory.SEEING
        hdr['L0_IN_M'] = self._factory.OUTER_SCALE_L0
        hdr['D_IN_M'] = self._factory.TELESCOPE_PUPIL_DIAMETER
        hdr['WIND_SP'] = str(self._factory.WIND_SPEED_LIST)
        hdr['WIND_DIR'] = str(self._factory.WIND_DIR_LIST)
        hdr['CN2_W'] = str(self._factory.Cn2_WEIGHTS_LIST)
        hdr['NGS_POS'] = str(self._factory.ONAXIS_SOURCE_COORD)
        hdr['NGS_MAG'] = self._factory.ONAXIS_SOURCE_MAG
        hdr['NGS_WL'] = self._factory.ONAXIS_SOURCE_WL_IN_NM
        hdr['LGS_POS'] = str(self._factory.LGS_COORD)
        hdr['LGS_MAG'] = self._factory.LGS_MAG
        hdr['LGS_ALT'] = self._factory.LGS_HEIGHT_IN_M
        hdr['LGS_WL'] = self._factory.LGS_WL_IN_NM
        
        hdr['TSTEP_S'] = self.time_step # in seconds
        hdr['N_STEPS'] = self._n_steps
        #HARDWARE PARAMETERS
        hdr['SLM_RAD'] = self._factory.SLM_PUPIL_RADIUS # in pixels
        hdr['SLM_YX'] = str(self._factory.SLM_PUPIL_CENTER) # YX pixel coordinates
        
        fits.writeto(file_name, np.array(self._modal_coefficients_list), hdr)
        if self.STORE_PHASE_SCREENS is True:
            self._save_phase_screens_hdf5(ftag, np.array(self._phase_screen_list))
    
    #TODO: MemoryError must be resolved
    # emoryError: Unable to allocate 12.0 GiB
    # for an array with shape (5000, 1136, 1136) and data type float16
    #TODO: the function must not overwrite but add the data on existing file
    def _save_phase_screens_hdf5(self, ftag, data):
        frame_sizeY = self._slm_mask_diameter
        frame_sizeX = self._slm_mask_diameter
        file_name = phase_screen_folder() / (ftag + '.h5')
        with h5py.File(file_name, 'w') as f:
            # Create dataset with gzip compression and chunking
            f.create_dataset(
                'phase_screen', 
                data=data.astype(np.float32), 
                compression='gzip',
                compression_opts = 9, 
                chunks=(1, frame_sizeY, frame_sizeX)  # Save frame-wise chunks
            )
            # # Espandi dataset e salva
            # f['phase_screen'].resize((f['phase_screen'].shape[0] + self._phase_screen_buffer.counter(),) + f['phase_screen'].shape[1:])
            # f['phase_screen'][-self._phase_screen_buffer.counter():] = self._phase_screen_buffer._buf
            # Svuota il buffer dopo il salvataggio
            self._phase_screen_buffer = NumpyCircularBuffer(self.BUFFER_SIZE, (frame_sizeY, frame_sizeX), dtype=np.float32)
            
    @staticmethod
    def load_phase_screen_fits_data(ftag):
        set_data_dir()
        file_name = phase_screen_folder() / (ftag + '.fits')
        header = fits.getheader(file_name)
        hduList = fits.open(file_name)
        modal_coefficients_cube = hduList[0].data
        return  header, modal_coefficients_cube
    
    @staticmethod
    def load_phase_screen_hdf5(fname, frame_idx=None):
        file_name = phase_screen_folder() / (fname + '.h5')
        with h5py.File(file_name, 'r') as f:
            if frame_idx is None:
                return f['phase_screen'][:]  # Load full dataset
            else:
                return f['phase_screen'][frame_idx]  # Load specific frame
