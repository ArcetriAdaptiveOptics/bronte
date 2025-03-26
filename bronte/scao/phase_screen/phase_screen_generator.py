import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np, cpuArray
from bronte.startup import set_data_dir
from bronte.package_data import phase_screen_folder
from astropy.io import fits 
import h5py

class PhaseScreenGenerator():
    
    PROPAGATION_DIR = 'on_axis' #'lgs1'
    
    def __init__(self, scao_factory):
        
        self._factory = scao_factory
        self._setup_atmosphere()
        self._set_inputs()
        self._define_groups()
        self._phase_screen_list = []
    

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
        
    def _define_groups(self):
        
        group1 = [self._seeing, self._wind_speed, self._wind_direction]
        group2 = [self._atmo]
        group3 = [self._prop]
        self._groups = [group1, group2, group3]
        
    def _update_phase_screen_list(self):
        
        ef_output = 'out_'+self.PROPAGATION_DIR+'_source_ef'
        ef = self._groups[2][0].outputs[ef_output]
        phase_screen_in_um = cpuArray(ef.phaseInNm) * 1e-3
        #phase_screen_to_raster = self._slm_raster.get_recentered_phase_screen_on_slm_pupil_frame(phase_screen)
        self._phase_screen_list.append(phase_screen_in_um)
        
    def run(self, Nsteps = 30):
        
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
                    
            self._update_phase_screen_list()
            
        for group in self._groups:
            for obj in group:
                obj.finalize()
    
    def save_phase_screens(self, ftag):
        
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
        
        fits.writeto(file_name, np.array([]), hdr)
        self._save_phase_screens_hdf5(ftag, np.array(self._phase_screen_list))
    
    def _save_phase_screens_hdf5(self, ftag, data):
        frame_sizeY = self._phase_screen_list[0].shape[0]
        frame_sizeX = self._phase_screen_list[0].shape[1]
        file_name = phase_screen_folder() / (ftag + '.h5')
        with h5py.File(file_name, 'w') as f:
            # Create dataset with gzip compression and chunking
            f.create_dataset(
                'phase_screen', 
                data=data.astype(np.float16), 
                compression='gzip',
                compression_opts = 7, 
                chunks=(1, frame_sizeY, frame_sizeX)  # Save frame-wise chunks
            )
    @staticmethod
    def load_phase_screen_header(ftag):
        set_data_dir()
        header_file_name = phase_screen_folder() / (ftag + '.fits')
        header = fits.getheader(header_file_name)
        return  header
    
    @staticmethod
    def load_phase_screen_hdf5(fname, frame_idx=None):
        file_name = phase_screen_folder() / (fname + '.h5')
        with h5py.File(file_name, 'r') as f:
            if frame_idx is None:
                return f['phase_screen'][:]  # Load full dataset
            else:
                return f['phase_screen'][frame_idx]  # Load specific frame
