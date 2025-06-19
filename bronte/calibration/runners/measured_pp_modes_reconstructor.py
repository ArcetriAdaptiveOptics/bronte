import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np
from specula.processing_objects.modalrec import Modalrec
from specula.data_objects.recmat import Recmat
from bronte.startup import  set_data_dir
from bronte.package_data import reconstructor_folder, other_folder
from astropy.io import fits
from bronte.utils.set_basic_logging import get_logger
from arte.utils.decorator import logEnterAndExit 
from functools import cached_property

class PushPullModesMeasurer():
    
    def __init__(self, calib_factory, recmat_tag, xp=np):
        
        self._logger = get_logger("PushPullModesMeasurer")
        self._recmat_tag = recmat_tag
        self._factory = calib_factory
        if self._factory.PP_VCT_LOADED is not True:
            self._logger.info("WARNING: Push-Pull vector not Loaded! Load it from factory")
        self._Nmodes = self._factory.N_MODES_TO_CORRECT 
        self._build_processing_objects()        
        self._set_inputs()
        self._define_groups()
        self._initialize_modes_buffer()
        
    @logEnterAndExit("Creating ProcessingObjects...",
                  "ProcessingObjects created.", level='debug')
    def _build_processing_objects(self):
        
        self._prop = self._factory.disturb_propagation
        self._slopec = self._factory.slope_computer
        self._dm = self._factory.virtual_deformable_mirror
        self._bench_devices = self._factory.testbench_devices
        self._pp = self._factory.push_pull
        self._empty_layer = self._factory.empty_layer
        self._empty_layer.generation_time = 0
        self._rec = self.load_reconstructor
        
    @cached_property
    def load_reconstructor(self):
        recmat = Recmat.restore(reconstructor_folder() / (self._recmat_tag+ "_bronte_rec.fits"))
        #added factor 2 missed on IFs normalization
        N_pp = 2
        recmat.recmat = N_pp*recmat.recmat  
        return Modalrec(self._Nmodes, recmat=recmat)
        
    
    @logEnterAndExit("Setting ProcessingObjects inputs ...",
                      "ProcessingObjects inputs set.", level='debug')
    def _set_inputs(self):
        
        self._bench_devices.inputs['ef'].set(self._prop.outputs['out_on_axis_source_ef'])
        self._slopec.inputs['in_pixels'].set(self._bench_devices.outputs['out_pixels'])
        self._dm.inputs['in_command'].set(self._pp.output)
        self._prop.inputs['atmo_layer_list'].set([self._empty_layer])
        self._prop.inputs['common_layer_list'].set([self._dm.outputs['out_layer']])
        self._rec.inputs['in_slopes'].set(self._slopec.outputs['out_slopes'])
    
    def _define_groups(self):
        
        group1 = [self._pp]
        group2 = [self._dm]
        group3 = [self._prop]
        group4 = [self._bench_devices]
        group5 = [self._slopec]
        group6 = [self._rec]
        #group7 = []
        
        self._groups = [group1, group2, group3, group4, group5, group6]#, group7]
    
    
    @logEnterAndExit("Setting telemetry buffer...",
                      "Telemetry buffer set.", level='debug')   
    def _initialize_modes_buffer(self):
        
        self._rec_modes_list = []
        self._slopes_vector_list = []
        
    @logEnterAndExit("Updating reconstructed modes buffer...",
                  "reconstructed modes updated.", level='debug')  
    def _update_rec_modes(self):
        
        specula_slopes = self._groups[4][0].outputs['out_slopes']
        self._slopes_vector_list.append(specula_slopes.slopes.copy())   
        rec_modes_in_nm = self._groups[5][0].modes.value
        self._rec_modes_list.append(rec_modes_in_nm)
        
    @logEnterAndExit("Starting Push-Pull...",
          "Push-Pull Terminated.", level='debug')
    def run(self):
        
        self.time_step = self._factory.TIME_STEP_IN_SEC
        self._n_steps = 2 * self._Nmodes *2
        tf = (self._n_steps - 1) * self.time_step
        
        for group in self._groups:
            for obj in group:
                obj.loop_dt = self.time_step * 1e9
                obj.setup(self.time_step * 1e9, self._n_steps)
        
        for step in range(self._n_steps):
            t = 0 + step * self.time_step
            self._logger.info(
                "\n+ Push/Pull @ time: %f/%f s\t steps: %d/%d" % (t, tf, step+1, self._n_steps))
            for group in self._groups:
                for obj in group:
                    obj.check_ready(t*1e9)
                    self._logger.info(f"Triggering {str(obj)}")
                    obj.trigger()
                    obj.post_trigger()
            self._update_rec_modes()
            
        for group in self._groups:
            for obj in group:
                obj.finalize()
    
    @logEnterAndExit("Saving data...",
          "Data saved.", level='debug')        
    def save(self, ftag):
        
        file_name = other_folder()/(ftag + '.fits')
        hdr = fits.Header()
        hdr['REC_TAG'] = self._recmat_tag
        hdr['SOFF_TAG'] = self._factory.SLOPE_OFFSET_TAG
        rec_modes = np.array(self._rec_modes_list)
        pp_vector_in_nm = self._factory._pp_ampl_vect
        slopes_vector = np.array(self._slopes_vector_list)
        fits.writeto(file_name, rec_modes, hdr)
        fits.append(file_name, pp_vector_in_nm)
        fits.append(file_name, slopes_vector)
        
    @staticmethod
    def load(ftag):
        set_data_dir()
        file_name = other_folder()/(ftag + '.fits')
        hdr = fits.getheader(file_name)
        hduList = fits.open(file_name)
        rec_modes = hduList[0].data
        pp_vector_in_nm = hduList[1].data
        slopes_vector = hduList[2].data
        return rec_modes, pp_vector_in_nm, slopes_vector, hdr