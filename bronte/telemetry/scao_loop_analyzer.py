import numpy as np

from bronte.mains.main240802_ao_test import TestAoLoop


class ScaoLoopAnalyzer():
    
    def __init__(self, fname):
        
        self._tag_list,\
         self._atmospheric_param_list,\
          self._loop_param_list,\
           self._hardware_param_list,\
            self._long_exp_psf,\
             self._short_exp_psfs,\
              self._slopes_x_maps,\
               self._slopes_y_maps,\
               self._interaction_matrix,\
               self._reconstructor = TestAoLoop.load_telemetry(fname)