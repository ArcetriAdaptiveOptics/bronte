import numpy as np
from arte.types.mask import CircularMask
#from bronte.startup import set_data_dir
#from bronte.package_data import reconstructor_folder
#from astropy.io import fits
from arte.utils.modal_decomposer import ModalDecomposer
#from arte.utils.zernike_decomposer import ZernikeModalDecomposer
from arte.utils.zernike_generator import ZernikeGenerator
from arte.utils.rebin import rebin
from arte.types.slopes import Slopes

class ShSamplingOnZernikeAnalyser():
    
    
    
    def __init__(self, Nmodes):
        
        self._Nmodes = Nmodes
        