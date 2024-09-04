import numpy as np
from bronte.wfs.slm_rasterizer import SlmRasterizer
from plico_dm import deformableMirror

class SlmResposeTime():
    
    #zernike coefficients to direct the beam on the photodiodes
    ZERNIKE_COEFF_TO_PD1 = np.array([0,0,0])
    ZERNIKE_COEFF_TO_PD2 = np.array([-8.378854e-05, -7.196884e-06])
    
    def __init__(self):

        self._slm = deformableMirror('localhost', 7010)
        self._sr = SlmRasterizer()
    
    def get_command_to_go_on_photodiode1(self, coeff1 = None):
        if coeff1 is None:
            coeff1 = self.ZERNIKE_COEFF_TO_PD1
        zernike_coeff = self._sr.get_zernike_coefficients_from_numpy_array(coeff1)
        wf = self._sr.zernike_coefficients_to_raster(zernike_coeff).toNumpyArray()
        cmd1 = self._sr.reshape_map2vector(wf)
        return cmd1
    
    def get_command_to_go_on_photodiode2(self, coeff2 = None):
        if coeff2 is None:
            coeff2 = self.ZERNIKE_COEFF_TO_PD2
        zernike_coeff = self._sr.get_zernike_coefficients_from_numpy_array(coeff2)
        wf = self._sr.zernike_coefficients_to_raster(zernike_coeff).toNumpyArray()
        cmd2 = self._sr.reshape_map2vector(wf)
        return cmd2
    
    def apply_cmds(self, cmd1=None, cmd2=None, times=10):
        
        if cmd1 is None:
            cmd1 = self.get_command_to_go_on_photodiode1()
        if cmd2 is None:
            cmd2 = self.get_command_to_go_on_photodiode2()
        t=0
        
        while(t <=times):
            self._slm.set_shape(cmd1)
            self._slm.set_shape(cmd2)
            t+=1
        
        self._slm.set_shape(cmd1)
    
    def get_tiptilt_coeff_from_desplacement(self, dx = -15.95e-3, dy = -1.37e-3, f=250e-3, D=571*9.2e-6):
        
        c2 = dx * D /(4*f)
        c3 = dy * D /(4*f)
        
        return np.array([c2,c3])
    
    def apply_tiptilt_from_desplacement(self, dx=-15.95e-3, dy=-1.37e-3, f=250e-3, D=571*9.2e-6):
        
        tt_coeff = self.get_tiptilt_coeff_from_desplacement(dx, dy, f, D)
        ztt_coeff = self._sr.get_zernike_coefficients_from_numpy_array(tt_coeff)
        tt = self._sr.zernike_coefficients_to_raster(ztt_coeff).toNumpyArray()
        tt_cmd = self._sr.reshape_map2vector(tt)
        self._slm.set_shape(tt_cmd)
    
        
        