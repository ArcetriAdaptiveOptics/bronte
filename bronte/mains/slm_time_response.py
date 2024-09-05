import numpy as np
from bronte.wfs.slm_rasterizer import SlmRasterizer
from plico_dm import deformableMirror

class SlmResposeTime():
    
    #zernike coefficients to direct the beam on the photodiodes
    ZERNIKE_COEFF_TO_PD1 = np.array([0,0,0])
    ZERNIKE_COEFF_TO_PD2 = np.array([-8.40512e-05, -1.05064e-05])
    
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
    
    def get_tiptilt_coeff_from_desplacement(self, dx = -8e-3, dy = -1e-3, f=250e-3, D=2*571*9.2e-6):
        
        c2 = dx * D /(4*f)
        c3 = dy * D /(4*f)
        
        return np.array([c2,c3])
    
    def apply_tiptilt_from_desplacement(self, dx=-8e-3, dy=-1e-3, f=250e-3, D=2*571*9.2e-6):
        
        tt_coeff = self.get_tiptilt_coeff_from_desplacement(dx, dy, f, D)
        ztt_coeff = self._sr.get_zernike_coefficients_from_numpy_array(tt_coeff)
        tt = self._sr.zernike_coefficients_to_raster(ztt_coeff).toNumpyArray()
        tt_cmd = self._sr.reshape_map2vector(tt)
        self._slm.set_shape(tt_cmd)
    
    @staticmethod
    def get_data_from_DAQamiCSVfile(csv_file_name):
        
        Nrows_to_skip = 7
        import csv
        with open(csv_file_name, newline='') as f:
            reader = csv.reader(f)
            time_in_sec = []
            pd1_voltage_in_volt = []
            pd2_voltage_in_volt = []
            idx=0
            
            for row in reader:
                #print(idx)
                if idx>=Nrows_to_skip:
                    time_in_sec.append(float(row[1]))
                    pd1_voltage_in_volt.append(float(row[2]))
                    pd2_voltage_in_volt.append(float(row[3]))
                idx+=1
        return np.array(time_in_sec), np.array(pd1_voltage_in_volt), np.array(pd2_voltage_in_volt)
        