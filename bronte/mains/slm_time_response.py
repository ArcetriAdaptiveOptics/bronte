import numpy as np
from bronte.wfs.slm_rasterizer import SlmRasterizer
from plico_dm import deformableMirror
import logging
from arte.utils.decorator import logEnterAndExit

class SlmResposeTime():
    
    #zernike coefficients to direct the beam on the photodiodes
    ZERNIKE_COEFF_TO_PD1 = np.array([0,0,0])
    ZERNIKE_COEFF_TO_PD2 = np.array([-1.05064e-04,  5.25320e-06])
    
    def __init__(self):
        self._set_up_basic_logging()
        self._slm = deformableMirror('localhost', 7010)
        self._sr = SlmRasterizer()
        self._logger = logging.getLogger("SlmResponseTime")
    
    def _set_up_basic_logging(self):
        import importlib
        importlib.reload(logging)
        FORMAT = '%(asctime)s:%(levelname)s:%(name)s  %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    
    
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
    
    @logEnterAndExit("Appling sequential cmds focus on PDs",
                     "Sequential cmds ended on PD2", level='debug')
    def apply_cmds(self, cmd1=None, cmd2=None, times=10):
        
        if cmd1 is None:
            cmd1 = self.get_command_to_go_on_photodiode1()
        if cmd2 is None:
            cmd2 = self.get_command_to_go_on_photodiode2()
   
        for t in np.arange(times):
            self._slm.set_shape(cmd1)
            self._slm.set_shape(cmd2)
            
        
        #self._slm.set_shape(cmd1)
    # @logEnterAndExit("UGLY sequential cmds focus on PDs",
    #                  "Sequential cmds ended on PD2", level='debug')
    # def ugly_run_10times(self, cmd1=None, cmd2=None):
    #     if cmd1 is None:
    #         cmd1 = self.get_command_to_go_on_photodiode1()
    #     if cmd2 is None:
    #         cmd2 = self.get_command_to_go_on_photodiode2()
    #
    #     self._slm.set_shape(cmd1)
    #     self._slm.set_shape(cmd2)
    #     self._slm.set_shape(cmd1)
    #     self._slm.set_shape(cmd2)
    #     self._slm.set_shape(cmd1)
    #     self._slm.set_shape(cmd2)
    #     self._slm.set_shape(cmd1)
    #     self._slm.set_shape(cmd2)
    #     self._slm.set_shape(cmd1)
    #     self._slm.set_shape(cmd2)
    #     self._slm.set_shape(cmd1)
    #     self._slm.set_shape(cmd2)
    #     self._slm.set_shape(cmd1)
    #     self._slm.set_shape(cmd2)
    #     self._slm.set_shape(cmd1)
    #     self._slm.set_shape(cmd2)
    #     self._slm.set_shape(cmd1)
    #     self._slm.set_shape(cmd2)
    #     self._slm.set_shape(cmd1)
    #     self._slm.set_shape(cmd2)
    #

    
    def get_tiptilt_coeff_from_desplacement(self, dx = -10e-3, dy = 0.5e-3, f=250e-3, D=2*571*9.2e-6):
        
        c2 = dx * D /(4*f)
        c3 = dy * D /(4*f)
        
        return np.array([c2,c3])
    
    def apply_tiptilt_from_desplacement(self, dx=-10e-3, dy=0.5e-3, f=250e-3, D=2*571*9.2e-6):
        
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
                
                if idx>=Nrows_to_skip:
                    time_in_sec.append(float(row[1]))
                    pd1_voltage_in_volt.append(float(row[2]))
                    pd2_voltage_in_volt.append(float(row[3]))
                idx+=1
        return np.array(time_in_sec), np.array(pd1_voltage_in_volt), np.array(pd2_voltage_in_volt)
        
        
class ResponseTimeAnalyzer():
    
    def __init__(self, csv_file_name):
        
        self._fname = csv_file_name
        self._t, self._v1, self._v2 = SlmResposeTime().get_data_from_DAQamiCSVfile(csv_file_name)
    
    def display_data(self):
        import matplotlib.pyplot as plt
        
        plt.figure()
        plt.clf()
        plt.plot(self._t, self._v1, 'b-', label='PD1')
        plt.plot(self._t, self._v2, 'r-', label='PD2')
        plt.legend(loc='best')
        plt.grid('--',alpha=0.3)
        plt.xlabel('Time [s]')
        plt.ylabel('Voltage [V]')
        
    def display_data_indexing(self):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.clf()
        plt.plot( self._v1, 'b-', label='PD1')
        plt.plot(self._v2, 'r-', label='PD2')
        plt.legend(loc='best')
        plt.grid('--',alpha=0.3)
        plt.xlabel('Index')
        plt.ylabel('Voltage [V]')
        
    def display_moving_average_signals(self, window_size = 8):
        
        moving_average_v1 = []
        moving_average_v2 = []
        
        i=0
        while(i<len(self._v1) - window_size + 1):
            
            window_average_v1 = np.sum(self._v1[i : i + window_size])/window_size
            window_average_v2 = np.sum(self._v2[i : i + window_size])/window_size
            moving_average_v1.append(window_average_v1)
            moving_average_v2.append(window_average_v2)
        
        v1_mean = np.array(moving_average_v1)
        v2_mean = np.array(moving_average_v2)
        
        import matplotlib.pyplot as plt
        plt.figure()
        plt.clf()
        plt.plot(v1_mean, 'b-', label='PD1')
        plt.plot(v2_mean, 'r-', label='PD2')
        plt.legend(loc='best')
        plt.grid('--',alpha=0.3)
        plt.xlabel('Index')
        plt.ylabel('Voltage [V]')
        
def display_rise_time_analysis_of_plicodata20240906_1231():
    
    fname_plico = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\slm_time_response\\photodiode\\20240906_1232_1\\Analog - 9-6-2024 12-32-43.52892 PM.csv"
    rta = ResponseTimeAnalyzer(fname_plico)
    
    c_index = np.arange(0, 119800)
    b_index = np.arange(205000, len(rta._v2))
    
    vc = rta._v2[c_index].mean()
    err_vc = rta._v2[c_index].std()
    
    vb = rta._v2[b_index].mean()
    err_vb = rta._v2[b_index].std()
    
    idx0 = np.arange(116649, 121943)
    idx1 = np.arange(125515, 130920)
    idx2 = np.arange(134718, 139649)
    idx3 = np.arange(143503, 148501)
    idx4 = np.arange(152611, 158248)
    idx5 = np.arange(162446, 167848)
    idx6 = np.arange(171635, 176770)
    idx7 = np.arange(180500, 185983)
    idx8 = np.arange(190043, 195429)
    idx9 = np.arange(199133, 204419)
    
    va0 = rta._v1[idx0].mean()
    err_va0 = rta._v1[idx0].std()
    va1 = rta._v1[idx1].mean()
    err_va1 = rta._v1[idx1].std()
    va2 = rta._v1[idx2].mean()
    err_va2 = rta._v1[idx2].std()
    va3 = rta._v1[idx3].mean()
    err_va3 = rta._v1[idx3].std()
    va4 = rta._v1[idx4].mean()
    err_va4 = rta._v1[idx4].std()
    va5 = rta._v1[idx5].mean()
    err_va5 = rta._v1[idx5].std()
    va6 = rta._v1[idx6].mean()
    err_va6 = rta._v1[idx6].std()
    va7 = rta._v1[idx7].mean()
    err_va7 = rta._v1[idx7].std()
    va8 = rta._v1[idx8].mean()
    err_va8= rta._v1[idx8].std()
    va9 = rta._v1[idx9].mean()
    err_va9 = rta._v1[idx9].std()
    
    rta.display_data()
    import matplotlib.pyplot as plt
    plt.title("Plico data 240906_1231")
    
    plt.hlines(vc, 0 , rta._t.max(), ls='-',colors='black')
    plt.hlines(vc+err_vc, 0 , rta._t.max(), ls='--',colors='black')
    plt.hlines(vc-err_vc, 0 , rta._t.max(), ls='--',colors='black')
    
    plt.hlines(vb, 0 , rta._t.max(), ls='-',colors='black')
    plt.hlines(vb+err_vb, 0 , rta._t.max(), ls='--',colors='black')
    plt.hlines(vb-err_vb, 0 , rta._t.max(), ls='--',colors='black')
    
    dindx  = 500
    
    plt.hlines(va0, rta._t[idx0.min()- dindx], rta._t[idx0.max()+dindx], ls='-',colors='cyan')
    plt.hlines(va0+err_va0, rta._t[idx0.min()- dindx], rta._t[idx0.max()+dindx], ls='--',colors='cyan')
    plt.hlines(va0-err_va0, rta._t[idx0.min()- dindx], rta._t[idx0.max()+dindx], ls='--',colors='cyan')
    
    plt.hlines(va1, rta._t[idx1.min()- dindx], rta._t[idx1.max()+dindx], ls='-',colors='cyan')
    plt.hlines(va1+err_va1, rta._t[idx1.min()- dindx], rta._t[idx1.max()+dindx], ls='--',colors='cyan')
    plt.hlines(va1-err_va1, rta._t[idx1.min()- dindx], rta._t[idx1.max()+dindx], ls='--',colors='cyan')
    
    plt.hlines(va2, rta._t[idx2.min()- dindx], rta._t[idx2.max()+dindx], ls='-',colors='cyan')
    plt.hlines(va2+err_va2, rta._t[idx2.min()- dindx], rta._t[idx2.max()+dindx], ls='--',colors='cyan')
    plt.hlines(va2-err_va2, rta._t[idx2.min()- dindx], rta._t[idx2.max()+dindx], ls='--',colors='cyan')
    
    plt.hlines(va3, rta._t[idx3.min()- dindx], rta._t[idx3.max()+dindx], ls='-',colors='cyan')
    plt.hlines(va3+err_va3, rta._t[idx3.min()- dindx], rta._t[idx3.max()+dindx], ls='--',colors='cyan')
    plt.hlines(va3-err_va3, rta._t[idx3.min()- dindx], rta._t[idx3.max()+dindx], ls='--',colors='cyan')
    
    plt.hlines(va4, rta._t[idx4.min()- dindx], rta._t[idx4.max()+dindx], ls='-',colors='cyan')
    plt.hlines(va4+err_va4, rta._t[idx4.min()- dindx], rta._t[idx4.max()+dindx], ls='--',colors='cyan')
    plt.hlines(va4-err_va4, rta._t[idx4.min()- dindx], rta._t[idx4.max()+dindx], ls='--',colors='cyan')
    
    plt.hlines(va5, rta._t[idx5.min()- dindx], rta._t[idx5.max()+dindx], ls='-',colors='cyan')
    plt.hlines(va5+err_va5, rta._t[idx5.min()- dindx], rta._t[idx5.max()+dindx], ls='--',colors='cyan')
    plt.hlines(va5-err_va5, rta._t[idx5.min()- dindx], rta._t[idx5.max()+dindx], ls='--',colors='cyan')
    
    plt.hlines(va6, rta._t[idx6.min()- dindx], rta._t[idx6.max()+dindx], ls='-',colors='cyan')
    plt.hlines(va6+err_va6, rta._t[idx6.min()- dindx], rta._t[idx6.max()+dindx], ls='--',colors='cyan')
    plt.hlines(va6-err_va6, rta._t[idx6.min()- dindx], rta._t[idx6.max()+dindx], ls='--',colors='cyan')
    
    plt.hlines(va7, rta._t[idx7.min()- dindx], rta._t[idx7.max()+dindx], ls='-',colors='cyan')
    plt.hlines(va7+err_va7, rta._t[idx7.min()- dindx], rta._t[idx7.max()+dindx], ls='--',colors='cyan')
    plt.hlines(va7-err_va7, rta._t[idx7.min()- dindx], rta._t[idx7.max()+dindx], ls='--',colors='cyan')
    
    plt.hlines(va8, rta._t[idx8.min()- dindx], rta._t[idx8.max()+dindx], ls='-',colors='cyan')
    plt.hlines(va8+err_va8, rta._t[idx8.min()- dindx], rta._t[idx8.max()+dindx], ls='--',colors='cyan')
    plt.hlines(va8-err_va8, rta._t[idx8.min()- dindx], rta._t[idx8.max()+dindx], ls='--',colors='cyan')
    
    plt.hlines(va9, rta._t[idx9.min()- dindx], rta._t[idx9.max()+dindx], ls='-',colors='cyan')
    plt.hlines(va9+err_va9, rta._t[idx9.min()- dindx], rta._t[idx9.max()+dindx], ls='--',colors='cyan')
    plt.hlines(va9-err_va9, rta._t[idx9.min()- dindx], rta._t[idx9.max()+dindx], ls='--',colors='cyan')
    
    
    window_size = 8
    
    moving_average_v1 = []
    moving_average_v2 = []
    
    v1_roi = rta._v1[idx0.min():idx1.max()].copy()
    v2_roi = rta._v2[idx0.min():idx1.max()].copy()
    
    i=0
    while(i<len(v1_roi) - window_size + 1):
        
        window_average_v1 = np.sum(v1_roi[i : i + window_size])/window_size
        window_average_v2 = np.sum(v2_roi[i : i + window_size])/window_size
        moving_average_v1.append(window_average_v1)
        moving_average_v2.append(window_average_v2)
    
    v1_mean = np.array(moving_average_v1)
    v2_mean = np.array(moving_average_v2)

    plt.figure()
    plt.clf()
    plt.plot(v1_mean, 'b-', label='PD1')
    plt.plot(v2_mean, 'r-', label='PD2')
    plt.legend(loc='best')
    plt.grid('--',alpha=0.3)
    plt.xlabel('Index')
    plt.ylabel('Voltage [V]')
        
def display_rise_time_analysis_of_data20240906_1523():
    
    fname_blink_data = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\slm_time_response\\photodiode\\20240906_1523_blink_loop1_dwell100ms_pistons\\Analog - 9-6-2024 3-23-56.02487 PM.csv"
    rta = ResponseTimeAnalyzer(fname_blink_data)
    
    
    c_index = np.arange(0,53251)
    b_index = np.arange(88713, len(rta._v2))
    
    vc = rta._v2[c_index].mean()
    err_vc = rta._v2[c_index].std()
    
    vb = rta._v2[b_index].mean()
    err_vb = rta._v2[b_index].std()
    
    idx0 = np.arange(0, 54178)
    idx1 = np.arange(58757, 62879)
    idx2 = np.arange(67281, 71391)
    idx3 = np.arange(75853, 79828)
    idx4 = np.arange(84372, 88380)
    
    va0 = rta._v1[idx0].mean()
    err_va0 = rta._v1[idx0].std()
    va1 = rta._v1[idx1].mean()
    err_va1 = rta._v1[idx1].std()
    va2 = rta._v1[idx2].mean()
    err_va2 = rta._v1[idx2].std()
    va3 = rta._v1[idx3].mean()
    err_va3 = rta._v1[idx3].std()
    va4 = rta._v1[idx4].mean()
    err_va4 = rta._v1[idx4].std()
    
    import matplotlib.pyplot as plt
    rta.display_data()
    plt.title("Blink data 240906_1523 Dwell Time 100ms pistons")
    
    plt.hlines(vc, 0 , rta._t.max(), ls='-',colors='black')
    plt.hlines(vc+err_vc, 0 , rta._t.max(), ls='--',colors='black')
    plt.hlines(vc-err_vc, 0 , rta._t.max(), ls='--',colors='black')
    
    plt.hlines(vb, 0 , rta._t.max(), ls='-',colors='black')
    plt.hlines(vb+err_vb, 0 , rta._t.max(), ls='--',colors='black')
    plt.hlines(vb-err_vb, 0 , rta._t.max(), ls='--',colors='black')
    
    dindx  = 500
    
    plt.hlines(va0, 0, rta._t[idx0.max()+dindx], ls='-',colors='cyan')
    plt.hlines(va0+err_va0, 0, rta._t[idx0.max()+dindx], ls='--',colors='cyan')
    plt.hlines(va0-err_va0, 0, rta._t[idx0.max()+dindx], ls='--',colors='cyan')
    
    plt.hlines(va1, rta._t[idx1.min()- dindx], rta._t[idx1.max()+dindx], ls='-',colors='cyan')
    plt.hlines(va1+err_va1, rta._t[idx1.min()- dindx], rta._t[idx1.max()+dindx], ls='--',colors='cyan')
    plt.hlines(va1-err_va1, rta._t[idx1.min()- dindx], rta._t[idx1.max()+dindx], ls='--',colors='cyan')
    
    plt.hlines(va2, rta._t[idx2.min()- dindx], rta._t[idx2.max()+dindx], ls='-',colors='cyan')
    plt.hlines(va2+err_va2, rta._t[idx2.min()- dindx], rta._t[idx2.max()+dindx], ls='--',colors='cyan')
    plt.hlines(va2-err_va2, rta._t[idx2.min()- dindx], rta._t[idx2.max()+dindx], ls='--',colors='cyan')
    
    plt.hlines(va3, rta._t[idx3.min()- dindx], rta._t[idx3.max()+dindx], ls='-',colors='cyan')
    plt.hlines(va3+err_va3, rta._t[idx3.min()- dindx], rta._t[idx3.max()+dindx], ls='--',colors='cyan')
    plt.hlines(va3-err_va3, rta._t[idx3.min()- dindx], rta._t[idx3.max()+dindx], ls='--',colors='cyan')
    
    plt.hlines(va4, rta._t[idx4.min()- dindx], rta._t[idx4.max()+dindx], ls='-',colors='cyan')
    plt.hlines(va4+err_va4, rta._t[idx4.min()- dindx], rta._t[idx4.max()+dindx], ls='--',colors='cyan')
    plt.hlines(va4-err_va4, rta._t[idx4.min()- dindx], rta._t[idx4.max()+dindx], ls='--',colors='cyan')
    
        
def display_rise_time_analysis_of_data20240906_1422():
    
    fname_blink_data  = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\slm_time_response\\photodiode\\20240906_1422_blink_loop10_dwell0ms\\Analog - 9-6-2024 2-22-21.39271 PM.csv"
    rta = ResponseTimeAnalyzer(fname_blink_data)
    
    c_index = np.arange(0, 49355)
    b_index = np.arange(54665, len(rta._v2))
    a_index = np.arange(0, 49402)
    
    va = rta._v1[a_index].mean()
    err_va = rta._v1[a_index].std()
    
    vb = rta._v2[b_index].mean()
    err_vb = rta._v2[b_index].std()
    
    vc = rta._v2[c_index].mean()
    err_vc = rta._v2[c_index].std()
    
    rta.display_data()
    import matplotlib.pyplot as plt
    plt.title("Blink data 240906_1422 Dwell Time 0ms")
    
    plt.hlines(va, 0, rta._t.max(), ls='-', colors='cyan')
    plt.hlines(va+err_va, 0, rta._t.max(), ls='--', colors='cyan')
    plt.hlines(va-err_va, 0, rta._t.max(), ls='--', colors='cyan')
    
    plt.hlines(vb, 0, rta._t.max(), ls='-', colors='black')
    plt.hlines(vb+err_vb, 0, rta._t.max(), ls='--', colors='black')
    plt.hlines(vb-err_vb, 0, rta._t.max(), ls='--', colors='black')
    
    plt.hlines(vc, 0, rta._t.max(), ls='-', colors='black')
    plt.hlines(vc+err_vc, 0, rta._t.max(), ls='--', colors='black')
    plt.hlines(vc-err_vc, 0, rta._t.max(), ls='--', colors='black')
        

def display_rise_time_analysis_of_data20240906_1441():
    
    fname_blink_data  = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\slm_time_response\\photodiode\\20240906_1441_blink_loop10_dwell100ms\\Analog - 9-6-2024 2-41-15.63336 PM.csv"
    rta = ResponseTimeAnalyzer(fname_blink_data)
    
    c_index = np.arange(0,44061)
    b_index = np.arange(121585,len(rta._v2))
    
    vc = rta._v2[c_index].mean()
    err_vc = rta._v2[c_index].std()
    
    vb = rta._v2[b_index].mean()
    err_vb = rta._v2[b_index].std()
    
    idx0 = np.arange(39930,44056)
    idx1 = np.arange(48562,52681)
    idx2 = np.arange(57061,61175)
    idx3 = np.arange(65581,69722)
    idx4 = np.arange(74143,78200)
    idx5 = np.arange(82659,86473)
    idx6 = np.arange(91207,95267)
    idx7 = np.arange(99733,103790)
    idx8 = np.arange(108217,112309)
    idx9 = np.arange(116799,120853)
    
    va0 = rta._v1[idx0].mean()
    err_va0 = rta._v1[idx0].std()
    va1 = rta._v1[idx1].mean()
    err_va1 = rta._v1[idx1].std()
    va2 = rta._v1[idx2].mean()
    err_va2 = rta._v1[idx2].std()
    va3 = rta._v1[idx3].mean()
    err_va3 = rta._v1[idx3].std()
    va4 = rta._v1[idx4].mean()
    err_va4 = rta._v1[idx4].std()
    va5 = rta._v1[idx5].mean()
    err_va5 = rta._v1[idx5].std()
    va6 = rta._v1[idx6].mean()
    err_va6 = rta._v1[idx6].std()
    va7 = rta._v1[idx7].mean()
    err_va7 = rta._v1[idx7].std()
    va8 = rta._v1[idx8].mean()
    err_va8= rta._v1[idx8].std()
    va9 = rta._v1[idx9].mean()
    err_va9 = rta._v1[idx9].std()
    
    rta.display_data()
    
    import matplotlib.pyplot as plt
    
    plt.title("Blink data 240906_1441 Dwell Time 100ms")
    
    plt.hlines(vc, 0 , rta._t.max(), ls='-',colors='black')
    plt.hlines(vc+err_vc, 0 , rta._t.max(), ls='--',colors='black')
    plt.hlines(vc-err_vc, 0 , rta._t.max(), ls='--',colors='black')
    
    plt.hlines(vb, 0 , rta._t.max(), ls='-',colors='black')
    plt.hlines(vb+err_vb, 0 , rta._t.max(), ls='--',colors='black')
    plt.hlines(vb-err_vb, 0 , rta._t.max(), ls='--',colors='black')
    
    dindx  = 500
    
    plt.hlines(va0, rta._t[idx0.min()- dindx], rta._t[idx0.max()+dindx], ls='-',colors='cyan')
    plt.hlines(va0+err_va0, rta._t[idx0.min()- dindx], rta._t[idx0.max()+dindx], ls='--',colors='cyan')
    plt.hlines(va0-err_va0, rta._t[idx0.min()- dindx], rta._t[idx0.max()+dindx], ls='--',colors='cyan')
    
    plt.hlines(va1, rta._t[idx1.min()- dindx], rta._t[idx1.max()+dindx], ls='-',colors='cyan')
    plt.hlines(va1+err_va1, rta._t[idx1.min()- dindx], rta._t[idx1.max()+dindx], ls='--',colors='cyan')
    plt.hlines(va1-err_va1, rta._t[idx1.min()- dindx], rta._t[idx1.max()+dindx], ls='--',colors='cyan')
    
    plt.hlines(va2, rta._t[idx2.min()- dindx], rta._t[idx2.max()+dindx], ls='-',colors='cyan')
    plt.hlines(va2+err_va2, rta._t[idx2.min()- dindx], rta._t[idx2.max()+dindx], ls='--',colors='cyan')
    plt.hlines(va2-err_va2, rta._t[idx2.min()- dindx], rta._t[idx2.max()+dindx], ls='--',colors='cyan')
    
    plt.hlines(va3, rta._t[idx3.min()- dindx], rta._t[idx3.max()+dindx], ls='-',colors='cyan')
    plt.hlines(va3+err_va3, rta._t[idx3.min()- dindx], rta._t[idx3.max()+dindx], ls='--',colors='cyan')
    plt.hlines(va3-err_va3, rta._t[idx3.min()- dindx], rta._t[idx3.max()+dindx], ls='--',colors='cyan')
    
    plt.hlines(va4, rta._t[idx4.min()- dindx], rta._t[idx4.max()+dindx], ls='-',colors='cyan')
    plt.hlines(va4+err_va4, rta._t[idx4.min()- dindx], rta._t[idx4.max()+dindx], ls='--',colors='cyan')
    plt.hlines(va4-err_va4, rta._t[idx4.min()- dindx], rta._t[idx4.max()+dindx], ls='--',colors='cyan')
    
    plt.hlines(va5, rta._t[idx5.min()- dindx], rta._t[idx5.max()+dindx], ls='-',colors='cyan')
    plt.hlines(va5+err_va5, rta._t[idx5.min()- dindx], rta._t[idx5.max()+dindx], ls='--',colors='cyan')
    plt.hlines(va5-err_va5, rta._t[idx5.min()- dindx], rta._t[idx5.max()+dindx], ls='--',colors='cyan')
    
    plt.hlines(va6, rta._t[idx6.min()- dindx], rta._t[idx6.max()+dindx], ls='-',colors='cyan')
    plt.hlines(va6+err_va6, rta._t[idx6.min()- dindx], rta._t[idx6.max()+dindx], ls='--',colors='cyan')
    plt.hlines(va6-err_va6, rta._t[idx6.min()- dindx], rta._t[idx6.max()+dindx], ls='--',colors='cyan')
    
    plt.hlines(va7, rta._t[idx7.min()- dindx], rta._t[idx7.max()+dindx], ls='-',colors='cyan')
    plt.hlines(va7+err_va7, rta._t[idx7.min()- dindx], rta._t[idx7.max()+dindx], ls='--',colors='cyan')
    plt.hlines(va7-err_va7, rta._t[idx7.min()- dindx], rta._t[idx7.max()+dindx], ls='--',colors='cyan')
    
    plt.hlines(va8, rta._t[idx8.min()- dindx], rta._t[idx8.max()+dindx], ls='-',colors='cyan')
    plt.hlines(va8+err_va8, rta._t[idx8.min()- dindx], rta._t[idx8.max()+dindx], ls='--',colors='cyan')
    plt.hlines(va8-err_va8, rta._t[idx8.min()- dindx], rta._t[idx8.max()+dindx], ls='--',colors='cyan')
    
    plt.hlines(va9, rta._t[idx9.min()- dindx], rta._t[idx9.max()+dindx], ls='-',colors='cyan')
    plt.hlines(va9+err_va9, rta._t[idx9.min()- dindx], rta._t[idx9.max()+dindx], ls='--',colors='cyan')
    plt.hlines(va9-err_va9, rta._t[idx9.min()- dindx], rta._t[idx9.max()+dindx], ls='--',colors='cyan')
    
    
    
    