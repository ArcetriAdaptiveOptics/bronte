import numpy as np
import matplotlib.pyplot as plt
from bronte.mains.slm_time_response import ResponseTimeAnalyzer

class PhotodiodeDataAnalyzer():


    FDIR = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\slm_time_response\\photodiode\\"
    
    def __init__(self):
        
        self._fname_bkg = self.FDIR + "20240906_1442\\Analog - 9-6-2024 2-43-04.62599 PM.csv"
        
        self._fname_fix_on_pd1 = self.FDIR  + "20240906_1445\\Analog - 9-6-2024 2-45-53.31368 PM.csv"
        self._fname_fix_on_pd2 = self.FDIR  + "20240906_1447\\Analog - 9-6-2024 2-47-42.39440 PM.csv"
        
        self._fname_plico = self.FDIR + "20240906_1232_1\\Analog - 9-6-2024 12-32-43.52892 PM.csv"
        
        self._fname_blink_dwt0ms  = self.FDIR + "20240906_1422_blink_loop10_dwell0ms\\Analog - 9-6-2024 2-22-21.39271 PM.csv"
        self._fname_blink_dwt100ms = self.FDIR + "20240906_1441_blink_loop10_dwell100ms\\Analog - 9-6-2024 2-41-15.63336 PM.csv"
        
        self._fname_blink_pistons = self.FDIR + "20240906_1523_blink_loop1_dwell100ms_pistons\\Analog - 9-6-2024 3-23-56.02487 PM.csv"
        
    def fix_command_on_pds(self):
        
        rta_pd1 = ResponseTimeAnalyzer(self._fname_fix_on_pd1, self._fname_bkg)
        rta_pd2 = ResponseTimeAnalyzer(self._fname_fix_on_pd2, self._fname_bkg)

        rta_pd1.display_moving_average_signals()
        rta_pd2.display_moving_average_signals()

        v1_mean = rta_pd1._v1_sma.mean()
        err_v1 = rta_pd1._v1_sma.std()
        
        v2_mean = rta_pd2._v2_sma.mean()
        err_v2 = rta_pd2._v2_sma.std()
        
        freq_pd1, fft_v1 = self._get_fft(rta_pd1._v1_sma, 4)
        
        plt.figure()
        plt.clf()
        plt.plot(freq_pd1, np.abs(fft_v1))
        plt.xlim(0, 10)
        
    def _get_fft(self, signal, Npad = 4):
        
        padded_signal = np.zeros((len(signal)*Npad))
        padded_signal[:len(signal)] = signal
        #the sampling rate is 40000Hz but SMA window is 8 samples
        
        sr = 40000
        ts = 1/sr
        N = len(padded_signal)
        T = N/sr
        freq = np.arange(N)/T
        
        freq =  np.fft.fftfreq(len(padded_signal), 1/T)
        fft_signal = np.fft.fft(padded_signal)
        
        return freq, fft_signal

    def plico_data(self):
        
        rta_plico = ResponseTimeAnalyzer(self._fname_plico, self._fname_bkg)
        display_rise_time_analysis_of_plicodata20240906_1232(rta_plico)

    def blink_data_dwell_time0ms(self):
        
        rta_blink = ResponseTimeAnalyzer(self._fname_blink_dwt0ms, self._fname_bkg)
        display_rise_time_analysis_of_data20240906_1422(rta_blink)
        
    def blink_data_dwell_time100ms_pistons(self):
        
        rta_blink_pistons = ResponseTimeAnalyzer(self._fname_blink_pistons, self._fname_bkg)
        display_rise_time_analysis_of_data20240906_1523(rta_blink_pistons)
        
    




def display_rise_time_analysis_of_plicodata20240906_1232(rta):
    
    # fname_plico = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\slm_time_response\\photodiode\\20240906_1232_1\\Analog - 9-6-2024 12-32-43.52892 PM.csv"
    # rta = ResponseTimeAnalyzer(fname_plico)
    
    c_index = np.arange(0, 119800)
    b_index = np.arange(205000, len(rta._v2_sma))
    
    vc = rta._v2_sma[c_index].mean()
    err_vc = 0.1 * vc
    
    vb = rta._v2_sma[b_index].mean()
    err_vb = 0.1 * vb
    
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
    
    va0 = rta._v1_sma[idx0].mean()
    err_va0 = 0.1 * va0
    va1 = rta._v1_sma[idx1].mean()
    err_va1 = 0.1 * va1
    va2 = rta._v1_sma[idx2].mean()
    err_va2 = 0.1 * va2
    va3 = rta._v1_sma[idx3].mean()
    err_va3 = 0.1 * va3
    va4 = rta._v1_sma[idx4].mean()
    err_va4 = 0.1 * va4
    va5 = rta._v1_sma[idx5].mean()
    err_va5 = 0.1 * va5
    va6 = rta._v1_sma[idx6].mean()
    err_va6 = 0.1 * va6
    va7 = rta._v1_sma[idx7].mean()
    err_va7 = 0.1 * va7
    va8 = rta._v1_sma[idx8].mean()
    err_va8= 0.1 * va8
    va9 = rta._v1_sma[idx9].mean()
    err_va9 = 0.1 * va9
    
    rta.display_moving_average_signals()
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
    
    
        
def display_rise_time_analysis_of_data20240906_1523(rta):
    
    # fname_blink_data = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\slm_time_response\\photodiode\\20240906_1523_blink_loop1_dwell100ms_pistons\\Analog - 9-6-2024 3-23-56.02487 PM.csv"
    # rta = ResponseTimeAnalyzer(fname_blink_data)
    
    
    c_index = np.arange(0,53251)
    b_index = np.arange(88713, len(rta._v2_sma))
    
    vc = rta._v2_sma[c_index].mean()
    err_vc = 0.1 * vc
    
    vb = rta._v2_sma[b_index].mean()
    err_vb = 0.1 * vb
    
    idx0 = np.arange(0, 54178)
    idx1 = np.arange(58757, 62879)
    idx2 = np.arange(67281, 71391)
    idx3 = np.arange(75853, 79828)
    idx4 = np.arange(84372, 88380)
    
    va0 = rta._v1_sma[idx0].mean()
    err_va0 = 0.1* va0
    va1 = rta._v1_sma[idx1].mean()
    err_va1 = 0.1* va1
    va2 = rta._v1_sma[idx2].mean()
    err_va2 = 0.1* va2
    va3 = rta._v1_sma[idx3].mean()
    err_va3 = 0.1* va3
    va4 = rta._v1_sma[idx4].mean()
    err_va4 = 0.1* va4
    
    
    rta.display_moving_average_signals()
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
    
        
def display_rise_time_analysis_of_data20240906_1422(rta):
    
    c_index = np.arange(0, 49355)
    b_index = np.arange(54665, len(rta._v2))
    a_index = np.arange(0, 49402)
    
    va = rta._v1_sma[a_index].mean()
    err_va = 0.1 * va
    
    vb = rta._v2_sma[b_index].mean()
    err_vb = 0.1 * vb
    
    vc = rta._v2_sma[c_index].mean()
    err_vc = 0.1 * vc
    
    rta.display_moving_average_signals()
    
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
        

# def display_rise_time_analysis_of_data20240906_1441():
#
#     fname_blink_data  = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\slm_time_response\\photodiode\\20240906_1441_blink_loop10_dwell100ms\\Analog - 9-6-2024 2-41-15.63336 PM.csv"
#     rta = ResponseTimeAnalyzer(fname_blink_data)
#
#     c_index = np.arange(0,44061)
#     b_index = np.arange(121585,len(rta._v2))
#
#     vc = rta._v2[c_index].mean()
#     err_vc = rta._v2[c_index].std()
#
#     vb = rta._v2[b_index].mean()
#     err_vb = rta._v2[b_index].std()
#
#     idx0 = np.arange(39930,44056)
#     idx1 = np.arange(48562,52681)
#     idx2 = np.arange(57061,61175)
#     idx3 = np.arange(65581,69722)
#     idx4 = np.arange(74143,78200)
#     idx5 = np.arange(82659,86473)
#     idx6 = np.arange(91207,95267)
#     idx7 = np.arange(99733,103790)
#     idx8 = np.arange(108217,112309)
#     idx9 = np.arange(116799,120853)
#
#     va0 = rta._v1[idx0].mean()
#     err_va0 = rta._v1[idx0].std()
#     va1 = rta._v1[idx1].mean()
#     err_va1 = rta._v1[idx1].std()
#     va2 = rta._v1[idx2].mean()
#     err_va2 = rta._v1[idx2].std()
#     va3 = rta._v1[idx3].mean()
#     err_va3 = rta._v1[idx3].std()
#     va4 = rta._v1[idx4].mean()
#     err_va4 = rta._v1[idx4].std()
#     va5 = rta._v1[idx5].mean()
#     err_va5 = rta._v1[idx5].std()
#     va6 = rta._v1[idx6].mean()
#     err_va6 = rta._v1[idx6].std()
#     va7 = rta._v1[idx7].mean()
#     err_va7 = rta._v1[idx7].std()
#     va8 = rta._v1[idx8].mean()
#     err_va8= rta._v1[idx8].std()
#     va9 = rta._v1[idx9].mean()
#     err_va9 = rta._v1[idx9].std()
#
#     rta.display_data()
#
#
#     plt.title("Blink data 240906_1441 Dwell Time 100ms")
#
#     plt.hlines(vc, 0 , rta._t.max(), ls='-',colors='black')
#     plt.hlines(vc+err_vc, 0 , rta._t.max(), ls='--',colors='black')
#     plt.hlines(vc-err_vc, 0 , rta._t.max(), ls='--',colors='black')
#
#     plt.hlines(vb, 0 , rta._t.max(), ls='-',colors='black')
#     plt.hlines(vb+err_vb, 0 , rta._t.max(), ls='--',colors='black')
#     plt.hlines(vb-err_vb, 0 , rta._t.max(), ls='--',colors='black')
#
#     dindx  = 500
#
#     plt.hlines(va0, rta._t[idx0.min()- dindx], rta._t[idx0.max()+dindx], ls='-',colors='cyan')
#     plt.hlines(va0+err_va0, rta._t[idx0.min()- dindx], rta._t[idx0.max()+dindx], ls='--',colors='cyan')
#     plt.hlines(va0-err_va0, rta._t[idx0.min()- dindx], rta._t[idx0.max()+dindx], ls='--',colors='cyan')
#
#     plt.hlines(va1, rta._t[idx1.min()- dindx], rta._t[idx1.max()+dindx], ls='-',colors='cyan')
#     plt.hlines(va1+err_va1, rta._t[idx1.min()- dindx], rta._t[idx1.max()+dindx], ls='--',colors='cyan')
#     plt.hlines(va1-err_va1, rta._t[idx1.min()- dindx], rta._t[idx1.max()+dindx], ls='--',colors='cyan')
#
#     plt.hlines(va2, rta._t[idx2.min()- dindx], rta._t[idx2.max()+dindx], ls='-',colors='cyan')
#     plt.hlines(va2+err_va2, rta._t[idx2.min()- dindx], rta._t[idx2.max()+dindx], ls='--',colors='cyan')
#     plt.hlines(va2-err_va2, rta._t[idx2.min()- dindx], rta._t[idx2.max()+dindx], ls='--',colors='cyan')
#
#     plt.hlines(va3, rta._t[idx3.min()- dindx], rta._t[idx3.max()+dindx], ls='-',colors='cyan')
#     plt.hlines(va3+err_va3, rta._t[idx3.min()- dindx], rta._t[idx3.max()+dindx], ls='--',colors='cyan')
#     plt.hlines(va3-err_va3, rta._t[idx3.min()- dindx], rta._t[idx3.max()+dindx], ls='--',colors='cyan')
#
#     plt.hlines(va4, rta._t[idx4.min()- dindx], rta._t[idx4.max()+dindx], ls='-',colors='cyan')
#     plt.hlines(va4+err_va4, rta._t[idx4.min()- dindx], rta._t[idx4.max()+dindx], ls='--',colors='cyan')
#     plt.hlines(va4-err_va4, rta._t[idx4.min()- dindx], rta._t[idx4.max()+dindx], ls='--',colors='cyan')
#
#     plt.hlines(va5, rta._t[idx5.min()- dindx], rta._t[idx5.max()+dindx], ls='-',colors='cyan')
#     plt.hlines(va5+err_va5, rta._t[idx5.min()- dindx], rta._t[idx5.max()+dindx], ls='--',colors='cyan')
#     plt.hlines(va5-err_va5, rta._t[idx5.min()- dindx], rta._t[idx5.max()+dindx], ls='--',colors='cyan')
#
#     plt.hlines(va6, rta._t[idx6.min()- dindx], rta._t[idx6.max()+dindx], ls='-',colors='cyan')
#     plt.hlines(va6+err_va6, rta._t[idx6.min()- dindx], rta._t[idx6.max()+dindx], ls='--',colors='cyan')
#     plt.hlines(va6-err_va6, rta._t[idx6.min()- dindx], rta._t[idx6.max()+dindx], ls='--',colors='cyan')
#
#     plt.hlines(va7, rta._t[idx7.min()- dindx], rta._t[idx7.max()+dindx], ls='-',colors='cyan')
#     plt.hlines(va7+err_va7, rta._t[idx7.min()- dindx], rta._t[idx7.max()+dindx], ls='--',colors='cyan')
#     plt.hlines(va7-err_va7, rta._t[idx7.min()- dindx], rta._t[idx7.max()+dindx], ls='--',colors='cyan')
#
#     plt.hlines(va8, rta._t[idx8.min()- dindx], rta._t[idx8.max()+dindx], ls='-',colors='cyan')
#     plt.hlines(va8+err_va8, rta._t[idx8.min()- dindx], rta._t[idx8.max()+dindx], ls='--',colors='cyan')
#     plt.hlines(va8-err_va8, rta._t[idx8.min()- dindx], rta._t[idx8.max()+dindx], ls='--',colors='cyan')
#
#     plt.hlines(va9, rta._t[idx9.min()- dindx], rta._t[idx9.max()+dindx], ls='-',colors='cyan')
#     plt.hlines(va9+err_va9, rta._t[idx9.min()- dindx], rta._t[idx9.max()+dindx], ls='--',colors='cyan')
#     plt.hlines(va9-err_va9, rta._t[idx9.min()- dindx], rta._t[idx9.max()+dindx], ls='--',colors='cyan')
