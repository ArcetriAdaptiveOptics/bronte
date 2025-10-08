import numpy as np
import matplotlib.pyplot as plt
from bronte.mains.slm_time_response import ResponseTimeAnalyzer

class PhotodiodeDataAnalyzer():


    FDIR = "D:\\phd_slm_edo\\old_data\\slm_time_response\\photodiode\\"#"C:\\Users\\labot\\Desktop\\misure_tesi_slm\\slm_time_response\\photodiode\\"
    
    def __init__(self):
        
        self._fname_bkg = self.FDIR + "20240906_1442\\Analog - 9-6-2024 2-43-04.62599 PM.csv"
        
        self._fname_fix_on_pd1 = self.FDIR  + "20240906_1445\\Analog - 9-6-2024 2-45-53.31368 PM.csv"
        self._fname_fix_on_pd2 = self.FDIR  + "20240906_1447\\Analog - 9-6-2024 2-47-42.39440 PM.csv"
        
        self._fname_plico = self.FDIR + "20240906_1232_1\\Analog - 9-6-2024 12-32-43.52892 PM.csv"
        
        self._fname_blink_dwt0ms  = self.FDIR + "20240906_1422_blink_loop10_dwell0ms\\Analog - 9-6-2024 2-22-21.39271 PM.csv"
        self._fname_blink_dwt100ms = self.FDIR + "20240906_1441_blink_loop10_dwell100ms\\Analog - 9-6-2024 2-41-15.63336 PM.csv"
        
        self._fname_blink_pistons = self.FDIR + "20240906_1523_blink_loop1_dwell100ms_pistons\\Analog - 9-6-2024 3-23-56.02487 PM.csv"
        
        self._fname_pd2_on_axis = self.FDIR + "20240912_1622\\Analog - 9-12-2024 4-22-40.67322 PM.csv"
        
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
        
        rta_plico = ResponseTimeAnalyzer(self._fname_plico)
        display_rise_time_analysis_of_plicodata20240906_1232(rta_plico)
        plt.xlim(3.7-3.7,3.73-3.7)

    def blink_data_dwell_time0ms(self):
        
        rta_blink = ResponseTimeAnalyzer(self._fname_blink_dwt0ms)
        return display_rise_time_analysis_of_data20240906_1422(rta_blink)
    
    def plico_data_pd2_on_axis(self):
        rta_plico_pd2_on_axis = ResponseTimeAnalyzer(self._fname_pd2_on_axis)
        return display_rise_time_analysis_of_plicodata20240912_1622(rta_plico_pd2_on_axis)
        
    def blink_data_dwell_time100ms_pistons(self):
        
        rta_blink_pistons = ResponseTimeAnalyzer(self._fname_blink_pistons, self._fname_bkg)
        display_rise_time_analysis_of_data20240906_1523(rta_blink_pistons)
        
    

def display_rise_time_analysis_of_plicodata20240912_1622(rta):
    
    d_index = np.arange(0, 100200)
    vd = rta._v1_sma[d_index].mean()
    abs_err_vd = np.abs(rta._v1_sma[d_index] - rta._v1[d_index])
    mae_vd = np.mean(abs_err_vd)
    err_vd = 3 * rta._v1_sma[d_index].std()#3 * mae_vd
    
    c_index = np.arange(100750, 104043)
    vc = rta._v2_sma[c_index].mean()
    abs_err_vc = np.abs(rta._v2_sma[c_index] - rta._v2[c_index])
    mae_vc = np.mean(abs_err_vc)
    err_vc = 3 * rta._v2_sma[c_index].std()#3 * mae_vc
    
    b_index = np.arange(100725, 104111)
    vb = rta._v1_sma[b_index].mean()
    abs_err_vb = np.abs(rta._v1_sma[b_index]-rta._v1[b_index])
    mae_vb = np.mean(abs_err_vb)
    err_vb = 3 * rta._v1_sma[b_index].std() #3 * mae_vb
    
    idx0 = np.arange(95683, 100170)
    va0 = rta._v2_sma[idx0].mean()
    abs_err_va0 = np.abs(rta._v2_sma[idx0] - rta._v2[idx0])
    mae_va0 = np.mean(abs_err_va0)
    err_va0 = 3 * rta._v2_sma[idx0].std()#3 * mae_va0
    
    idx1 = np.arange(104483, 109085)
    va1 = rta._v2_sma[idx1].mean()
    abs_err_va1 = np.abs(rta._v2_sma[idx1] - rta._v2[idx1])
    mae_va1 = np.mean(abs_err_va1)
    err_va1 = 3 * rta._v2_sma[idx1].std()#3 * mae_va1
    
    idx2 = np.arange(112764, 117578)
    va2 = rta._v2_sma[idx2].mean()
    abs_err_va2 = np.abs(rta._v2_sma[idx2] - rta._v2[idx2])
    mae_va2 = np.mean(abs_err_va2)
    err_va2 = 3 * rta._v2_sma[idx2].std()#3 * mae_va2
    
    idx3 = np.arange(121486, 126226)
    va3 = rta._v2_sma[idx3].mean()
    abs_err_va3 = np.abs(rta._v2_sma[idx3] - rta._v2[idx3])
    mae_va3 = np.mean(abs_err_va3)
    err_va3 = 3 * rta._v2_sma[idx3].std() #3 * mae_va3
    
    idx4 = np.arange(129887, 134926)
    va4 = rta._v2_sma[idx4].mean()
    abs_err_va4 = np.abs(rta._v2_sma[idx4] - rta._v2[idx4])
    mae_va4 = np.mean(abs_err_va4)
    err_va4 = 3 * rta._v2_sma[idx4].std()#3 * mae_va4
    
    idx5 = np.arange(138550, 143404)
    va5 = rta._v2_sma[idx5].mean()
    abs_err_va5 = np.abs(rta._v2_sma[idx5] - rta._v2[idx5])
    mae_va5 = np.mean(abs_err_va5)
    err_va5 = 3 * rta._v2_sma[idx5].std() #3 * mae_va5
    
    idx6 = np.arange(147507,152332)
    va6 = rta._v2_sma[idx6].mean()
    abs_err_va6 = np.abs(rta._v2_sma[idx6] - rta._v2[idx6])
    mae_va6 = np.mean(abs_err_va6)
    err_va6 = 3 * rta._v2_sma[idx6].std()#3 * mae_va6
    
    idx7 = np.arange(156000,160600)
    va7 = rta._v2_sma[idx7].mean()
    abs_err_va7 = np.abs(rta._v2_sma[idx7] - rta._v2[idx7])
    mae_va7 = np.mean(abs_err_va7)
    err_va7 = 3 * rta._v2_sma[idx7].std()#3 * mae_va7
    
    idx8 = np.arange(164449,169053)
    va8 = rta._v2_sma[idx8].mean()
    abs_err_va8 = np.abs(rta._v2_sma[idx8] - rta._v2[idx8])
    mae_va8 = np.mean(abs_err_va8)
    err_va8 = 3 * rta._v2_sma[idx8].std() #3 * mae_va8
    
    idx9 = np.arange(173219,178258)
    va9 = rta._v2_sma[idx9].mean()
    abs_err_va9 = np.abs(rta._v2_sma[idx9] - rta._v2[idx9])
    mae_va9 = np.mean(abs_err_va9)
    err_va9 = 3 * rta._v2_sma[idx9].std()#3 * mae_va9
    
    #rta.display_data_indexing()
    rta.display_moving_average_signals()
    
    plt.hlines(vd, 0, rta._t.max(), ls = '-', colors ='cyan')
    plt.hlines(vd + err_vd, 0, rta._t.max(), ls = '--', colors ='cyan')
    
    plt.hlines(vc, 0, rta._t.max(), ls = '-', colors ='black')
    plt.hlines(vc + err_vc, 0, rta._t.max(), ls = '--', colors ='black')
    
    plt.hlines(vb, 0, rta._t.max(), ls = '-', colors ='cyan')
    plt.hlines(vb - err_vb, 0, rta._t.max(), ls = '--', colors ='cyan')
    
    dindx  = 500
    
    plt.hlines(va0, rta._t[idx0.min()- dindx], rta._t[idx0.max()+dindx], ls='-',colors='black')
    plt.hlines(va0-err_va0, rta._t[idx0.min()- dindx], rta._t[idx0.max()+dindx], ls='--',colors='black')
    
    plt.hlines(va1, rta._t[idx1.min()- dindx], rta._t[idx1.max()+dindx], ls='-',colors='black')
    plt.hlines(va1-err_va1, rta._t[idx1.min()- dindx], rta._t[idx1.max()+dindx], ls='--',colors='black')
    
    plt.hlines(va2, rta._t[idx2.min()- dindx], rta._t[idx2.max()+dindx], ls='-',colors='black')
    plt.hlines(va2-err_va2, rta._t[idx2.min()- dindx], rta._t[idx2.max()+dindx], ls='--',colors='black')
    
    plt.hlines(va3, rta._t[idx3.min()- dindx], rta._t[idx3.max()+dindx], ls='-',colors='black')
    plt.hlines(va3-err_va3, rta._t[idx3.min()- dindx], rta._t[idx3.max()+dindx], ls='--',colors='black')
    
    plt.hlines(va4, rta._t[idx4.min()- dindx], rta._t[idx4.max()+dindx], ls='-',colors='black')
    plt.hlines(va4-err_va4, rta._t[idx4.min()- dindx], rta._t[idx4.max()+dindx], ls='--',colors='black')
    
    plt.hlines(va5, rta._t[idx5.min()- dindx], rta._t[idx5.max()+dindx], ls='-',colors='black')
    plt.hlines(va5-err_va5, rta._t[idx5.min()- dindx], rta._t[idx5.max()+dindx], ls='--',colors='black')
    
    plt.hlines(va6, rta._t[idx6.min()- dindx], rta._t[idx6.max()+dindx], ls='-',colors='black')
    plt.hlines(va6-err_va6, rta._t[idx6.min()- dindx], rta._t[idx6.max()+dindx], ls='--',colors='black')
    
    plt.hlines(va7, rta._t[idx7.min()- dindx], rta._t[idx7.max()+dindx], ls='-',colors='black')
    plt.hlines(va7-err_va7, rta._t[idx7.min()- dindx], rta._t[idx7.max()+dindx], ls='--',colors='black')
    
    plt.hlines(va8, rta._t[idx8.min()- dindx], rta._t[idx8.max()+dindx], ls='-',colors='black')
    plt.hlines(va8-err_va8, rta._t[idx8.min()- dindx], rta._t[idx8.max()+dindx], ls='--',colors='black')
    
    plt.hlines(va9, rta._t[idx9.min()- dindx], rta._t[idx9.max()+dindx], ls='-',colors='black')
    plt.hlines(va9-err_va9, rta._t[idx9.min()- dindx], rta._t[idx9.max()+dindx], ls='--',colors='black')
    

    return (va0, err_va0, mae_va0), (vb, err_vb, mae_vb), (vc, err_vc, mae_vc), (vd, err_vd, mae_vd)

def display_rise_time_analysis_of_plicodata20240906_1232(rta):
    
    # fname_plico = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\slm_time_response\\photodiode\\20240906_1232_1\\Analog - 9-6-2024 12-32-43.52892 PM.csv"
    # rta = ResponseTimeAnalyzer(fname_plico)
    
    d_index = np.arange(0, 119800)
    vd = rta._v2_sma[d_index].mean()
    abs_err_vd = np.abs(rta._v2_sma[d_index]-rta._v2[d_index])
    mae_vd = np.mean(abs_err_vd)
    err_vd = 3 * rta._v2_sma[d_index].std() #3 * mae_vd #0.1 * vd
    
    c_index = np.arange(205500,len(rta._v1_sma))
    vc = rta._v1_sma[c_index].mean()
    abs_err_vc = np.abs(rta._v1_sma[c_index]-rta._v1[c_index])
    mae_vc = np.mean(abs_err_vc)
    err_vc = 3 * rta._v1_sma[c_index].std() #3 * mae_vc #0.1 * vc
    
    b_index = np.arange(205000, len(rta._v2_sma))
    vb = rta._v2_sma[b_index].mean()
    abs_err_vb = np.abs(rta._v2_sma[b_index] - rta._v2[b_index])
    mae_vb = np.mean(abs_err_vb)  
    err_vb = 3 * rta._v2_sma[b_index].std() #3 * mae_vb #0.1 * vb
    
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
    abs_err_va0 = np.abs(rta._v1_sma[idx0]-rta._v1[idx0])
    mae_va0 = np.mean(abs_err_va0)
    err_va0 = 3 * rta._v1_sma[idx0].std()#3 * mae_va0 # 0.1 * va0
    
    va1 = rta._v1_sma[idx1].mean()
    abs_err_va1 = np.abs(rta._v1_sma[idx1]-rta._v1[idx1])
    mae_va1 = np.mean(abs_err_va1)
    err_va1 = 3 * rta._v1_sma[idx1].std() #3 * mae_va1 #0.1 * va1
    
    va2 = rta._v1_sma[idx2].mean()
    abs_err_va2 = np.abs(rta._v1_sma[idx2]-rta._v1[idx2])
    mae_va2 = np.mean(abs_err_va2)
    err_va2 = 3 * rta._v1_sma[idx2].std() #3 * mae_va2 #0.1 * va2
    
    va3 = rta._v1_sma[idx3].mean()
    abs_err_va3 = np.abs(rta._v1_sma[idx3]-rta._v1[idx3])
    mae_va3 = np.mean(abs_err_va3)
    err_va3 = 3 * rta._v1_sma[idx3].std() #3 * mae_va3 #0.1 * va3
    
    va4 = rta._v1_sma[idx4].mean()
    abs_err_va4 = np.abs(rta._v1_sma[idx4]-rta._v1[idx4])
    mae_va4 = np.mean(abs_err_va4)
    err_va4 = 3 * rta._v1_sma[idx4].std()#3 * mae_va4 #0.1 * va4
    
    va5 = rta._v1_sma[idx5].mean()
    abs_err_va5 = np.abs(rta._v1_sma[idx5]-rta._v1[idx5])
    mae_va5 = np.mean(abs_err_va5)
    err_va5 = 3 * rta._v1_sma[idx5].std() #3 * mae_va5 #0.1 * va5
    
    va6 = rta._v1_sma[idx6].mean()
    abs_err_va6 = np.abs(rta._v1_sma[idx6]-rta._v1[idx6])
    mae_va6 = np.mean(abs_err_va6)
    err_va6 = 3 * rta._v1_sma[idx6].std() #3 * mae_va6 #0.1 * va6
    
    va7 = rta._v1_sma[idx7].mean()
    abs_err_va7 = np.abs(rta._v1_sma[idx7]-rta._v1[idx7])
    mae_va7 = np.mean(abs_err_va7)
    err_va7 = 3 * rta._v1_sma[idx7].std() #3 * mae_va7 #0.1 * va7
    
    va8 = rta._v1_sma[idx8].mean()
    abs_err_va8 = np.abs(rta._v1_sma[idx8]-rta._v1[idx8])
    mae_va8 = np.mean(abs_err_va8)
    err_va8 = 3 * rta._v1_sma[idx8].std() #3 * mae_va8 #0.1 * va8
    
    va9 = rta._v1_sma[idx9].mean()
    abs_err_va9 = np.abs(rta._v1_sma[idx9]-rta._v1[idx9])
    mae_va9 = np.mean(abs_err_va9)
    err_va9 = 3 * rta._v1_sma[idx9].std()#3 * mae_va9 #0.1 * va9
    
    #rta.display_data_indexing()
    rta.display_moving_average_signals()
    # plt.title("Plico data 240906_1231")
    #
    # plt.hlines(vd, 0 , rta._t.max(), ls='-',colors='black')
    # plt.hlines(vd+err_vd, 0 , rta._t.max(), ls='--',colors='black')
    # #plt.hlines(vd-err_vd, 0 , rta._t.max(), ls='--',colors='black')
    #
    # plt.hlines(vc, 0, rta._t.max(), ls='-', colors='cyan')
    # plt.hlines(vc+vc*0.1, 0, rta._t.max(), ls='--', colors='cyan')
    # #plt.hlines(vc+err_vc, 0, rta._t.max(), ls='--', colors='cyan')
    #
    # plt.hlines(vb, 0 , rta._t.max(), ls='-',colors='black')
    # #plt.hlines(vb+err_vb, 0 , rta._t.max(), ls='--',colors='black')
    # plt.hlines(vb-err_vb, 0 , rta._t.max(), ls='--',colors='black')
    #
    # dindx  = 500
    #
    # plt.hlines(va0, rta._t[idx0.min()- dindx], rta._t[idx0.max()+dindx], ls='-',colors='cyan')
    # #plt.hlines(va0+err_va0, rta._t[idx0.min()- dindx], rta._t[idx0.max()+dindx], ls='--',colors='cyan')
    # plt.hlines(va0-err_va0, rta._t[idx0.min()- dindx], rta._t[idx0.max()+dindx], ls='--',colors='cyan')
    #
    # plt.hlines(va1, rta._t[idx1.min()- dindx], rta._t[idx1.max()+dindx], ls='-',colors='cyan')
    # #plt.hlines(va1+err_va1, rta._t[idx1.min()- dindx], rta._t[idx1.max()+dindx], ls='--',colors='cyan')
    # plt.hlines(va1-err_va1, rta._t[idx1.min()- dindx], rta._t[idx1.max()+dindx], ls='--',colors='cyan')
    #
    # plt.hlines(va2, rta._t[idx2.min()- dindx], rta._t[idx2.max()+dindx], ls='-',colors='cyan')
    # #plt.hlines(va2+err_va2, rta._t[idx2.min()- dindx], rta._t[idx2.max()+dindx], ls='--',colors='cyan')
    # plt.hlines(va2-err_va2, rta._t[idx2.min()- dindx], rta._t[idx2.max()+dindx], ls='--',colors='cyan')
    #
    # plt.hlines(va3, rta._t[idx3.min()- dindx], rta._t[idx3.max()+dindx], ls='-',colors='cyan')
    # #plt.hlines(va3+err_va3, rta._t[idx3.min()- dindx], rta._t[idx3.max()+dindx], ls='--',colors='cyan')
    # #plt.hlines(va3-err_va3, rta._t[idx3.min()- dindx], rta._t[idx3.max()+dindx], ls='--',colors='cyan')
    # plt.hlines(0.9*va3, rta._t[idx3.min()- dindx], rta._t[idx3.max()+dindx], ls='--',colors='cyan')
    #
    # plt.hlines(va4, rta._t[idx4.min()- dindx], rta._t[idx4.max()+dindx], ls='-',colors='cyan')
    # #plt.hlines(va4+err_va4, rta._t[idx4.min()- dindx], rta._t[idx4.max()+dindx], ls='--',colors='cyan')
    # plt.hlines(va4-err_va4, rta._t[idx4.min()- dindx], rta._t[idx4.max()+dindx], ls='--',colors='cyan')
    #
    # plt.hlines(va5, rta._t[idx5.min()- dindx], rta._t[idx5.max()+dindx], ls='-',colors='cyan')
    # #plt.hlines(va5+err_va5, rta._t[idx5.min()- dindx], rta._t[idx5.max()+dindx], ls='--',colors='cyan')
    # plt.hlines(va5-err_va5, rta._t[idx5.min()- dindx], rta._t[idx5.max()+dindx], ls='--',colors='cyan')
    #
    # plt.hlines(va6, rta._t[idx6.min()- dindx], rta._t[idx6.max()+dindx], ls='-',colors='cyan')
    # #plt.hlines(va6+err_va6, rta._t[idx6.min()- dindx], rta._t[idx6.max()+dindx], ls='--',colors='cyan')
    # #plt.hlines(va6-err_va6, rta._t[idx6.min()- dindx], rta._t[idx6.max()+dindx], ls='--',colors='cyan')
    # plt.hlines(0.9*va6, rta._t[idx6.min()- dindx], rta._t[idx6.max()+dindx], ls='--',colors='cyan')
    #
    # plt.hlines(va7, rta._t[idx7.min()- dindx], rta._t[idx7.max()+dindx], ls='-',colors='black')
    # #plt.hlines(va7+err_va7, rta._t[idx7.min()- dindx], rta._t[idx7.max()+dindx], ls='--',colors='cyan')
    # plt.hlines(va7-err_va7, rta._t[idx7.min()- dindx], rta._t[idx7.max()+dindx], ls='--',colors='cyan')
    #
    # plt.hlines(va8, rta._t[idx8.min()- dindx], rta._t[idx8.max()+dindx], ls='-',colors='cyan')
    # #plt.hlines(va8+err_va8, rta._t[idx8.min()- dindx], rta._t[idx8.max()+dindx], ls='--',colors='cyan')
    # plt.hlines(va8-err_va8, rta._t[idx8.min()- dindx], rta._t[idx8.max()+dindx], ls='--',colors='cyan')
    #
    # plt.hlines(va9, rta._t[idx9.min()- dindx], rta._t[idx9.max()+dindx], ls='-',colors='cyan')
    # #plt.hlines(va9+err_va9, rta._t[idx9.min()- dindx], rta._t[idx9.max()+dindx], ls='--',colors='cyan')
    # plt.hlines(va9-err_va9, rta._t[idx9.min()- dindx], rta._t[idx9.max()+dindx], ls='--',colors='cyan')
    #

    return (va0, err_va0, mae_va0), (vb, err_vb, mae_vb), (vc, err_vc, mae_vc), (vd, err_vd, mae_vd)
        
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
    
    d_index = np.arange(0, 49355)
    vd = rta._v2_sma[d_index].mean()
    abs_err_vd = np.abs(rta._v2[d_index] - rta._v2_sma[d_index])
    mae_vd = np.mean(abs_err_vd)
    err_vd = 3*rta._v2_sma[d_index].std() #3 * mae_vd #0.1 * vd
    
    c_index = np.arange(54237, len(rta._v1_sma))
    vc = rta._v1_sma[c_index].mean()
    abs_err_vc = np.abs(rta._v1_sma[c_index] - rta._v1[c_index])
    mae_vc = np.mean(abs_err_vc)
    err_vc = 3 * rta._v1_sma[c_index].std()#3 * mae_vc #0.1 * vc
    
    b_index = np.arange(54665, len(rta._v2))
    vb = rta._v2_sma[b_index].mean()
    abs_err_vb = np.abs(rta._v2[b_index] - rta._v2_sma[b_index])
    mae_vb = np.mean(abs_err_vb)
    err_vb = 3 * rta._v2_sma[b_index].std()#3 * mae_vb #0.1 * vb
    
    a_index = np.arange(0, 49402)
    va = rta._v1_sma[a_index].mean()
    abs_err_va = np.abs(rta._v1_sma[a_index] - rta._v1[a_index])
    mae_va = np.mean(abs_err_va)
    err_va = 3 * rta._v1_sma[a_index].std() #3 * mae_va #0.1 * va
    
    
    #rta.display_data_indexing()
    rta.display_moving_average_signals()
    
    plt.title("Blink data 240906_1422 Dwell Time 0ms")
    
    plt.hlines(va, 0, rta._t.max(), ls='-', colors='cyan')
    #plt.hlines(va+err_va, 0, rta._t.max(), ls='--', colors='cyan')
    plt.hlines(va-err_va, 0, rta._t.max(), ls='--', colors='cyan')
    
    plt.hlines(vb, 0, rta._t.max(), ls='-', colors='black')
    #plt.hlines(vb+err_vb, 0, rta._t.max(), ls='--', colors='black')
    plt.hlines(vb-err_vb, 0, rta._t.max(), ls='--', colors='black')
    
    plt.hlines(vc, 0, rta._t.max(), ls='-', colors='cyan')
    plt.hlines(vc+err_vc, 0, rta._t.max(), ls='--', colors='cyan')
    #plt.hlines(vc-err_vc, 0, rta._t.max(), ls='--', colors='cyan')
    
    plt.hlines(vd, 0, rta._t.max(), ls='-', colors='black')
    plt.hlines(vd+err_vd, 0, rta._t.max(), ls='--', colors='black')
    #plt.hlines(vd-err_vd, 0, rta._t.max(), ls='--', colors='black')
    
    return (va, err_va, mae_va), (vb, err_vb, mae_vb), (vc, err_vc, mae_vc), (vd, err_vd, mae_vd)

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
