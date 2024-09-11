import numpy as np
from bronte.mains.slm_time_response import ResponseTimeAnalyzer

        
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
