from bronte import startup
from bronte.scao.telemetry.scao_telemetry_data_analyser import ScaoTelemetryDataAnalyser
import numpy as np
import matplotlib.pyplot as plt

def main():
    
    subap_tag = '250612_143100'
    slope_offset_tag = '250613_140600'
    rec_tag = '250619_141800'
    telemetry_ftag = '250619_171100'
    
    sf = startup.specula_startup()
    sf.SUBAPS_TAG = subap_tag
    sf.REC_MAT_TAG = rec_tag
    sf.SLOPE_OFFSET_TAG = slope_offset_tag
    
    slope_offset = sf.slope_offset
    rec_mat = sf.reconstructor.recmat.recmat
    c_offset_in_nm = np.dot(rec_mat, slope_offset)
    wf_ref = sf.slm_rasterizer.zernike_coefficients_to_raster(c_offset_in_nm).toNumpyArray()
    
    plt.figure()
    plt.clf()
    plt.plot(slope_offset, label = r'$s_{offset}=IM c_{0}$')
    plt.xlabel('2Nsubap index')
    plt.ylabel('Slopes [normalized]')
    plt.grid('--', alpha=0.3)
    plt.legend(loc='best')
    
    j_noll_vector = np.arange(0, len(c_offset_in_nm))+2
    plt.figure()
    plt.plot(j_noll_vector, c_offset_in_nm, '.-',label = r'$c_{offset}=Rs_{offset}$')
    plt.xlabel('j mode index')
    plt.ylabel('cj nm rms wf')
    plt.grid('--', alpha=0.3)
    plt.legend(loc='best')
    
    plt.figure()
    plt.clf()
    plt.imshow(wf_ref)
    plt.colorbar(label = 'nm rms wf')
    
    stda = ScaoTelemetryDataAnalyser(telemetry_ftag)
    
    return slope_offset, rec_mat, stda