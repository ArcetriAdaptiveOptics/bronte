import numpy as np
from bronte.startup import startup
from bronte.package_data import shframes_folder, subaperture_set_folder
from astropy.io import fits
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import copy
import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np

from specula.data_objects.subap_data import SubapData
from specula.data_objects.pixels import Pixels
from specula.processing_objects.sh_slopec import ShSlopec
from specula.data_objects.slopes import Slopes


def execute_tilt_scan(ftag):
    sh_size = 2048
    slm_size_x = 1920
    slm_size_y = 1152 
    slm_response_time = 0.005
    
    factory = startup()
    texp = 10.
    factory.sh_camera.setExposureTime(texp)
    scan_points = 201
    semi_amp_vector = np.linspace(-100e-6, 100e-6, scan_points)
    
    xx = np.linspace(-1, 1, slm_size_x)
    norm_tiltx_cmd = np.tile(xx, slm_size_y)
    
    yy = np.linspace(-1, 1, slm_size_y)
    tilty = np.zeros((slm_size_y, slm_size_x))
    for idx in np.arange(slm_size_x):
        tilty[:,idx]=yy
    norm_tilty_cmd = tilty.reshape(slm_size_y*slm_size_x)
    
    norm_cmd = norm_tilty_cmd
    
    frame_cube = np.zeros((scan_points, sh_size, sh_size))
    
    factory.deformable_mirror.set_shape(np.zeros(slm_size_y*slm_size_x))
    time.sleep(slm_response_time)
    
    for idx, semi_amp in enumerate(semi_amp_vector):
        
        print(f"step{idx}:\t semi_amp = {semi_amp} m") 
        cmd = semi_amp * norm_cmd
        factory.deformable_mirror.set_shape(cmd)
        time.sleep(slm_response_time)
        frame_cube[idx] = factory.sh_camera.getFutureFrames(1).toNumpyArray() 
    
    header = fits.Header()
    header['T_EXP'] = texp
    file_name = shframes_folder() /(ftag + '.fits') 
    fits.writeto(file_name, frame_cube, header)
    fits.append(file_name, semi_amp_vector)
    

def load_data(ftag):
    
    file_name = shframes_folder() /(ftag + '.fits') 
    hdr = fits.getheader(file_name)
    texp = hdr['T_EXP']
    hduList = fits.open(file_name)
    frame_cube = hduList[0].data
    semi_amp_vector = hduList[1].data
    
    return frame_cube, semi_amp_vector, texp

class TiltScannerForSlopesLinearityAnalyzer():
    
    def __init__(self, factory, ftag):
        
        self._frame_cube, self._semi_amp_vec, self._texp = load_data(ftag)
        self._Ntilts = self._semi_amp_vec.shape[0]
        self._ref_frame = self._frame_cube[int((self._Ntilts-1)/2)]
        self._factory = factory
        self._nsub = factory.slope_computer.total_number_of_subapertures()
        self._specula_subap = SubapData.restore_from_bronte(
            subaperture_set_folder() / (self._factory.SUBAPS_TAG + ".fits"))
        self._compute_bronte_slopes_cube()
        self._compute_specula_slopes_cube()
        self._compute_rms_slopes()
    
    def _compute_bronte_slopes_cube(self):
        
        slopes_cube = np.zeros((self._Ntilts, self._nsub, 2))
        
        for idx in np.arange(self._Ntilts):
            
            self._factory.slope_computer.set_frame(self._frame_cube[idx])
            slopes_cube[idx] = self._factory.slope_computer.slopes()
            
        self._bronte_slopes_cube = slopes_cube  
    
    def _compute_specula_slopes_cube(self):
        
        slopes_cube = np.zeros((self._Ntilts, self._nsub, 2))
        
        slopec = ShSlopec(subapdata=self._specula_subap)
        pix = Pixels(dimx = self._ref_frame.shape[1], dimy =self._ref_frame.shape[0] )
        
        for idx in np.arange(self._Ntilts):
            
            pix.pixels = self._frame_cube[idx]
            slopec.inputs['in_pixels'].set(pix)
            slopec.calc_slopes_nofor()
        
            s = copy.copy(slopec.slopes.slopes)
            slopes_cube[idx,:, 0] = Slopes(slopes = s).xslopes#get2d(None, pupdata=self._specula_subap)
            slopes_cube[idx,:, 1] = Slopes(slopes = s).yslopes
        self._specula_slopes_cube = slopes_cube
    
    def display_slopes(self, idx):
        specula_slope_x = self._specula_slopes_cube[idx, :, 0]
        specula_slope_y = self._specula_slopes_cube[idx, :, 1]
        
        bronte_slope_x = self._bronte_slopes_cube[idx, :, 0]
        bronte_slope_y = self._bronte_slopes_cube[idx, :, 1]
        
        fig, axs = plt.subplots(1, 2, sharex = True,
                                 sharey = True)
        
        axs[0].set_title('Slope X')
        axs[0].plot(bronte_slope_x, 'b-', label='bronte')
        axs[0].plot(specula_slope_x, 'r-', label='specula')
        axs[0].legend(loc='best')
        axs[1].set_title('Slope Y')
        axs[1].plot(bronte_slope_y, 'b-')
        axs[1].plot(specula_slope_y, 'r-')
        fig.subplots_adjust(wspace=0.5)
        ptv_um = np.round(2*abs(self._semi_amp_vec[idx]) /1e-6, 3)
        fig.suptitle(f"Slopes: tilt ptv {ptv_um} um")
        fig.tight_layout()
    
    def _compute_rms_slopes(self):
        
        self._rms_slopes_x_bronte = np.zeros(self._Ntilts)
        self._rms_slopes_y_bronte = np.zeros(self._Ntilts)
        self._rms_slopes_x_specula = np.zeros(self._Ntilts)
        self._rms_slopes_y_specula = np.zeros(self._Ntilts)
        
        for idx in np.arange(self._Ntilts):
            self._rms_slopes_x_bronte[idx] = self._bronte_slopes_cube[idx,:,0].std()
            self._rms_slopes_y_bronte[idx] = self._bronte_slopes_cube[idx,:,1].std()    
            self._rms_slopes_x_specula[idx] = self._specula_slopes_cube[idx,:,0].std()
            self._rms_slopes_y_specula[idx] = self._specula_slopes_cube[idx,:,1].std() 
    
    def show_rms_slopes(self, title = 'Tilt'):
        
        fig, axs = plt.subplots(1, 2, sharex = True,
                                 sharey = True)
        
        axs[0].plot(2*self._semi_amp_vec/1e-6, self._rms_slopes_x_bronte, 'b-', label='bronte')
        axs[0].plot(2*self._semi_amp_vec/1e-6, self._rms_slopes_x_specula, 'r-', label='specula')
        axs[0].set_xlabel('ptv [um]')
        axs[0].set_ylabel('rms slopes [au]')
        axs[0].set_title('RMS Slope X')
        axs[0].legend(loc='best')
        
        axs[1].plot(2*self._semi_amp_vec/1e-6, self._rms_slopes_y_bronte, 'b-')
        axs[1].plot(2*self._semi_amp_vec/1e-6, self._rms_slopes_y_specula, 'r-')
        axs[1].set_xlabel('ptv [um]')
        axs[1].set_title('RMS Slope Y')
        fig.subplots_adjust(wspace=0.5)
        
        fig.suptitle(title + " Slopes rms")
            
    def display_specula_slopes(self, idx):
        title = 'Specula Slopes:'
        slopes_cube = self._specula_slopes_cube
        self._display_slopes(idx, slopes_cube, title)
         
    def display_bronte_slopes(self, idx):
        title = 'Bronte Slopes:'
        slopes_cube = self._bronte_slopes_cube
        self._display_slopes(idx, slopes_cube, title)
    
    def _display_slopes(self, idx, slopes_cube, title):
        
        slope_x = slopes_cube[idx, :, 0]
        slope_y = slopes_cube[idx, :, 1]
        
        fig, axs = plt.subplots(1, 2, sharex = True,
                                 sharey = True)
        
        axs[0].set_title('Slope X')
        axs[0].plot(slope_x, 'b-')
        axs[1].set_title('Slope Y')
        axs[1].plot(slope_y, 'r-')
        fig.subplots_adjust(wspace=0.5)
        ptv_um = np.round(2*abs(self._semi_amp_vec[idx]) /1e-6, 3)
        fig.suptitle(title + f"tilt ptv {ptv_um} um")
        fig.tight_layout()