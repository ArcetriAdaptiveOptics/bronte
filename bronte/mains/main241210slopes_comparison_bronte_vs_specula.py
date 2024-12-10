import numpy as np 
import matplotlib.pyplot as plt
from bronte.mains import main241203sh_frame_acquisition_for_slopes_comparison
from bronte.startup import startup
from astropy.io import fits
from bronte.utils.data_cube_cleaner import DataCubeCleaner
from mpl_toolkits.axes_grid1 import make_axes_locatable
from bronte.package_data import subaperture_set_folder, reconstructor_folder

import specula
specula.init(-1, precision=1)
from specula import np


from specula.data_objects.subap_data import SubapData
from specula.data_objects.intmat import Intmat
from specula.data_objects.slopes import Slopes


class BronteVsSpeculaSlopeComputer():
    '''
    bkg is the master bkg (master dark sub)
    '''
    
    def __init__(self, tag_name, bkg = None):
        
        self._factory = startup()
        self._raw_frames,\
            self._zc_vec,\
                self._mode_list,\
                    self._texp =\
        main241203sh_frame_acquisition_for_slopes_comparison.load_data(tag_name)
        self._n_slopes = self._factory.slope_computer.total_number_of_subapertures()*2
        if bkg is not None:
            self._bkg = bkg
        else:
            self._bkg = np.zeros((2048,2048))
            
        self._red_frames = self._get_redCube_from_rawCube(
            self._raw_frames, self._bkg)
    
    
    def _display_frame(self, idx, flatsub = False):
        
        if flatsub is True:
            ima =  self._red_frames[idx] - self._red_frames[0]
        else:
            ima = self._red_frames[idx]
        
        plt.figure()
        plt.clf()
        plt.imshow(ima)
        plt.colorbar()
    
    def _compute_slope_offset_with_bronte(self):
        
        offset_frame = self._red_frames[0].astype(float)
        self._factory.slope_computer.set_frame(offset_frame)
        self._slope_offset = self._factory.slope_computer.slopes().reshape(self._n_slopes)
        self._factory.slope_computer.set_slope_offset(self._slope_offset)
        
    def compute_slopes_with_bronte_and_display(self, idx):
        
        self._compute_slope_offset_with_bronte()
        
        sh_frame = self._red_frames[idx].astype(float)
        
        #self._n_slopes = self._factory.slope_computer.total_number_of_subapertures()*2
        
        self._factory.slope_computer.set_frame(sh_frame)
        
        slope_map_x = self._factory.slope_computer.slopes_x_map()
        slope_map_y = self._factory.slope_computer.slopes_y_map()
        
        self._display_slope_maps(slope_map_x, slope_map_y)

    def _display_slope_maps(self, slope_map_x, slope_map_y):
        
        fig, axs = plt.subplots(1, 2, sharex = True,
                                 sharey = True)
        
        axs[0].set_title('Slope Map X')
        im_map_x = axs[0].imshow(slope_map_x)
        # Use make_axes_locatable to create a colorbar of the same height
        divider_x = make_axes_locatable(axs[0])
        cax_x = divider_x.append_axes("right", size="5%", pad=0.15)  # Adjust size and padding
        fig.colorbar(im_map_x, cax=cax_x, label='a.u.')
        
        axs[1].set_title('Slope Map Y')
        im_map_y = axs[1].imshow(slope_map_y)
        
        divider_y = make_axes_locatable(axs[1])
        cax_y = divider_y.append_axes("right", size="5%", pad=0.15)
        fig.colorbar(im_map_y, cax=cax_y, label='a.u.')
        fig.subplots_adjust(wspace=0.5)
        fig.tight_layout()
    
    def compute_slopes_with_specula_and_display(self, idx):
        
        #subapdata = SubapData.restore_from_bronte(subaperture_set_folder() / (self._factory.SUBAPS_TAG + ".fits"))  

        self._im = Intmat.restore(
            reconstructor_folder() / "241203_140300_bronte_im.fits")
        
        subapdata = SubapData.restore_from_bronte(
            subaperture_set_folder() / "241129_162300.fits")
        s = Slopes(slopes = self._im._intmat[idx])
        self._slope_map = s.get2d(cm=None, pupdata=subapdata)
        slope_map_x = self._slope_map[0]
        slope_map_y = self._slope_map[1]
        
        self._display_slope_maps(slope_map_x, slope_map_y)
    
    def _get_redCube_from_rawCube(self, raw_cube, frame2sub):
        
        Nframes = raw_cube.shape[0]
        red_cube = np.zeros(raw_cube.shape)
        
        for idx in range(Nframes):
            
            red_cube[idx] = raw_cube[idx].astype(float) - frame2sub.astype(float)
        
        red_cube[red_cube<0] = 0.
        
        return red_cube
        
        
def main():
    
    fname_dark = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\bronte\\shwfs_frames\\darks\\241210_105900.fits"
    fname_bkgs = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\bronte\\shwfs_frames\\bkgs\\241210_110100.fits"
    
    texp, master_dark = _open_file(fname_dark)
    texp2, raw_bkgs = _open_file(fname_bkgs)
    
    if texp != texp2:
        raise ValueError(
            f"dark and bkgs must have tha same texp: texp_dark: {texp}ms\t texp_bkgs: {texp2}ms")
    master_bkg = DataCubeCleaner.get_master_from_rawCube(raw_bkgs, master_dark)
    # _display_bkg_and_dark(master_dark, master_bkg)
    
    ftag = '241203_190300'
    bvs = BronteVsSpeculaSlopeComputer(ftag, master_bkg)
    
    return bvs

def _display_bkg_and_dark(master_dark, master_bkg):
    
    plt.figure()
    plt.clf()
    plt.imshow(master_bkg)
    plt.colorbar()
    plt.figure()
    plt.clf()
    plt.imshow(master_dark)
    plt.colorbar()
    
def _open_file(fname):
    header = fits.getheader(fname)
    hduList = fits.open(fname)
    texp = header['TEXP_MS']
    data = hduList[0].data
    return texp, data