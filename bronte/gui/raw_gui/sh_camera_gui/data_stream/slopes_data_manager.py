import sys
#import numpy as np
from PyQt5 import QtCore#, QtWidgets
#import pyqtgraph as pg
import time

from bronte.utils.slopes_vector_analyser import SlopesVectorAnalyser


class SlopesDataManager(QtCore.QObject):
    
    def __init__(self, gui_master, update_interval=0):
        super().__init__()
        self._gui_master = gui_master
        self._sva = SlopesVectorAnalyser(gui_master.SUBAPS_TAG)
        self._sva.reload_slope_pc(pix_thr_ratio=gui_master.SH_PIX_THR, abs_pix_thr=0)

        self.slope_vector = None
        self.flux_per_sub_vector = None
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_data)
        self.timer.start(update_interval)

    def _update_data(self):
        
        try:
            frame = self._gui_master._bkg_sub_frame
            self.slope_vector, self.flux_per_sub_vector  = self._sva.get_slopes_from_frame(
                frame = frame,
                fluxperSub = True)
        except Exception as e:
            print(f"[SlopesDataManager] Error updating data: {e}")
    
    def get_slope_vector(self):
        
        return self.slope_vector
    
    def get_slopes2D(self,):
        
        return self._sva.get2Dslope_maps_from_slopes_vector(self.slope_vector)
    
    def get_flux_map(self):
        
        if self.flux_per_sub_vector is not None:
            return self._sva.get_2Dflux_map(self.flux_per_sub_vector)
        return None

    def load_reconstructor(self):
        
        try:
            self._sva.load_reconstructor(self._gui_master.REC_TAG)
        except:
            raise ValueError("REC TAG is None")
        
    def get_modal_coefficients(self, slope_vector):
        
        modes = self._sva.get_modes(slope_vector)
        return modes
