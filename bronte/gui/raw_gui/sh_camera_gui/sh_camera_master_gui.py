import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import time  # Import necessario per gestire il tempo reale
from bronte.gui.raw_gui.psf_camera_gui.strehl_ratio_live_stream import StrehlRatioPlotter
from bronte.utils.camera_master_bkg import CameraMasterMeasurer
from bronte.gui.raw_gui.base_camera_gui import BaseRealTimeCameraDisplay
from bronte.wfs.slope_computer import PCSlopeComputer
from bronte.wfs.subaperture_set import ShSubapertureSet
from bronte.package_data import subaperture_set_folder
from arte.utils.decorator import override

class RealtimeSHWFSDisplay(BaseRealTimeCameraDisplay):
    
    FRAME_SHAPE = (2048, 2048)
    SHWFS_BKG_TAG = '250610_152400'
    SUBAPS_TAG = '250610_140500'#'250120_122000' 
    SH_PIX_THR = 200
    
    def __init__(self,
                sh_cam_name = '193.206.155.69',
                sh_cam_port = 7110,
                wtitle = "Live SHWFS CAMERA",
                ptitle = "Subapertures",
                update_interval = 0):
        
        super().__init__(sh_cam_name, sh_cam_port, wtitle, ptitle, update_interval)
        self._load_subaperture_grid()
        
        # subap grid check box
        self.grid_checkbox = QtWidgets.QCheckBox("Show Subaperture Grid")
        self.grid_checkbox.setChecked(False)  # default is disabled
        self.grid_checkbox.stateChanged.connect(self.update_plot)
        self.layout.addWidget(self.grid_checkbox)
        
        self.show()
    
    def _load_subaperture_grid(self):
        self._subap_map = 0
        if self.SUBAPS_TAG is not None:
            subap_set = ShSubapertureSet.restore(subaperture_set_folder() / (self.SUBAPS_TAG + '.fits'))
            sc = PCSlopeComputer(subap_set)
            self._subap_map = sc.subapertures_map()
    
    @override
    def get_frame2display(self):
        
        frame = self._get_frame_from_camera()
        bkg_sub_frame = self._get_bkg_subtracted_frame(frame)
        
        if self.grid_checkbox.isChecked():
            frame2display = self._subap_map * 1000 + bkg_sub_frame
        else:
            frame2display = bkg_sub_frame
    
        return frame2display
    
    @override
    def _load_camera_master_bkg(self):
        self._master_bkg = np.zeros(self.FRAME_SHAPE)
        if self.SHWFS_BKG_TAG is not None:
            self._master_bkg, self._sh_texp = CameraMasterMeasurer.load_master(self.SHWFS_BKG_TAG, 'shwfs')
            self._camera.setExposureTime(self._sh_texp)
    
    
# run plotter
if __name__ == "__main__":
    plotter = RealtimeSHWFSDisplay()
    plotter.start()
