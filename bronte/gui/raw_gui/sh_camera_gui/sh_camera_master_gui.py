import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import time  # Import necessario per gestire il tempo reale
from bronte.gui.raw_gui.psf_camera_gui.strehl_ratio_live_stream import StrehlRatioPlotter
from bronte.utils.camera_master_bkg import CameraMasterMeasurer
from bronte.gui.raw_gui.base_camera_gui import BaseRealTimeCameraDisplay
from arte.utils.decorator import override

class RealtimePsfDisplay(BaseRealTimeCameraDisplay):
    
    FRAME_SHAPE = (2048, 2048)
    SHWFS_BKG_TAG = '250211_135800'
    
    def __init__(self,
                sh_cam_name = '193.206.155.69',
                sh_cam_port = 7110,
                wtitle = "Live SHWFS CAMERA",
                ptitle = "PSF",
                update_interval = 0):
        
        super().__init__(sh_cam_name, sh_cam_port, wtitle, ptitle, update_interval)
    
        # Aggiunto secondo bottone per aprire il grafico dello Strehl Ratio
        self.button_strehl = QtWidgets.QPushButton("Open Strehl Ratio Window")
        self.button_strehl.clicked.connect(self.open_strehl_plot)
        self.layout.addWidget(self.button_strehl)

        self.plot_strehl_window = None  # Finestra del grafico Strehl Ratio
        self.show()
    
    @override
    def _load_camera_master_bkg(self):
        self._master_bkg = np.zeros(self.FRAME_SHAPE)
        if self.PSFCAM_BKG_TAG is not None:
            self._master_bkg, self._pc_texp = CameraMasterMeasurer.load_master(self.SHWFS_BKG_TAG, 'shwfs')
            self._camera.setExposureTime(self._pc_texp)
            
    # def open_strehl_plot(self):
    #     """Apre una nuova finestra con il grafico dello Strehl Ratio."""
    #     if self.plot_strehl_window is None or not self.plot_strehl_window.isVisible():
    #         self.plot_strehl_window = StrehlRatioPlotter(self)
    #         self.plot_strehl_window.show()
    
# Avvia il plotter
if __name__ == "__main__":
    plotter = RealtimePsfDisplay()
    plotter.start()
