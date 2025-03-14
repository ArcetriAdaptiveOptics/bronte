import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import time  # Import necessario per gestire il tempo reale
from pysilico import camera
from bronte.startup import set_data_dir

class BaseRealTimeCameraDisplay(QtWidgets.QMainWindow):
    
    def __init__(self, camera_name, camera_port, wtitle, ptitle, update_interval=0):
        
        super().__init__()
        set_data_dir()
        self._load_camera(camera_name, camera_port)
        self._load_camera_master_bkg()
        
        # builds the Qt application
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])

        # sets the main window
        self.setWindowTitle(wtitle)
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        # Widget PyQtGraph
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graphics_widget)

        # generate the plot
        self.plot_item = self.graphics_widget.addPlot(title=ptitle)
        self.image_item = pg.ImageItem()
        self.plot_item.addItem(self.image_item)

        # Enable mouse zoom and pan
        self.plot_item.setMouseEnabled(True, True)
        self.plot_item.setAspectLocked(True)

        # generate the colorbar
        self.colorbar = pg.HistogramLUTItem()
        self.colorbar.setImageItem(self.image_item)
        self.graphics_widget.addItem(self.colorbar)

        # List of available colormaps
        self.colormaps = ["viridis", "plasma", "inferno", "magma", "cividis"]
        self.current_colormap_index = 0  

        # set a default colormap
        self.set_colormap(self.colormaps[self.current_colormap_index])

        # Timer to update the plot
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(update_interval)  # update every 'update_interval' ms
        
    
    def _load_camera(self, name, port):
        self._camera = camera(name, port)
    
    def _load_camera_master_bkg(self):
        self._master_bkg = 0
    
    def _get_frame2display(self):
        ima = self._camera.getFutureFrames(1).toNumpyArray() - self._master_bkg
        ima[ima < 0] = 0
        return ima
    
    def set_colormap(self, colormap_name):
        """Changes the colormap"""
        colormap = pg.colormap.get(colormap_name)
        self.colorbar.gradient.setColorMap(colormap)

    def update_plot(self):
        """updates the display with a new image"""
        self._ima = self._get_frame2display()
        self.image_item.setImage(self._ima.T, autoLevels=False)
        self.colorbar.setLevels(np.min(self._ima), np.max(self._ima))
        QtWidgets.QApplication.processEvents()  # Mantiene la UI reattiva
        
    # def open_1d_plot(self):
    #     """opens a new window with a 1D plot."""
    #     if self.plot_1d_window is None or not self.plot_1d_window.isVisible():
    #         self.plot_1d_window = RealtimePlotter1D()
    #         self.plot_1d_window.show()
    
    def keyPressEvent(self, event):
        """Changes colormap pressing 'C'"""
        if event.key() == QtCore.Qt.Key_C:
            self.current_colormap_index = (self.current_colormap_index + 1) % len(self.colormaps)
            new_colormap = self.colormaps[self.current_colormap_index]
            self.set_colormap(new_colormap)

    def start(self):
        """Starts the GUI"""
        self.app.exec_()
    
    def closeEvent(self, event):
        """Closes the window and stops the timer"""
        self.timer.stop()
        event.accept()