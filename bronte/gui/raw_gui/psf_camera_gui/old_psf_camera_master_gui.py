import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import time  # Import necessario per gestire il tempo reale
from bronte.gui.raw_gui.psf_camera_gui.strehl_ratio_live_stream import StrehlRatioPlotter
from bronte.utils.camera_master_bkg import CameraMasterMeasurer
from pysilico import camera
from bronte.startup import set_data_dir

class RealtimePsfDisplay(QtWidgets.QMainWindow):
    
    PSFCAM_BKG_TAG = '250314_151800'
    FRAME_SHAPE = (1024, 1360)
    CAMERA_PORT = 7100
    CAMERA_NAME = '193.206.155.69'
    
    def __init__(self, title="Live PSF CAMERA", update_interval=0):
        super().__init__()
        set_data_dir()
        self._load_psf_camera()
        self._load_psf_camera_master_bkg()
        
        # Creazione dell'applicazione Qt
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])

        # Imposta la finestra principale
        self.setWindowTitle(title)
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        # Widget PyQtGraph
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graphics_widget)

        # Creazione del plot
        self.plot_item = self.graphics_widget.addPlot(title="PSF")
        self.image_item = pg.ImageItem()
        self.plot_item.addItem(self.image_item)

        # Abilita zoom e pan con il mouse
        self.plot_item.setMouseEnabled(True, True)
        self.plot_item.setAspectLocked(True)

        # Creazione della colorbar
        self.colorbar = pg.HistogramLUTItem()
        self.colorbar.setImageItem(self.image_item)
        self.graphics_widget.addItem(self.colorbar)

        # Lista di colormap disponibili
        self.colormaps = ["viridis", "plasma", "inferno", "magma", "cividis"]
        self.current_colormap_index = 0  # Indice della colormap attuale

        # Imposta la colormap iniziale
        self.set_colormap(self.colormaps[self.current_colormap_index])

        # Timer per aggiornare il plot
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(update_interval)  # Aggiorna ogni 'update_interval' ms
        
        # Aggiunto secondo bottone per aprire il grafico dello Strehl Ratio
        self.button_strehl = QtWidgets.QPushButton("Apri Strehl Ratio")
        self.button_strehl.clicked.connect(self.open_strehl_plot)
        self.layout.addWidget(self.button_strehl)

        self.plot_strehl_window = None  # Finestra del grafico Strehl Ratio
        #self.plot_1d_window = None  # Finestra del grafico 1D
        
        # Mostra la finestra
        self.show()
    
    def _load_psf_camera(self):
        self._psf_camera = camera(self.CAMERA_NAME, self.CAMERA_PORT)
    
    def _load_psf_camera_master_bkg(self):
        self._master_bkg = np.zeros(self.FRAME_SHAPE)
        if self.PSFCAM_BKG_TAG is not None:
            self._master_bkg, self._pc_texp = CameraMasterMeasurer.load_master(self.PSFCAM_BKG_TAG, 'psf_bkg')
            self._psf_camera.setExposureTime(self._pc_texp)
    
    def _get_frame2display(self):
        ima = self._psf_camera.getFutureFrames(1).toNumpyArray() - self._master_bkg
        ima[ima < 0] = 0
        return ima
    def set_colormap(self, colormap_name):
        """Cambia la colormap del plot"""
        colormap = pg.colormap.get(colormap_name)
        self.colorbar.gradient.setColorMap(colormap)

    def update_plot(self):
        """Aggiorna il plot con un nuovo array 2D"""
        self._ima = self._get_frame2display()
        self.image_item.setImage(self._ima.T, autoLevels=False)
        self.colorbar.setLevels(np.min(self._ima), np.max(self._ima))
        QtWidgets.QApplication.processEvents()  # Mantiene la UI reattiva
        
    # def open_1d_plot(self):
    #     """Apre una nuova finestra con il grafico 1D."""
    #     if self.plot_1d_window is None or not self.plot_1d_window.isVisible():
    #         self.plot_1d_window = RealtimePlotter1D()
    #         self.plot_1d_window.show()
    
    def open_strehl_plot(self):
        """Apre una nuova finestra con il grafico dello Strehl Ratio."""
        if self.plot_strehl_window is None or not self.plot_strehl_window.isVisible():
            self.plot_strehl_window = StrehlRatioPlotter(self)
            self.plot_strehl_window.show()

    def keyPressEvent(self, event):
        """Cambia la colormap premendo il tasto 'C'"""
        if event.key() == QtCore.Qt.Key_C:
            self.current_colormap_index = (self.current_colormap_index + 1) % len(self.colormaps)
            new_colormap = self.colormaps[self.current_colormap_index]
            self.set_colormap(new_colormap)

    def start(self):
        """Avvia l'interfaccia grafica"""
        self.app.exec_()
    
    def closeEvent(self, event):
        """Chiude la finestra e ferma il timer"""
        self.timer.stop()
        event.accept()

# Avvia il plotter
if __name__ == "__main__":
    plotter = RealtimePsfDisplay()
    plotter.start()
