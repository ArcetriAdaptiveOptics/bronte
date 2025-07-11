import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
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
            self._slope_vector, self.flux_per_sub_vector  = self._sva.get_slopes_from_frame(
                frame = frame,
                fluxperSub = True)
        except Exception as e:
            print(f"[SlopesDataManager] Error updating data: {e}")
    
    def get_slope_vector(self):
        return self._slope_vector
    
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

class SlopesPlotterWindow(QtWidgets.QMainWindow):
    
    def __init__(self, slopes_data_manager, update_interval=1):
        
        super().__init__()
        self._data_manager = slopes_data_manager
        self.setWindowTitle("Slopes Vector")

        self.plot_widget = pg.PlotWidget(title="Slope Vector")
        self.slope_curve = self.plot_widget.plot(pen='y')
        self.setCentralWidget(self.plot_widget)

        self.plot_widget.setLabel("bottom", "2Nsubap [X0,...,XN,Y0,...,YN]")
        self.plot_widget.setLabel("left", "Slopes")

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(update_interval)
        self.show()

    def update_plot(self):
        
        slopes_vector = self._data_manager.get_slope_vector()
        if slopes_vector is not None:
            self.slope_curve.setData(slopes_vector)

class FluxMapWindow(QtWidgets.QMainWindow):
    
    def __init__(self, slopes_data_manager, update_interval=0):
        
        super().__init__()
        self._data_manager = slopes_data_manager
        self.setWindowTitle("Flux Map")
        self.image_view = pg.ImageView()
        self.setCentralWidget(self.image_view)
        self.resize(800, 600)
        self._view_range = None
        self._hist_window = None
        
        # self.hist_button = QtWidgets.QPushButton("Histogram Flux Window")
        # self.hist_button.clicked.connect(self._open_histogram)
        # self.layout.addWidget(self.hist_button)
        #
        # self.timer = QtCore.QTimer()
        # self.timer.timeout.connect(self.update_flux_map)
        # self.timer.start(update_interval)
        #
        # self.show()

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        layout = QtWidgets.QVBoxLayout()
        central_widget.setLayout(layout)

        # ImageView per la mappa
        self.image_view = pg.ImageView()
        layout.addWidget(self.image_view)

        # Bottone per aprire istogramma
        self.hist_button = QtWidgets.QPushButton("Show Flux Histogram")
        self.hist_button.clicked.connect(self._open_histogram)
        layout.addWidget(self.hist_button)

        # Timer aggiornamento
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_flux_map)
        self.timer.start(update_interval)

        self.show()

    def update_flux_map(self):
        
        flux_map = self._data_manager.get_flux_map()
    
        if flux_map is not None:
            # Se è la prima volta, fai l'autorange
            if self._view_range is None:
                self.image_view.setImage(flux_map.T, autoLevels=True, autoRange=True)
                view = self.image_view.getView()
                if view is not None:
                    self._view_range = view.viewRange()  # Salva lo zoom iniziale
            else:
                # Se la finestra è già aperta e stai aggiornando, NON cambiare zoom
                self.image_view.setImage(flux_map.T, autoLevels=False, autoRange=False)
                
    def _open_histogram(self):
        if self._hist_window is None:
            self._hist_window = FluxHistogramWindow(self._data_manager)
        self._hist_window.show()
        self._hist_window.raise_()
        
class FluxHistogramWindow(QtWidgets.QMainWindow):
    
    def __init__(self, slopes_data_manager, update_interval=1):
        super().__init__()
        self._data_manager = slopes_data_manager
        self.setWindowTitle("Flux Histogram")
        self.resize(500, 400)

        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)
        self.hist_item = pg.PlotCurveItem(pen='g')
        self.plot_widget.addItem(self.hist_item)
        self.plot_widget.setLabel('left', 'Count')
        self.plot_widget.setLabel('bottom', 'Flux Value')

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_histogram)
        self.timer.start(update_interval)

        self.show()

    def update_histogram(self):
        flux_vec = self._data_manager.flux_per_sub_vector
        if flux_vec is not None:
            y, x = np.histogram(flux_vec, bins=200)
            x_centers = (x[:-1] + x[1:]) / 2
            self.hist_item.setData(x_centers, y)
