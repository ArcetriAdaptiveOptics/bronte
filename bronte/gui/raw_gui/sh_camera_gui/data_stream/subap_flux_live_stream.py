import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import time


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
