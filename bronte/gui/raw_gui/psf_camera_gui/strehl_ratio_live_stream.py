import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
#from PySide6 import QtWidgets, QtCore
import pyqtgraph as pg
import time  # Import necessario per gestire il tempo reale
from bronte.utils.raw_strehl_ratio_computer import StrehlRatioComputer



class StrehlRatioPlotter(QtWidgets.QMainWindow):
    """Finestra per il grafico dello Strehl Ratio in tempo reale."""
    
    def __init__(self, gui_master, title="Live Strehl Ratio", update_interval=4):
        super().__init__()
        self.setWindowTitle(title)
        self._gui_master = gui_master
        self._sr_pc = StrehlRatioComputer()
        # TODO: Select a proper dimension for the roi image where to compute SR
        self._roi_hsize = int(3.5*self._sr_pc._dl_size_in_pixels)#20
        # Layout principale
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        # Creazione del plot
        self.plot_widget = pg.PlotWidget(title="Strehl Ratio")
        self.layout.addWidget(self.plot_widget)
        self.curve = self.plot_widget.plot(pen="y")  # Linea blu

        # Configura asse X come tempo relativo
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.setLabel("left", "Strehl Ratio")

        # Timer per aggiornare il plot
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(update_interval)

        # Dati iniziali
        self.start_time = time.time()  # Tempo iniziale
        self.times = [0]  # Lista dei tempi relativi
        self.values = [self.get_strehl_ratio()]  # Lista dei valori Strehl Ratio

        self.show()
    
    def _get_roi(self, image):
        
        ymax = np.where(image == image.max())[0][0]
        xmax = np.where(image == image.max())[1][0]
    
        if(self._gui_master.FRAME_SHAPE[0]//2 - self._roi_hsize < 0 or 
            self._gui_master.FRAME_SHAPE[0]//2 +  self._roi_hsize > self._gui_master.FRAME_SHAPE[0] or 
            self._gui_master.FRAME_SHAPE[1]//2 - self._roi_hsize < 0 or 
            self._gui_master.FRAME_SHAPE[1]//2 +  self._roi_hsize > self._gui_master.FRAME_SHAPE[1]):
            raise ValueError("Warning: ROI extraction out of bounds!")
        
        roi_image = image[ymax - self._roi_hsize : ymax+ self._roi_hsize +1,
                          xmax - self._roi_hsize : xmax + self._roi_hsize + 1]

        return roi_image
    
    def get_strehl_ratio(self):
        """Restituisce un valore scalare simulato di Strehl Ratio."""
        roi_ima = self._get_roi(self._gui_master._ima)
        if roi_ima.size == 0:
            raise ValueError("Warning: ROI image is empty!")
        return self._sr_pc.get_SR_from_image(roi_ima, enable_display = False)

    def update_plot(self):
        """Aggiorna il grafico con il nuovo valore e aggiorna l'asse temporale."""
        current_time = time.time() - self.start_time  # Tempo relativo
        new_value = self.get_strehl_ratio()

        self.times.append(current_time)
        self.values.append(new_value)

        # Mantiene solo gli ultimi 100 valori per evitare troppi dati
        if len(self.times) > 100:
            self.times.pop(0)
            self.values.pop(0)

        self.curve.setData(self.times, self.values)  # Aggiorna il grafico

    def closeEvent(self, event):
        """Ferma il timer quando la finestra viene chiusa."""
        self.timer.stop()
        event.accept()

