import sys
#import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import time

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