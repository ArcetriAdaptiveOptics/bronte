import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import time

# class SlopesPlotterWindow(QtWidgets.QMainWindow):
#
#     def __init__(self, slopes_data_manager, update_interval=1):
#
#         super().__init__()
#         self._data_manager = slopes_data_manager
#         self.setWindowTitle("Slopes Viewer")
#
#         self.plot_widget = pg.PlotWidget(title="Slope Vector")
#         self.slope_curve = self.plot_widget.plot(pen='y')
#         self.setCentralWidget(self.plot_widget)
#
#         self.plot_widget.setLabel("bottom", "2Nsubap [X0,...,XN,Y0,...,YN]")
#         self.plot_widget.setLabel("left", "Slopes")
#
#         self._switch_btw_vector_and_map = False
#
#         self.timer = QtCore.QTimer()
#         self.timer.timeout.connect(self.update_plot)
#         self.timer.start(update_interval)
#         self.show()
#
#
#     def update_plot(self):
#
#         slopes_vector = self._data_manager.get_slope_vector()
#         if slopes_vector is not None:
#             if self._switch_btw_vector_and_map == False:
#                 self.slope_curve.setData(slopes_vector)
#             else:
#                 slope_map_x, slope_map_y = self._data_manager.get_slopes2D()


class SlopesPlotterWindow(QtWidgets.QMainWindow):

    def __init__(self, slopes_data_manager, update_interval=100):
        super().__init__()
        self._data_manager = slopes_data_manager
        self.setWindowTitle("Slopes Vector")
        self._switch_btw_vector_and_map = False
        self._view_range = None
        self._syncing = False 

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        # ======== Widget 1D ========
        self.plot_widget_1d = pg.PlotWidget(title="Slope Vector")
        self.slope_curve = self.plot_widget_1d.plot(pen='y')
        self.plot_widget_1d.setLabel("bottom", "2Nsubap [X0,...,XN,Y0,...,YN]")
        self.plot_widget_1d.setLabel("left", "Slopes")
        self.layout.addWidget(self.plot_widget_1d)

        # ======== Widget 2D ========
        self.widget_2d = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.widget_2d)
        self.widget_2d.hide()

        # Immagine X
        self.label_x = pg.LabelItem("Slope X", size='10pt')
        self.widget_2d.addItem(self.label_x, row=0, col=0)

        self.view_x = self.widget_2d.addViewBox(row=1, col=0)
        self.img_x = pg.ImageItem()
        self.view_x.addItem(self.img_x)
        self.view_x.setAspectLocked(True)
        self.view_x.enableAutoRange(False, False)

        # Immagine Y
        self.widget_2d.nextColumn()
        self.label_y = pg.LabelItem("Slope Y", size='10pt')
        self.widget_2d.addItem(self.label_y, row=0, col=1)

        self.view_y = self.widget_2d.addViewBox(row=1, col=1)
        self.img_y = pg.ImageItem()
        self.view_y.addItem(self.img_y)
        self.view_y.setAspectLocked(True)
        self.view_y.enableAutoRange(False, False)

        # ======= Colorbar condivisa ========
        self.colormap = pg.colormap.get('viridis')  # viridis colormap
        self.img_x.setLookupTable(self.colormap.getLookupTable())
        self.img_y.setLookupTable(self.colormap.getLookupTable())

        self.hist_lut = pg.HistogramLUTItem(image=self.img_x)
        self.hist_lut.setImageItem(self.img_x)
        self.hist_lut.gradient.loadPreset('viridis')
        self.hist_lut.vb.hide()  # Nasconde la parte dell’istogramma
        self.widget_2d.addItem(self.hist_lut, row=0, col=2, rowspan=2)

        # Bottone per cambiare da 1d a 2d
        self.toggle_button = QtWidgets.QPushButton("Switch 1D/2D")
        self.toggle_button.clicked.connect(self._toggle_view)
        self.layout.addWidget(self.toggle_button)

        # Sync tra le due viewbox
        self.view_x.sigRangeChanged.connect(self._sync_view_y)
        self.view_y.sigRangeChanged.connect(self._sync_view_x)

        # Timer aggiornamento
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(update_interval)

        self.show()

    def _sync_view_y(self):
        if self._syncing:
            return
        self._syncing = True
        self.view_y.setRange(rect=self.view_x.viewRect(), padding=0)
        self._syncing = False

    def _sync_view_x(self):
        if self._syncing:
            return
        self._syncing = True
        self.view_x.setRange(rect=self.view_y.viewRect(), padding=0)
        self._syncing = False

    def _toggle_view(self):
        self._switch_btw_vector_and_map = not self._switch_btw_vector_and_map
        if self._switch_btw_vector_and_map:
            self.plot_widget_1d.hide()
            self.widget_2d.show()
        else:
            self.widget_2d.hide()
            self.plot_widget_1d.show()

    def update_plot(self):
        
        if not self._switch_btw_vector_and_map:
            slopes_vector = self._data_manager.get_slope_vector()
            if slopes_vector is not None:
                self.slope_curve.setData(slopes_vector)
        else:
            slope_map_x, slope_map_y = self._data_manager.get_slopes2D()

            if slope_map_x is not None and slope_map_y is not None:
                #combined = np.concatenate([slope_map_x.ravel(), slope_map_y.ravel()])
                #min_val, max_val = np.min(combined), np.max(combined)
                min_val = np.min((slope_map_x.min(), slope_map_y.min()))
                max_val = np.max((slope_map_y.max(), slope_map_y.max()))
                self.img_x.setImage(slope_map_x.T, autoLevels=False)
                self.img_y.setImage(slope_map_y.T, autoLevels=False)
                self.img_x.setLevels([min_val, max_val])
                self.img_y.setLevels([min_val, max_val])
                self.hist_lut.setLevels(min_val, max_val)

                # Imposta range visibile una sola volta
                if self._view_range is None:
                    h, w = slope_map_x.shape
                    self._view_range = [[0, w], [0, h]]
                    self.view_x.setRange(xRange=[0, w], yRange=[0, h], padding=0)
                    self.view_y.setRange(xRange=[0, w], yRange=[0, h], padding=0)