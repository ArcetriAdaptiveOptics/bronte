from bronte import startup
import pyqtgraph as pg
import numpy as np
from pyqtgraph.Qt import QtWidgets, QtCore

def main():
    
    bf = startup.startup()
    plotter = RealtimePlotter()
    
    for idx in range(100):
        ima = bf.psf_camera.getFutureFrames(1).toNumpyArray()
        plotter.do_plot(ima)
        
    plotter.start()
    plotter.close()
    plotter.app.quit()
    QtWidgets.QApplication.quit()


class RealtimePlotter(QtWidgets.QMainWindow):
    
    def __init__(self):
        super().__init__()
        
        # Creazione dell'applicazione Qt
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])

        # Imposta la finestra principale
        self.setWindowTitle("Plot 2D in Tempo Reale")
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        # Widget PyQtGraph
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graphics_widget)

        # Creazione del plot
        self.plot_item = self.graphics_widget.addPlot(title="PSF CAMERA")
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

        # Mostra la finestra
        self.show()

        # Imposta la colormap iniziale
        self.set_colormap(self.colormaps[self.current_colormap_index])

    def set_colormap(self, colormap_name):
        """Cambia la colormap del plot"""
        colormap = pg.colormap.get(colormap_name)
        self.colorbar.gradient.setColorMap(colormap)

    def do_plot(self, array_2d):
        """Aggiorna il plot con un nuovo array 2D"""
        self.image_item.setImage(array_2d.T, autoLevels=False)
        self.colorbar.setLevels(np.min(array_2d), np.max(array_2d))
        QtWidgets.QApplication.processEvents()  # Mantiene la UI reattiva
        
    def keyPressEvent(self, event):
        """Cambia la colormap premendo un tasto"""
        if event.key() == QtCore.Qt.Key_C:  # Se premi il tasto 'C'
            self.current_colormap_index = (self.current_colormap_index + 1) % len(self.colormaps)
            new_colormap = self.colormaps[self.current_colormap_index]
            #print(f"Colormap cambiata a: {new_colormap}")
            self.set_colormap(new_colormap)

    def start(self):
        """Avvia l'interfaccia grafica"""
        self.app.exec_()
    
    def close(self):
        """Chiude la finestra del plot"""
        super().close()
