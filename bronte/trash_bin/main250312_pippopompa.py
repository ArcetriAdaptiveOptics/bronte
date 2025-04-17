# import numpy as np
# from pippo import get_image
# from pippo.real_time_disp import RealTimeDisplay
#
# class ScaoRuanner():
#
#     def __init__(self):
#         pass
#
#
#     def run(self):
#
#         for idx in range(100):
#
#             ima1 = get_image()
#             ima2 = get_image()
#             RealTimeDisplay(ima1,ima2)
#
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore

class RealTimeDisplay(QtWidgets.QMainWindow):
    def __init__(self, title="Real Time Display"):
        """Inizializza la finestra e il plot"""
        super().__init__()
        self.setWindowTitle(title)
        
        # Layout e widget grafico
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        self.graphics_widget = pg.GraphicsLayoutWidget(self)
        layout.addWidget(self.graphics_widget)
        
        self.plot_item = self.graphics_widget.addPlot(title=title)
        self.image_item = pg.ImageItem()
        self.plot_item.addItem(self.image_item)
        self.plot_item.setAspectLocked(True)
        
        # Colorbar
        self.colorbar = pg.HistogramLUTItem()
        self.colorbar.setImageItem(self.image_item)
        self.graphics_widget.addItem(self.colorbar)
        
        # Abilita zoom e pan
        self.plot_item.setMouseEnabled(True, True)
        
        self.show()

    def update_image(self, image):
        """Metodo per aggiornare l'immagine nel plot"""
        self.image_item.setImage(image.T, autoLevels=False)
        self.colorbar.setLevels(np.min(image), np.max(image))

    def close_event(self):
        """Override per chiudere la finestra quando richiesto"""
        self.close()

class ScaoRuanner:
    def __init__(self):
        self.plotter1 = RealTimeDisplay("Plot 1")
        self.plotter2 = RealTimeDisplay("Plot 2")
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])

    def run(self):
        """Ciclo principale per generare immagini e aggiornarle nei plot"""
        
        def update_plots():
            """Aggiorna i plot con nuove immagini"""
            for idx in range(100):
                ima1 = np.random.rand(200, 300)
                ima2 = np.random.rand(200, 300)

                # Aggiorna le finestre con le nuove immagini
                self.plotter1.update_image(ima1)
                self.plotter2.update_image(ima2)

                # Processa gli eventi dell'interfaccia
                QtWidgets.QApplication.processEvents()
                QtCore.QThread.msleep(100)  # Aggiungi una pausa tra gli aggiornamenti

            # Una volta che il ciclo è terminato, chiudi le finestre
            print("Ciclo completato. Chiudo le finestre.")
            self.plotter1.close()
            self.plotter2.close()

        # Avvia l'aggiornamento delle immagini
        QtCore.QTimer.singleShot(0, update_plots)
        
        # Esegui l'applicazione
        self.app.exec_()

# Esegui il programma
if __name__ == "__main__":
    runner = ScaoRuanner()
    runner.run()

