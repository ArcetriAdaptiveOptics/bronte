import sys
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
import threading

class RealtimePlotter(QtWidgets.QMainWindow):
    def __init__(self, title="Plot 2D in Tempo Reale"):
        super().__init__()

        # Imposta la finestra principale
        self.setWindowTitle(title)
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        # Crea il widget per il plot
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graphics_widget)

        self.plot_item = self.graphics_widget.addPlot(title=title)
        self.image_item = pg.ImageItem()
        self.plot_item.addItem(self.image_item)

        # Imposta le opzioni del plot
        self.plot_item.setAspectLocked(True)
        self.colorbar = pg.HistogramLUTItem()
        self.colorbar.setImageItem(self.image_item)
        self.graphics_widget.addItem(self.colorbar)

        # Mostra la finestra
        self.show()

    def do_plot(self, array_2d):
        """Aggiorna il plot con il nuovo array 2D"""
        self.image_item.setImage(array_2d.T, autoLevels=False)
        self.colorbar.setLevels(np.min(array_2d), np.max(array_2d))
        QtWidgets.QApplication.processEvents()  # Mantiene la UI reattiva


def update_plot(plotter1, plotter2, ima1_list, ima2_list):
    """Funzione per aggiornare i plot senza bloccare la UI"""
    for ima1, ima2 in zip(ima1_list, ima2_list):
        # Aggiorna i plot in tempo reale
        plotter1.do_plot(ima1)
        plotter2.do_plot(ima2)

        QtCore.QThread.msleep(100)  # Pausa tra i plot (100ms)

    print("Ciclo completato, chiudo GUI.")
    QtWidgets.QApplication.quit()


def main():
    # Crea le finestre per il plot
    plotter1 = RealtimePlotter("Titolo 1")
    plotter2 = RealtimePlotter("Titolo 2")

    # Crea le liste per immagazzinare i dati
    ima1_list = []
    ima2_list = []

    # Genera ima1 e ima2 nel ciclo for nel main
    for _ in range(100):
        ima1 = np.random.rand(200, 300)
        ima2 = np.random.rand(200, 300)
        ima1_list.append(ima1)
        ima2_list.append(ima2)

    # Avvia il ciclo di aggiornamento dei plot in un thread separato
    plot_thread = threading.Thread(target=update_plot, args=(plotter1, plotter2, ima1_list, ima2_list), daemon=True)
    plot_thread.start()

    # Crea e avvia l'applicazione Qt
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    app.exec_()


if __name__ == "__main__":
    main()
